/**
 * SillyTavern Memory RAG Extension v1.0.0
 *
 * Единый файл — SillyTavern загружает JS как обычный <script>,
 * поэтому все модули объединены сюда (без ES import/export).
 *
 * Архитектура:
 *  1. MemoryExtractor  — определяет важность и тип событий
 *  2. EmbeddingModule  — text → vector (Local/OpenAI/Custom)
 *  3. VectorDB         — JSON+IndexedDB | ChromaDB | Qdrant
 *  4. PromptInjector   — вставляет ## Memory в system prompt
 *  5. Hooks            — перехватывает события SillyTavern
 *  6. SettingsUI       — панель настроек в Extensions drawer
 */

(function () {
  'use strict';

  // ════════════════════════════════════════════════════════════
  // 1. MEMORY EXTRACTOR
  // ════════════════════════════════════════════════════════════

  class MemoryExtractor {
    constructor(settings) {
      this.settings = settings;
    }

    updateSettings(s) { this.settings = s; }

    // ── Main entry ──────────────────────────────────────────

    async extract(messages, context) {
      if (this.settings.llmClassifier) {
        try { return await this._extractWithLLM(messages, context); }
        catch (e) {
          console.warn('[MemoryRAG] LLM extraction failed, using heuristics:', e);
        }
      }
      return this._extractHeuristic(messages);
    }

    // ── LLM-based ───────────────────────────────────────────

    async _extractWithLLM(messages, context) {
      const conversation = messages
        .filter(m => !m.is_system)
        .map(m => `${m.is_user ? 'User' : 'AI'}: ${m.mes}`)
        .join('\n');

      const charName = context.name2 ?? 'Character';
      const userName = context.name1 ?? 'User';

      const prompt =
`You are a memory extraction system. Analyze the conversation and extract 1-3 important facts worth remembering.

Only extract: character facts, key plot events, relationship changes, emotionally significant moments.
Skip: trivial chitchat, repeated info, meta-commentary.

Characters: ${userName} (user), ${charName} (AI)

Conversation:
${conversation}

Respond ONLY with valid JSON array (no markdown):
[{"text":"...","type":"fact|event|emotion|relationship","importance":0.0-1.0}]
If nothing important, respond: []`;

      const response = await fetch('/api/completions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt,
          max_tokens: 512,
          temperature: 0.3,
          stream: false,
          _skipMemoryRAG: true,
        }),
      });

      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      const raw = data.choices?.[0]?.text ?? data.content ?? '';

      return this._parseLLMResponse(raw);
    }

    _parseLLMResponse(raw) {
      try {
        const cleaned = raw.replace(/```json|```/g, '').trim();
        const parsed = JSON.parse(cleaned);
        if (!Array.isArray(parsed)) return [];
        return parsed
          .filter(e => e.text && e.type && typeof e.importance === 'number')
          .map(e => ({
            text: e.text.trim(),
            type: this._normalizeType(e.type),
            importance: Math.min(1, Math.max(0, e.importance)),
          }));
      } catch { return []; }
    }

    // ── Heuristic ───────────────────────────────────────────

    _extractHeuristic(messages) {
      const results = [];
      for (const msg of messages) {
        if (msg.is_system) continue;
        const score = this.quickScore(msg.mes, msg.is_user ? 'user' : 'ai');
        if (score < this.settings.minImportanceScore) continue;
        const sentences = this._splitSentences(msg.mes);
        for (const s of sentences.slice(0, 2)) {
          if (this._sentenceIsImportant(s)) {
            results.push({ text: s, type: this.classifyType(s), importance: score });
          }
        }
      }
      return results.slice(0, 3);
    }

    // ── Quick scoring ────────────────────────────────────────

    quickScore(text, role = 'ai') {
      let score = 0.1;
      const lower = text.toLowerCase();
      const words = text.split(/\s+/).length;
      if (words > 20) score += 0.1;
      if (words > 50) score += 0.1;

      const signals = [
        'killed','died','arrived','left','found','discovered','defeated',
        'escaped','entered','attacked','betrayed','promised','decided','revealed',
        'love','hate','fear','trust','friend','enemy','ally','lover','partner',
        'named','called','known as','works as','lives','born','my name',
        'my goal','my secret','my past',
      ];
      const hits = signals.filter(s => lower.includes(s)).length;
      score += Math.min(0.4, hits * 0.06);

      if (/\bI (am|will|won't|can|can't|must|need)\b/i.test(text)) score += 0.1;
      if (/\bmy (name|goal|purpose|secret|past|home|family)\b/i.test(text)) score += 0.15;

      const properNouns = text.match(/(?<!\.\s)\b[A-Z][a-z]{2,}\b/g) ?? [];
      score += Math.min(0.15, properNouns.length * 0.03);

      return Math.min(1.0, score);
    }

    classifyType(text) {
      const lower = text.toLowerCase();
      if (/\b(friend|enemy|ally|lover|partner|rival|hate|love|trust|betray|together)\b/.test(lower)) return 'relationship';
      if (/\b(feel|felt|emotion|heart|cry|tears|angry|sad|happy|afraid|joy)\b/.test(lower)) return 'emotion';
      if (/\b(happened|went|came|left|died|killed|found|discovered|arrived|attacked|escaped)\b/.test(lower)) return 'event';
      return 'fact';
    }

    _sentenceIsImportant(s) {
      return s.split(/\s+/).length >= 5 && this.quickScore(s) >= this.settings.minImportanceScore;
    }

    _splitSentences(text) {
      return text.split(/(?<=[.!?])\s+/).map(s => s.trim()).filter(s => s.length > 10);
    }

    _normalizeType(raw) {
      return { fact:'fact', event:'event', emotion:'emotion', emotional:'emotion', relationship:'relationship' }[raw?.toLowerCase()] ?? 'fact';
    }
  }

  // ════════════════════════════════════════════════════════════
  // 2. EMBEDDING MODULE
  // ════════════════════════════════════════════════════════════

  class EmbeddingModule {
    constructor(settings) {
      this.settings = settings;
      this._pipeline = null;
      this._cache = new Map();
      this._cacheMax = 200;
    }

    async init() {
      if (this.settings.embeddingProvider === 'local') {
        await this._initLocal();
      }
    }

    async _initLocal() {
      try {
        const { pipeline, env } = await import(
          'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2/dist/transformers.min.js'
        );
        env.allowLocalModels = true;
        env.localModelPath = '/models/';
        env.allowRemoteModels = true;
        console.log('[MemoryRAG] Loading model:', this.settings.embeddingModel);
        this._pipeline = await pipeline('feature-extraction', this.settings.embeddingModel, { quantized: true });
        console.log('[MemoryRAG] Model ready ✓');
      } catch (err) {
        console.error('[MemoryRAG] Local model load failed:', err);
        throw err;
      }
    }

    async embed(text) {
      const key = text.trim().slice(0, 512);
      if (this._cache.has(key)) return this._cache.get(key);

      let vector;
      switch (this.settings.embeddingProvider) {
        case 'local':  vector = await this._embedLocal(key);  break;
        case 'openai': vector = await this._embedOpenAI(key); break;
        case 'custom': vector = await this._embedCustom(key); break;
        default: throw new Error('Unknown embedding provider: ' + this.settings.embeddingProvider);
      }

      if (this._cache.size >= this._cacheMax) {
        this._cache.delete(this._cache.keys().next().value);
      }
      this._cache.set(key, vector);
      return vector;
    }

    async _embedLocal(text) {
      if (!this._pipeline) throw new Error('Local model not initialized');
      const out = await this._pipeline(text, { pooling: 'mean', normalize: true });
      return Array.from(out.data);
    }

    async _embedOpenAI(text) {
      if (!this.settings.customEmbeddingKey) throw new Error('OpenAI key missing');
      const res = await fetch('https://api.openai.com/v1/embeddings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${this.settings.customEmbeddingKey}` },
        body: JSON.stringify({ model: this.settings.embeddingModel || 'text-embedding-3-small', input: text }),
      });
      if (!res.ok) throw new Error(`OpenAI ${res.status}`);
      return (await res.json()).data[0].embedding;
    }

    async _embedCustom(text) {
      if (!this.settings.customEmbeddingUrl) throw new Error('Custom URL missing');
      const headers = { 'Content-Type': 'application/json' };
      if (this.settings.customEmbeddingKey) headers['Authorization'] = `Bearer ${this.settings.customEmbeddingKey}`;
      const res = await fetch(this.settings.customEmbeddingUrl, {
        method: 'POST', headers,
        body: JSON.stringify({ input: text }),
      });
      if (!res.ok) throw new Error(`Custom API ${res.status}`);
      const d = await res.json();
      if (Array.isArray(d)) return d;
      if (d.embedding) return d.embedding;
      if (d.data?.[0]?.embedding) return d.data[0].embedding;
      throw new Error('Unknown embedding response format');
    }

    static cosineSimilarity(a, b) {
      if (!a || !b || a.length !== b.length) return 0;
      let dot = 0, na = 0, nb = 0;
      for (let i = 0; i < a.length; i++) { dot += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i]; }
      const denom = Math.sqrt(na) * Math.sqrt(nb);
      return denom === 0 ? 0 : dot / denom;
    }
  }

  // ════════════════════════════════════════════════════════════
  // 3. VECTOR DB  (JSON/IDB + ChromaDB + Qdrant)
  // ════════════════════════════════════════════════════════════

  // ── JSON / IndexedDB adapter ─────────────────────────────

  class JSONAdapter {
    constructor(settings) {
      this.settings = settings;
      this._memories = [];
      this._key = 'default';
    }

    async init(charId) {
      this._key = `memory-rag:${charId}`;
      this._memories = await this._loadIDB(this._key) ?? [];
    }

    async switchNamespace(charId) {
      await this._saveIDB(this._key, this._memories);
      this._key = `memory-rag:${charId}`;
      this._memories = await this._loadIDB(this._key) ?? [];
    }

    async addMemory(entry) {
      if (this._memories.length >= this.settings.maxMemories) this._evict();
      this._memories.push(entry);
      await this._saveIDB(this._key, this._memories);
    }

    async search(queryEmb, topK = 5, threshold = 0.6) {
      if (!this._memories.length) return [];
      const now = Date.now();
      const maxAge = 30 * 24 * 60 * 60 * 1000;
      const w = this.settings.rankingWeights;

      const scored = this._memories
        .map(mem => {
          const sim = EmbeddingModule.cosineSimilarity(queryEmb, mem.embedding);
          if (sim < threshold) return null;
          const recency = 1 - Math.min(1, (now - mem.timestamp) / maxAge);
          const score = sim * w.similarity + mem.importance * w.importance + recency * w.recency;
          return { ...mem, _score: score, _similarity: sim };
        })
        .filter(Boolean)
        .sort((a, b) => b._score - a._score);

      return scored.slice(0, topK);
    }

    async deleteMemory(id) {
      this._memories = this._memories.filter(m => m.id !== id);
      await this._saveIDB(this._key, this._memories);
    }

    async clear(charId) {
      this._memories = [];
      const db = await this._openDB();
      const tx = db.transaction('memories', 'readwrite');
      tx.objectStore('memories').delete(`memory-rag:${charId}`);
    }

    getStats() {
      const byType = {};
      for (const m of this._memories) byType[m.type] = (byType[m.type] ?? 0) + 1;
      return { total: this._memories.length, byType, backend: 'json+idb', key: this._key };
    }

    async exportAll() { return JSON.parse(JSON.stringify(this._memories)); }

    async importAll(data) {
      this._memories = data;
      await this._saveIDB(this._key, this._memories);
    }

    _evict() {
      this._memories.sort((a, b) =>
        a.importance !== b.importance ? a.importance - b.importance : a.timestamp - b.timestamp
      );
      this._memories.shift();
    }

    _openDB() {
      return new Promise((res, rej) => {
        const req = indexedDB.open('memory-rag-db', 1);
        req.onupgradeneeded = e => {
          if (!e.target.result.objectStoreNames.contains('memories'))
            e.target.result.createObjectStore('memories');
        };
        req.onsuccess = () => res(req.result);
        req.onerror  = () => rej(req.error);
      });
    }

    async _saveIDB(key, data) {
      const db = await this._openDB();
      return new Promise((res, rej) => {
        const tx = db.transaction('memories', 'readwrite');
        tx.objectStore('memories').put(data, key);
        tx.oncomplete = res;
        tx.onerror    = () => rej(tx.error);
      });
    }

    async _loadIDB(key) {
      const db = await this._openDB();
      return new Promise((res, rej) => {
        const tx  = db.transaction('memories', 'readonly');
        const req = tx.objectStore('memories').get(key);
        req.onsuccess = () => res(req.result ?? null);
        req.onerror   = () => rej(req.error);
      });
    }
  }

  // ── ChromaDB adapter ─────────────────────────────────────

  class ChromaAdapter {
    constructor(settings) {
      this.settings = settings;
      this._base = settings.chromaUrl.replace(/\/$/, '');
      this._colId   = null;
      this._colName = null;
    }

    async init(charId) {
      this._colName = `memory-rag-${charId}`;
      this._colId   = await this._getOrCreate(this._colName);
    }

    async switchNamespace(charId) {
      this._colName = `memory-rag-${charId}`;
      this._colId   = await this._getOrCreate(this._colName);
    }

    async addMemory({ id, text, embedding, type, importance, timestamp }) {
      await this._req(`/api/v1/collections/${this._colId}/add`, 'POST', {
        ids: [id], embeddings: [embedding], documents: [text],
        metadatas: [{ type, importance, timestamp }],
      });
    }

    async search(queryEmb, topK = 5, threshold = 0.6) {
      const d = await this._req(`/api/v1/collections/${this._colId}/query`, 'POST', {
        query_embeddings: [queryEmb], n_results: topK * 2,
        include: ['documents', 'metadatas', 'distances'],
      });
      const now = Date.now(), maxAge = 30*24*60*60*1000;
      const w   = this.settings.rankingWeights;
      const out = [];
      (d.ids?.[0] ?? []).forEach((id, i) => {
        const sim = 1 - (d.distances[0][i] / 2);
        if (sim < threshold) return;
        const meta    = d.metadatas[0][i];
        const recency = 1 - Math.min(1, (now - meta.timestamp) / maxAge);
        const score   = sim * w.similarity + meta.importance * w.importance + recency * w.recency;
        out.push({ id, text: d.documents[0][i], ...meta, _score: score, _similarity: sim });
      });
      return out.sort((a, b) => b._score - a._score).slice(0, topK);
    }

    async deleteMemory(id) {
      await this._req(`/api/v1/collections/${this._colId}/delete`, 'POST', { ids: [id] });
    }

    async clear(charId) {
      try {
        const col = await this._req(`/api/v1/collections/memory-rag-${charId}`, 'GET');
        await this._req(`/api/v1/collections/${col.id}`, 'DELETE');
      } catch {}
    }

    getStats() { return { backend: 'chroma', collection: this._colName }; }
    async exportAll() { return this._req(`/api/v1/collections/${this._colId}/get`, 'POST', { include: ['documents','metadatas','embeddings'] }); }
    async importAll(data) { if (data.ids?.length) await this._req(`/api/v1/collections/${this._colId}/add`, 'POST', data); }

    async _getOrCreate(name) {
      try { return (await this._req(`/api/v1/collections/${name}`, 'GET')).id; }
      catch { return (await this._req('/api/v1/collections', 'POST', { name, metadata: { 'hnsw:space': 'cosine' } })).id; }
    }

    async _req(path, method = 'GET', body = null) {
      const opts = { method, headers: { 'Content-Type': 'application/json' } };
      if (body) opts.body = JSON.stringify(body);
      const res = await fetch(this._base + path, opts);
      if (!res.ok && method !== 'DELETE') throw new Error(`Chroma ${method} ${path} → ${res.status}`);
      return method === 'DELETE' ? null : res.json();
    }
  }

  // ── Qdrant adapter ───────────────────────────────────────

  class QdrantAdapter {
    constructor(settings) {
      this.settings = settings;
      this._base = settings.qdrantUrl.replace(/\/$/, '');
      this._key  = settings.qdrantApiKey ?? '';
      this._col  = null;
      this._dim  = settings.embeddingDimension ?? 384;
    }

    async init(charId) {
      this._col = `memory_rag_${charId}`.replace(/[^a-zA-Z0-9_]/g, '_');
      await this._ensureCollection();
    }

    async switchNamespace(charId) {
      this._col = `memory_rag_${charId}`.replace(/[^a-zA-Z0-9_]/g, '_');
      await this._ensureCollection();
    }

    async addMemory({ id, text, embedding, type, importance, timestamp }) {
      await this._req(`/collections/${this._col}/points`, 'PUT', {
        points: [{ id: this._hashId(id), vector: embedding, payload: { id, text, type, importance, timestamp } }],
      });
    }

    async search(queryEmb, topK = 5, threshold = 0.6) {
      const d   = await this._req(`/collections/${this._col}/points/search`, 'POST', {
        vector: queryEmb, limit: topK, score_threshold: threshold, with_payload: true,
      });
      const now = Date.now(), maxAge = 30*24*60*60*1000;
      const w   = this.settings.rankingWeights;
      return (d.result ?? []).map(hit => {
        const { id, text, type, importance, timestamp } = hit.payload;
        const sim     = hit.score;
        const recency = 1 - Math.min(1, (now - timestamp) / maxAge);
        const score   = sim * w.similarity + importance * w.importance + recency * w.recency;
        return { id, text, type, importance, timestamp, _score: score, _similarity: sim };
      });
    }

    async deleteMemory(id) {
      await this._req(`/collections/${this._col}/points/delete`, 'POST', { points: [this._hashId(id)] });
    }

    async clear(charId) {
      try { await this._req(`/collections/memory_rag_${charId}`.replace(/[^a-zA-Z0-9_/]/g, '_'), 'DELETE'); } catch {}
    }

    getStats() { return { backend: 'qdrant', collection: this._col }; }
    async exportAll() { return (await this._req(`/collections/${this._col}/points/scroll`, 'POST', { limit: 10000, with_payload: true, with_vector: true })).result?.points ?? []; }
    async importAll(data) {
      if (!data.length) return;
      await this._req(`/collections/${this._col}/points`, 'PUT', {
        points: data.map(p => ({ id: this._hashId(p.payload.id), vector: p.vector, payload: p.payload })),
      });
    }

    async _ensureCollection() {
      try { await this._req(`/collections/${this._col}`, 'GET'); }
      catch { await this._req(`/collections/${this._col}`, 'PUT', { vectors: { size: this._dim, distance: 'Cosine' } }); }
    }

    _hashId(str) {
      let h = 0;
      for (let i = 0; i < str.length; i++) h = ((h << 5) - h + str.charCodeAt(i)) | 0;
      return Math.abs(h);
    }

    async _req(path, method = 'GET', body = null) {
      const headers = { 'Content-Type': 'application/json' };
      if (this._key) headers['api-key'] = this._key;
      const opts = { method, headers };
      if (body) opts.body = JSON.stringify(body);
      const res = await fetch(this._base + path, opts);
      if (!res.ok && res.status !== 404 && method !== 'DELETE') throw new Error(`Qdrant ${method} ${path} → ${res.status}`);
      return method === 'DELETE' ? null : res.json();
    }
  }

  // ── VectorDB facade ──────────────────────────────────────

  class VectorDB {
    constructor(settings) { this.settings = settings; this._a = null; }

    async init(charId) {
      this._a = this.settings.vectorDBType === 'chroma' ? new ChromaAdapter(this.settings)
              : this.settings.vectorDBType === 'qdrant' ? new QdrantAdapter(this.settings)
              : new JSONAdapter(this.settings);
      await this._a.init(charId);
    }

    async switchNamespace(charId) { return this._a.switchNamespace(charId); }
    async addMemory(e)            { return this._a.addMemory(e); }
    async search(q, k, t)         { return this._a.search(q, k, t); }
    async deleteMemory(id)        { return this._a.deleteMemory(id); }
    async clear(charId)           { return this._a.clear(charId); }
    getStats()                    { return this._a.getStats(); }
    async exportAll(charId)       { return this._a.exportAll(charId); }
    async importAll(data, charId) { return this._a.importAll(data, charId); }
  }

  // ════════════════════════════════════════════════════════════
  // 4. PROMPT INJECTOR
  // ════════════════════════════════════════════════════════════

  class PromptInjector {
    constructor(settings) { this.settings = settings; }
    updateSettings(s) { this.settings = s; }

    inject(data, memories) {
      if (!memories?.length) return;
      const block = this._buildBlock(memories);
      if (!block) return;

      switch (this.settings.injectPosition) {
        case 'before_chat':  this._injectBeforeChat(data, block);  break;
        case 'after_system': this._injectAfterSystem(data, block); break;
        default:             this._injectSystem(data, block);
      }
    }

    _buildBlock(memories) {
      const icons = { fact:'📌', event:'⚡', emotion:'💭', relationship:'🔗' };
      const budget = this.settings.maxMemoryTokens * 4;
      let total = 0;
      const lines = [];
      for (const m of memories) {
        const line = `${icons[m.type] ?? '•'} ${m.text}`;
        if (total + line.length > budget) break;
        lines.push(line);
        total += line.length;
      }
      if (!lines.length) return null;
      return `${this.settings.memoryHeader}${lines.join('\n')}\n`;
    }

    _injectSystem(data, block) {
      if (data.messages?.length) {
        const sys = data.messages.find(m => m.role === 'system');
        if (sys) { sys.content += '\n\n' + block; return; }
        data.messages.unshift({ role: 'system', content: block });
        return;
      }
      if (typeof data.prompt === 'string') { data.prompt = block + '\n' + data.prompt; return; }
      data._memoryRAGBlock = block;
    }

    _injectBeforeChat(data, block) {
      if (data.messages?.length) {
        const idx = data.messages.findIndex(m => m.role !== 'system');
        if (idx >= 0) { data.messages.splice(idx, 0, { role: 'system', content: block }); return; }
      }
      this._injectSystem(data, block);
    }

    _injectAfterSystem(data, block) {
      if (data.messages?.length) {
        let last = -1;
        for (let i = 0; i < data.messages.length; i++) {
          if (data.messages[i].role === 'system') last = i; else break;
        }
        if (last >= 0) { data.messages.splice(last + 1, 0, { role: 'system', content: block }); return; }
      }
      this._injectSystem(data, block);
    }

    preview(memories) { return this._buildBlock(memories) ?? '(no memories)'; }
  }

  // ════════════════════════════════════════════════════════════
  // 5. SETTINGS UI
  // ════════════════════════════════════════════════════════════

  const SettingsUI = {
    render(settings, onChange) {
      const el = document.getElementById('memory-rag-settings');
      if (!el) return;
      el.innerHTML = this._html(settings);
      this._bind(el, settings, onChange);
    },

    _html(s) {
      return `
<div class="mrag-section">
  <div class="mrag-row mrag-row--spread">
    <span class="mrag-label mrag-label--main">Memory RAG <span class="mrag-badge" id="mrag-badge">${s.enabled?'ON':'OFF'}</span></span>
    <label class="mrag-toggle"><input type="checkbox" id="mrag-enabled" ${s.enabled?'checked':''}><span class="mrag-slider"></span></label>
  </div>
  <p class="mrag-description">Автоматически сохраняет важные события и подключает релевантный контекст к каждому запросу.</p>
</div>

<details class="mrag-section mrag-collapsible" open>
  <summary class="mrag-summary">⚙️ Embeddings</summary>
  <div class="mrag-body">
    <label class="mrag-label">Провайдер</label>
    <select id="mrag-provider" class="mrag-select">
      <option value="local"  ${s.embeddingProvider==='local' ?'selected':''}>🖥 Local (Transformers.js)</option>
      <option value="openai" ${s.embeddingProvider==='openai'?'selected':''}>🤖 OpenAI API</option>
      <option value="custom" ${s.embeddingProvider==='custom'?'selected':''}>🔌 Custom API</option>
    </select>
    <div id="mrag-local-opts" class="${s.embeddingProvider==='local'?'':'mrag-hidden'}">
      <label class="mrag-label">Модель</label>
      <input type="text" id="mrag-model" class="mrag-input" value="${s.embeddingModel}" placeholder="Xenova/all-MiniLM-L6-v2">
      <p class="mrag-hint">Первая загрузка ~20 сек, затем кэшируется.</p>
    </div>
    <div id="mrag-remote-opts" class="${s.embeddingProvider!=='local'?'':'mrag-hidden'}">
      <label class="mrag-label">API URL</label>
      <input type="text" id="mrag-custom-url" class="mrag-input" value="${s.customEmbeddingUrl}" placeholder="https://api.openai.com/v1/embeddings">
      <label class="mrag-label">API Key</label>
      <input type="password" id="mrag-custom-key" class="mrag-input" value="${s.customEmbeddingKey}" placeholder="sk-...">
    </div>
  </div>
</details>

<details class="mrag-section mrag-collapsible">
  <summary class="mrag-summary">🗄 Vector Database</summary>
  <div class="mrag-body">
    <label class="mrag-label">Бэкенд</label>
    <select id="mrag-dbtype" class="mrag-select">
      <option value="json"   ${s.vectorDBType==='json'  ?'selected':''}>📦 JSON / IndexedDB</option>
      <option value="chroma" ${s.vectorDBType==='chroma'?'selected':''}>🔵 ChromaDB</option>
      <option value="qdrant" ${s.vectorDBType==='qdrant'?'selected':''}>🟠 Qdrant</option>
    </select>
    <div id="mrag-chroma-opts" class="${s.vectorDBType==='chroma'?'':'mrag-hidden'}">
      <label class="mrag-label">ChromaDB URL</label>
      <input type="text" id="mrag-chroma-url" class="mrag-input" value="${s.chromaUrl}">
    </div>
    <div id="mrag-qdrant-opts" class="${s.vectorDBType==='qdrant'?'':'mrag-hidden'}">
      <label class="mrag-label">Qdrant URL</label>
      <input type="text" id="mrag-qdrant-url" class="mrag-input" value="${s.qdrantUrl}">
      <label class="mrag-label">API Key</label>
      <input type="password" id="mrag-qdrant-key" class="mrag-input" value="${s.qdrantApiKey}">
    </div>
    <label class="mrag-label">Макс. записей</label>
    <input type="number" id="mrag-max-mem" class="mrag-input" value="${s.maxMemories}" min="50" max="10000">
  </div>
</details>

<details class="mrag-section mrag-collapsible">
  <summary class="mrag-summary">🧠 Извлечение памяти</summary>
  <div class="mrag-body">
    <div class="mrag-row mrag-row--spread">
      <span class="mrag-label" style="margin:0">LLM-классификация</span>
      <label class="mrag-toggle"><input type="checkbox" id="mrag-llm" ${s.llmClassifier?'checked':''}><span class="mrag-slider"></span></label>
    </div>
    <p class="mrag-hint">Точнее, но добавляет один запрос к LLM каждые N сообщений.</p>
    <label class="mrag-label">Каждые N сообщений</label>
    <div class="mrag-row"><input type="range" id="mrag-every" class="mrag-range" value="${s.autoExtractEvery}" min="1" max="10"><span class="mrag-rval" id="mrag-every-v">${s.autoExtractEvery}</span></div>
    <label class="mrag-label">Мин. важность</label>
    <div class="mrag-row"><input type="range" id="mrag-min-imp" class="mrag-range" value="${s.minImportanceScore}" min="0" max="1" step="0.05"><span class="mrag-rval" id="mrag-min-imp-v">${s.minImportanceScore}</span></div>
  </div>
</details>

<details class="mrag-section mrag-collapsible">
  <summary class="mrag-summary">🔍 Поиск и ранжирование</summary>
  <div class="mrag-body">
    <label class="mrag-label">Воспоминаний в контексте (topK)</label>
    <div class="mrag-row"><input type="range" id="mrag-topk" class="mrag-range" value="${s.topK}" min="1" max="15"><span class="mrag-rval" id="mrag-topk-v">${s.topK}</span></div>
    <label class="mrag-label">Порог схожести</label>
    <div class="mrag-row"><input type="range" id="mrag-thresh" class="mrag-range" value="${s.similarityThreshold}" min="0.1" max="0.99" step="0.05"><span class="mrag-rval" id="mrag-thresh-v">${s.similarityThreshold}</span></div>
    <label class="mrag-label">Вес: схожесть / важность / свежесть</label>
    <div class="mrag-row"><input type="range" id="mrag-ws" class="mrag-range" value="${s.rankingWeights.similarity}" min="0" max="1" step="0.05"><span class="mrag-rval" id="mrag-ws-v">${s.rankingWeights.similarity}</span></div>
    <div class="mrag-row"><input type="range" id="mrag-wi" class="mrag-range" value="${s.rankingWeights.importance}" min="0" max="1" step="0.05"><span class="mrag-rval" id="mrag-wi-v">${s.rankingWeights.importance}</span></div>
    <div class="mrag-row"><input type="range" id="mrag-wr" class="mrag-range" value="${s.rankingWeights.recency}" min="0" max="1" step="0.05"><span class="mrag-rval" id="mrag-wr-v">${s.rankingWeights.recency}</span></div>
  </div>
</details>

<details class="mrag-section mrag-collapsible">
  <summary class="mrag-summary">💉 Инъекция в промпт</summary>
  <div class="mrag-body">
    <label class="mrag-label">Позиция</label>
    <select id="mrag-pos" class="mrag-select">
      <option value="system"       ${s.injectPosition==='system'      ?'selected':''}>System prompt</option>
      <option value="after_system" ${s.injectPosition==='after_system'?'selected':''}>После system prompt</option>
      <option value="before_chat"  ${s.injectPosition==='before_chat' ?'selected':''}>Перед историей чата</option>
    </select>
    <label class="mrag-label">Макс. токенов</label>
    <input type="number" id="mrag-maxtok" class="mrag-input" value="${s.maxMemoryTokens}" min="100" max="2000">
  </div>
</details>

<div class="mrag-section mrag-section--actions">
  <button id="mrag-btn-stats"  class="mrag-btn mrag-btn--secondary">📊 Статистика</button>
  <button id="mrag-btn-export" class="mrag-btn mrag-btn--secondary">📤 Экспорт</button>
  <button id="mrag-btn-import" class="mrag-btn mrag-btn--secondary">📥 Импорт</button>
  <button id="mrag-btn-clear"  class="mrag-btn mrag-btn--danger">🗑 Очистить</button>
  <div class="mrag-row mrag-row--spread" style="width:100%;margin-top:6px">
    <span class="mrag-label" style="margin:0">Debug лог</span>
    <label class="mrag-toggle"><input type="checkbox" id="mrag-debug" ${s.debugMode?'checked':''}><span class="mrag-slider"></span></label>
  </div>
</div>
<div id="mrag-stats-out" class="mrag-stats-panel mrag-hidden"></div>
<input type="file" id="mrag-import-file" accept=".json" style="display:none">
`;
    },

    _bind(el, settings, onChange) {
      const $  = id => el.querySelector('#' + id);
      const on = (id, ev, fn) => $( id)?.addEventListener(ev, fn);
      const range = (id, valId, key, nested) => {
        on(id, 'input', e => {
          const v = parseFloat(e.target.value);
          $(valId).textContent = v;
          if (nested) {
            settings.rankingWeights[key] = v;
            onChange({ rankingWeights: { ...settings.rankingWeights } });
          } else {
            settings[key] = v;
            onChange({ [key]: v });
          }
        });
      };

      on('mrag-enabled', 'change', e => {
        onChange({ enabled: e.target.checked });
        $('mrag-badge').textContent = e.target.checked ? 'ON' : 'OFF';
      });

      on('mrag-provider', 'change', e => {
        const v = e.target.value;
        onChange({ embeddingProvider: v });
        $('mrag-local-opts').classList.toggle('mrag-hidden', v !== 'local');
        $('mrag-remote-opts').classList.toggle('mrag-hidden', v === 'local');
      });

      on('mrag-model',      'change', e => onChange({ embeddingModel: e.target.value }));
      on('mrag-custom-url', 'change', e => onChange({ customEmbeddingUrl: e.target.value }));
      on('mrag-custom-key', 'change', e => onChange({ customEmbeddingKey: e.target.value }));

      on('mrag-dbtype', 'change', e => {
        const v = e.target.value;
        onChange({ vectorDBType: v });
        $('mrag-chroma-opts').classList.toggle('mrag-hidden', v !== 'chroma');
        $('mrag-qdrant-opts').classList.toggle('mrag-hidden', v !== 'qdrant');
      });

      on('mrag-chroma-url', 'change', e => onChange({ chromaUrl: e.target.value }));
      on('mrag-qdrant-url', 'change', e => onChange({ qdrantUrl: e.target.value }));
      on('mrag-qdrant-key', 'change', e => onChange({ qdrantApiKey: e.target.value }));
      on('mrag-max-mem',    'change', e => onChange({ maxMemories: parseInt(e.target.value) }));
      on('mrag-llm',        'change', e => onChange({ llmClassifier: e.target.checked }));

      range('mrag-every',   'mrag-every-v',   'autoExtractEvery');
      range('mrag-min-imp', 'mrag-min-imp-v', 'minImportanceScore');
      range('mrag-topk',    'mrag-topk-v',    'topK');
      range('mrag-thresh',  'mrag-thresh-v',  'similarityThreshold');
      range('mrag-ws',      'mrag-ws-v',      'similarity', true);
      range('mrag-wi',      'mrag-wi-v',      'importance', true);
      range('mrag-wr',      'mrag-wr-v',      'recency',    true);

      on('mrag-pos',    'change', e => onChange({ injectPosition: e.target.value }));
      on('mrag-maxtok', 'change', e => onChange({ maxMemoryTokens: parseInt(e.target.value) }));
      on('mrag-debug',  'change', e => onChange({ debugMode: e.target.checked }));

      on('mrag-btn-stats', 'click', () => {
        const out = $('mrag-stats-out');
        out.innerHTML = '<pre>' + JSON.stringify(window.MemoryRAG?.getStats() ?? {}, null, 2) + '</pre>';
        out.classList.toggle('mrag-hidden');
      });

      on('mrag-btn-export', 'click', async () => {
        const data = await window.MemoryRAG?.exportMemories();
        if (!data) return;
        const a = Object.assign(document.createElement('a'), {
          href: URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })),
          download: `memory-rag-${Date.now()}.json`,
        });
        a.click();
      });

      on('mrag-btn-import', 'click', () => $('mrag-import-file').click());
      on('mrag-import-file', 'change', async e => {
        const file = e.target.files[0];
        if (!file) return;
        await window.MemoryRAG?.importMemories(JSON.parse(await file.text()));
        toastr.success('Память импортирована');
      });

      on('mrag-btn-clear', 'click', () => {
        if (confirm('Удалить все воспоминания для текущего персонажа?'))
          window.MemoryRAG?.clearMemories().then(() => toastr.info('Память очищена'));
      });
    },
  };

  // ════════════════════════════════════════════════════════════
  // 6. MAIN ORCHESTRATOR
  // ════════════════════════════════════════════════════════════

  const EXTENSION_NAME = 'memory-rag';

  const DEFAULT_SETTINGS = {
    enabled: true,
    embeddingProvider: 'local',
    embeddingModel: 'Xenova/all-MiniLM-L6-v2',
    customEmbeddingUrl: '',
    customEmbeddingKey: '',
    embeddingDimension: 384,
    vectorDBType: 'json',
    chromaUrl: 'http://localhost:8000',
    qdrantUrl: 'http://localhost:6333',
    qdrantApiKey: '',
    maxMemories: 1000,
    llmClassifier: true,
    autoExtractEvery: 3,
    minImportanceScore: 0.3,
    topK: 5,
    similarityThreshold: 0.6,
    rankingWeights: { similarity: 0.7, importance: 0.2, recency: 0.1 },
    injectPosition: 'system',
    memoryHeader: '## Memory\n\n',
    maxMemoryTokens: 800,
    debugMode: false,
  };

  let settings     = { ...DEFAULT_SETTINGS };
  let extractor    = null;
  let embedder     = null;
  let vectorDB     = null;
  let injector     = null;
  let msgCounter   = 0;
  let initialized  = false;

  // ── Helpers ──────────────────────────────────────────────

  function log(...a) { if (settings.debugMode) console.log('[MemoryRAG]', ...a); }

  function charId() {
    try {
      const ctx = SillyTavern.getContext();
      return String(ctx.characterId ?? ctx.groupId ?? 'default');
    } catch { return 'default'; }
  }

  function lastUserMsg() {
    const { chat } = SillyTavern.getContext();
    for (let i = chat.length - 1; i >= 0; i--)
      if (chat[i].is_user && !chat[i].is_system) return chat[i].mes;
    return null;
  }

  function genId() {
    return `mem_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`;
  }

  // ── Settings ─────────────────────────────────────────────

  function loadSettings() {
    const saved = extension_settings[EXTENSION_NAME] ?? {};
    settings = { ...DEFAULT_SETTINGS, ...saved, rankingWeights: { ...DEFAULT_SETTINGS.rankingWeights, ...(saved.rankingWeights ?? {}) } };
  }

  function saveSettings() {
    extension_settings[EXTENSION_NAME] = { ...settings };
    saveSettingsDebounced();
  }

  async function onSettingsChange(patch) {
    const prevProv = settings.embeddingProvider;
    const prevDB   = settings.vectorDBType;
    Object.assign(settings, patch);
    if (patch.rankingWeights) settings.rankingWeights = { ...settings.rankingWeights, ...patch.rankingWeights };
    saveSettings();

    if (patch.embeddingProvider !== undefined && patch.embeddingProvider !== prevProv ||
        patch.vectorDBType      !== undefined && patch.vectorDBType      !== prevDB) {
      await reInit();
    } else {
      extractor?.updateSettings(settings);
      injector?.updateSettings(settings);
    }
  }

  async function reInit() {
    initialized = false;
    extractor = new MemoryExtractor(settings);
    embedder  = new EmbeddingModule(settings);
    vectorDB  = new VectorDB(settings);
    injector  = new PromptInjector(settings);
    await embedder.init();
    await vectorDB.init(charId());
    initialized = true;
    log('Re-initialized');
  }

  // ── Memory pipeline ──────────────────────────────────────

  async function extractAndStore(context, lastIdx) {
    const start = Math.max(0, lastIdx - settings.autoExtractEvery + 1);
    const msgs  = context.chat.slice(start, lastIdx + 1);
    const found = await extractor.extract(msgs, context);

    for (const e of found) {
      if (e.importance < settings.minImportanceScore) continue;
      const embedding = await embedder.embed(e.text);
      await vectorDB.addMemory({
        id: genId(), text: e.text, embedding,
        type: e.type, importance: e.importance,
        timestamp: Date.now(),
        characterId: charId(),
        chatId: context.chatId ?? 'unknown',
      });
      log(`Stored [${e.type}] ${e.importance.toFixed(2)} — ${e.text.slice(0, 60)}`);
    }
  }

  // ── Hooks ────────────────────────────────────────────────

  async function onBeforePrompt(data) {
    if (!initialized || !settings.enabled) return;
    try {
      const msg = lastUserMsg();
      if (!msg) return;
      const qEmb = await embedder.embed(msg);
      const mems = await vectorDB.search(qEmb, settings.topK, settings.similarityThreshold);
      if (!mems.length) { log('No memories found'); return; }
      injector.inject(data, mems);
      log(`Injected ${mems.length} memories`);
    } catch (e) { console.error('[MemoryRAG] onBeforePrompt:', e); }
  }

  async function onMessageReceived(msgId) {
    if (!initialized || !settings.enabled) return;
    try {
      const ctx = SillyTavern.getContext();
      if (!ctx.chat[msgId]) return;
      msgCounter++;
      if (msgCounter % settings.autoExtractEvery === 0) {
        await extractAndStore(ctx, msgId);
      }
    } catch (e) { console.error('[MemoryRAG] onMessageReceived:', e); }
  }

  async function onMessageSent(msgId) {
    if (!initialized || !settings.enabled) return;
    try {
      const ctx = SillyTavern.getContext();
      const msg = ctx.chat[msgId];
      if (!msg || msg.is_system) return;
      const score = extractor.quickScore(msg.mes, 'user');
      if (score < settings.minImportanceScore) return;
      const embedding = await embedder.embed(msg.mes);
      await vectorDB.addMemory({
        id: genId(), text: msg.mes, embedding,
        type: extractor.classifyType(msg.mes), importance: score,
        timestamp: Date.now(), characterId: charId(), chatId: ctx.chatId ?? 'unknown',
      });
    } catch (e) { console.error('[MemoryRAG] onMessageSent:', e); }
  }

  async function onChatChanged() {
    if (!initialized) return;
    try {
      await vectorDB.switchNamespace(charId());
      msgCounter = 0;
      log('Switched namespace:', charId());
    } catch (e) { console.error('[MemoryRAG] onChatChanged:', e); }
  }

  // ── Init ─────────────────────────────────────────────────

  async function init() {
    try {
      loadSettings();

      extractor = new MemoryExtractor(settings);
      embedder  = new EmbeddingModule(settings);
      vectorDB  = new VectorDB(settings);
      injector  = new PromptInjector(settings);

      await embedder.init();
      await vectorDB.init(charId());

      const { eventSource, event_types, saveSettingsDebounced: ssd } = SillyTavern.getContext();
      window.saveSettingsDebounced = ssd;

      eventSource.on(event_types.GENERATE_BEFORE_COMBINE_PROMPTS, onBeforePrompt);
      eventSource.on(event_types.MESSAGE_RECEIVED,                onMessageReceived);
      eventSource.on(event_types.MESSAGE_SENT,                    onMessageSent);
      eventSource.on(event_types.CHAT_CHANGED,                    onChatChanged);

      SettingsUI.render(settings, onSettingsChange);

      initialized = true;
      console.log('[MemoryRAG] ✅ Initialized');

      // Public debug API
      window.MemoryRAG = {
        getStats:       ()     => vectorDB.getStats(),
        clearMemories:  ()     => vectorDB.clear(charId()),
        exportMemories: ()     => vectorDB.exportAll(charId()),
        importMemories: (data) => vectorDB.importAll(data, charId()),
        searchMemories: async (q) => {
          const emb = await embedder.embed(q);
          return vectorDB.search(emb, 10, 0.0);
        },
      };
    } catch (err) {
      console.error('[MemoryRAG] Init failed:', err);
    }
  }

  // ── Boot ─────────────────────────────────────────────────
  jQuery(async () => { await init(); });

})();
