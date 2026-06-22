/**
 * Memory RAG — SillyTavern Extension
 * Долгосрочная память на основе RAG (Retrieval-Augmented Generation)
 */

import { eventSource, event_types, saveSettingsDebounced } from '../../../../script.js';
import { extension_settings } from '../../../extensions.js';

const EXT_NAME = 'memory-rag';

// ════════════════════════════════════════════════════════════
// DEFAULT SETTINGS
// ════════════════════════════════════════════════════════════

const defaultSettings = {
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
  llmClassifier: false,
  autoExtractEvery: 3,
  minImportanceScore: 0.3,
  topK: 5,
  similarityThreshold: 0.55,
  rankingWeights: { similarity: 0.7, importance: 0.2, recency: 0.1 },
  injectPosition: 'system',
  memoryHeader: '## Memory\n\n',
  maxMemoryTokens: 800,
  debugMode: false,
};

function cfg() { return extension_settings[EXT_NAME]; }

// ════════════════════════════════════════════════════════════
// STYLES
// ════════════════════════════════════════════════════════════

function injectStyles() {
  if (document.getElementById('mrag-styles')) return;
  const el = document.createElement('style');
  el.id = 'mrag-styles';
  el.textContent = `
.mrag-panel{padding:6px 0;font-size:13px;}
.mrag-section{background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.08);border-radius:6px;padding:12px 14px;margin-bottom:8px;}
.mrag-section--actions{display:flex;flex-wrap:wrap;gap:6px;align-items:center;}
.mrag-collapsible{padding:0;}
.mrag-summary{padding:10px 14px;cursor:pointer;user-select:none;font-weight:600;list-style:none;display:flex;align-items:center;}
.mrag-summary::-webkit-details-marker{display:none;}
.mrag-summary::after{content:'▸';margin-left:auto;font-size:11px;transition:transform .15s;}
details[open]>.mrag-summary::after{transform:rotate(90deg);}
.mrag-body{padding:0 14px 12px;}
.mrag-label{display:block;font-size:11px;font-weight:600;letter-spacing:.04em;text-transform:uppercase;opacity:.5;margin:10px 0 4px;}
.mrag-label--main{font-size:14px;font-weight:700;text-transform:none;letter-spacing:0;opacity:1;display:flex;align-items:center;gap:8px;}
.mrag-description,.mrag-hint{font-size:11px;opacity:.45;margin:4px 0 0;line-height:1.5;}
.mrag-badge{font-size:10px;font-weight:700;padding:2px 7px;border-radius:20px;background:#7c6aff;color:#fff;}
.mrag-input,.mrag-select{width:100%;padding:6px 10px;background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.1);border-radius:4px;color:var(--SmartThemeBodyColor,#eee);font-size:13px;outline:none;box-sizing:border-box;transition:border-color .15s;}
.mrag-input:focus,.mrag-select:focus{border-color:rgba(255,255,255,.35);}
.mrag-row{display:flex;align-items:center;gap:8px;}
.mrag-row--spread{justify-content:space-between;}
.mrag-range{-webkit-appearance:none;width:calc(100% - 44px);height:4px;background:rgba(255,255,255,.15);border-radius:2px;outline:none;vertical-align:middle;}
.mrag-range::-webkit-slider-thumb{-webkit-appearance:none;width:14px;height:14px;border-radius:50%;background:#7c6aff;cursor:pointer;}
.mrag-rval{display:inline-block;width:36px;text-align:right;font-size:12px;color:#a78bfa;font-weight:600;vertical-align:middle;}
.mrag-toggle{position:relative;display:inline-block;width:38px;height:20px;flex-shrink:0;}
.mrag-toggle input{opacity:0;width:0;height:0;}
.mrag-slider{position:absolute;inset:0;background:rgba(255,255,255,.15);border-radius:20px;cursor:pointer;transition:background .15s;}
.mrag-slider::before{content:'';position:absolute;width:14px;height:14px;left:3px;bottom:3px;background:#fff;border-radius:50%;transition:transform .15s;}
.mrag-toggle input:checked+.mrag-slider{background:#7c6aff;}
.mrag-toggle input:checked+.mrag-slider::before{transform:translateX(18px);}
.mrag-hidden{display:none!important;}
.mrag-stats-panel{background:rgba(0,0,0,.3);border:1px solid rgba(255,255,255,.08);border-radius:4px;padding:10px;margin-top:6px;font-size:11px;max-height:160px;overflow:auto;}
.mrag-stats-panel pre{margin:0;white-space:pre-wrap;word-break:break-word;}
`;
  document.head.appendChild(el);
}

// ════════════════════════════════════════════════════════════
// SETTINGS PANEL HTML  (injected into #extensions_settings)
// ════════════════════════════════════════════════════════════

function settingsPanelHTML() {
  const s = cfg();
  return `
<div id="mrag-panel" class="mrag-panel">
  <div class="mrag-section">
    <div class="mrag-row mrag-row--spread">
      <span class="mrag-label mrag-label--main">🧠 Memory RAG <span class="mrag-badge" id="mrag-badge">${s.enabled ? 'ON' : 'OFF'}</span></span>
      <label class="mrag-toggle"><input type="checkbox" id="mrag-enabled" ${s.enabled ? 'checked' : ''}><span class="mrag-slider"></span></label>
    </div>
    <p class="mrag-description">Автоматически сохраняет важные события и подключает релевантный контекст к каждому запросу.</p>
  </div>

  <details class="mrag-section mrag-collapsible" open>
    <summary class="mrag-summary">⚙️ Embeddings</summary>
    <div class="mrag-body">
      <label class="mrag-label">Провайдер</label>
      <select id="mrag-provider" class="mrag-select">
        <option value="local"  ${s.embeddingProvider === 'local'  ? 'selected' : ''}>🖥 Local (Transformers.js)</option>
        <option value="openai" ${s.embeddingProvider === 'openai' ? 'selected' : ''}>🤖 OpenAI API</option>
        <option value="custom" ${s.embeddingProvider === 'custom' ? 'selected' : ''}>🔌 Custom API</option>
      </select>
      <div id="mrag-local-opts" class="${s.embeddingProvider === 'local' ? '' : 'mrag-hidden'}">
        <label class="mrag-label">Модель (HuggingFace ID)</label>
        <input type="text" id="mrag-model" class="mrag-input" value="${s.embeddingModel}" placeholder="Xenova/all-MiniLM-L6-v2">
        <p class="mrag-hint">Загружается автоматически. Первый запуск ~20 сек.</p>
      </div>
      <div id="mrag-remote-opts" class="${s.embeddingProvider !== 'local' ? '' : 'mrag-hidden'}">
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
        <option value="json"   ${s.vectorDBType === 'json'   ? 'selected' : ''}>📦 JSON / IndexedDB</option>
        <option value="chroma" ${s.vectorDBType === 'chroma' ? 'selected' : ''}>🔵 ChromaDB</option>
        <option value="qdrant" ${s.vectorDBType === 'qdrant' ? 'selected' : ''}>🟠 Qdrant</option>
      </select>
      <div id="mrag-chroma-opts" class="${s.vectorDBType === 'chroma' ? '' : 'mrag-hidden'}">
        <label class="mrag-label">ChromaDB URL</label>
        <input type="text" id="mrag-chroma-url" class="mrag-input" value="${s.chromaUrl}">
      </div>
      <div id="mrag-qdrant-opts" class="${s.vectorDBType === 'qdrant' ? '' : 'mrag-hidden'}">
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
        <label class="mrag-toggle"><input type="checkbox" id="mrag-llm" ${s.llmClassifier ? 'checked' : ''}><span class="mrag-slider"></span></label>
      </div>
      <p class="mrag-hint">Точнее, но добавляет запрос к LLM каждые N сообщений.</p>
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
      <label class="mrag-label">Веса: схожесть / важность / свежесть</label>
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
        <option value="system"       ${s.injectPosition === 'system'       ? 'selected' : ''}>System prompt</option>
        <option value="after_system" ${s.injectPosition === 'after_system' ? 'selected' : ''}>После system prompt</option>
        <option value="before_chat"  ${s.injectPosition === 'before_chat'  ? 'selected' : ''}>Перед историей чата</option>
      </select>
      <label class="mrag-label">Макс. токенов памяти</label>
      <input type="number" id="mrag-maxtok" class="mrag-input" value="${s.maxMemoryTokens}" min="100" max="2000">
    </div>
  </details>

  <div class="mrag-section mrag-section--actions">
    <button id="mrag-btn-stats"  class="menu_button">📊 Статистика</button>
    <button id="mrag-btn-export" class="menu_button">📤 Экспорт</button>
    <button id="mrag-btn-import" class="menu_button">📥 Импорт</button>
    <button id="mrag-btn-clear"  class="menu_button" style="color:#e05252;">🗑 Очистить</button>
    <div class="mrag-row mrag-row--spread" style="width:100%;margin-top:4px;">
      <span class="mrag-label" style="margin:0">Debug лог</span>
      <label class="mrag-toggle"><input type="checkbox" id="mrag-debug" ${s.debugMode ? 'checked' : ''}><span class="mrag-slider"></span></label>
    </div>
  </div>
  <div id="mrag-stats-out" class="mrag-stats-panel mrag-hidden"></div>
  <input type="file" id="mrag-import-file" accept=".json" style="display:none">
</div>`;
}

// ════════════════════════════════════════════════════════════
// BIND UI EVENTS
// ════════════════════════════════════════════════════════════

function bindEvents() {
  const $ = id => document.getElementById(id);
  const s = cfg();

  const save = (patch) => {
    Object.assign(extension_settings[EXT_NAME], patch);
    saveSettingsDebounced();
  };

  const range = (id, valId, key, nested) => {
    const el = $(id), vl = $(valId);
    if (!el || !vl) return;
    el.addEventListener('input', () => {
      const v = parseFloat(el.value);
      vl.textContent = v;
      if (nested) {
        cfg().rankingWeights[key] = v;
        save({ rankingWeights: { ...cfg().rankingWeights } });
      } else {
        save({ [key]: v });
      }
    });
  };

  $('mrag-enabled')?.addEventListener('change', e => {
    save({ enabled: e.target.checked });
    $('mrag-badge').textContent = e.target.checked ? 'ON' : 'OFF';
  });

  $('mrag-provider')?.addEventListener('change', e => {
    const v = e.target.value;
    save({ embeddingProvider: v });
    $('mrag-local-opts').classList.toggle('mrag-hidden', v !== 'local');
    $('mrag-remote-opts').classList.toggle('mrag-hidden', v === 'local');
    reInitEmbedder();
  });

  $('mrag-model')?.addEventListener('change',      e => { save({ embeddingModel: e.target.value }); reInitEmbedder(); });
  $('mrag-custom-url')?.addEventListener('change', e => save({ customEmbeddingUrl: e.target.value }));
  $('mrag-custom-key')?.addEventListener('change', e => save({ customEmbeddingKey: e.target.value }));

  $('mrag-dbtype')?.addEventListener('change', e => {
    const v = e.target.value;
    save({ vectorDBType: v });
    $('mrag-chroma-opts').classList.toggle('mrag-hidden', v !== 'chroma');
    $('mrag-qdrant-opts').classList.toggle('mrag-hidden', v !== 'qdrant');
    reInitDB();
  });

  $('mrag-chroma-url')?.addEventListener('change', e => save({ chromaUrl: e.target.value }));
  $('mrag-qdrant-url')?.addEventListener('change', e => save({ qdrantUrl: e.target.value }));
  $('mrag-qdrant-key')?.addEventListener('change', e => save({ qdrantApiKey: e.target.value }));
  $('mrag-max-mem')?.addEventListener('change',    e => save({ maxMemories: parseInt(e.target.value) }));
  $('mrag-llm')?.addEventListener('change',        e => save({ llmClassifier: e.target.checked }));

  range('mrag-every',   'mrag-every-v',   'autoExtractEvery');
  range('mrag-min-imp', 'mrag-min-imp-v', 'minImportanceScore');
  range('mrag-topk',    'mrag-topk-v',    'topK');
  range('mrag-thresh',  'mrag-thresh-v',  'similarityThreshold');
  range('mrag-ws',      'mrag-ws-v',      'similarity', true);
  range('mrag-wi',      'mrag-wi-v',      'importance', true);
  range('mrag-wr',      'mrag-wr-v',      'recency',    true);

  $('mrag-pos')?.addEventListener('change',    e => save({ injectPosition: e.target.value }));
  $('mrag-maxtok')?.addEventListener('change', e => save({ maxMemoryTokens: parseInt(e.target.value) }));
  $('mrag-debug')?.addEventListener('change',  e => save({ debugMode: e.target.checked }));

  $('mrag-btn-stats')?.addEventListener('click', () => {
    const out = $('mrag-stats-out');
    out.innerHTML = '<pre>' + JSON.stringify(vectorDB?.getStats() ?? {}, null, 2) + '</pre>';
    out.classList.toggle('mrag-hidden');
  });

  $('mrag-btn-export')?.addEventListener('click', async () => {
    const data = await vectorDB?.exportAll(charId());
    if (!data) return;
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' }));
    a.download = `memory-rag-${Date.now()}.json`;
    a.click();
  });

  $('mrag-btn-import')?.addEventListener('click', () => $('mrag-import-file').click());
  $('mrag-import-file')?.addEventListener('change', async e => {
    const file = e.target.files[0];
    if (!file) return;
    await vectorDB?.importAll(JSON.parse(await file.text()), charId());
    toastr.success('Память импортирована');
  });

  $('mrag-btn-clear')?.addEventListener('click', () => {
    if (confirm('Удалить все воспоминания для текущего персонажа?')) {
      vectorDB?.clear(charId()).then(() => toastr.info('Память очищена'));
    }
  });
}

// ════════════════════════════════════════════════════════════
// MEMORY EXTRACTOR
// ════════════════════════════════════════════════════════════

class MemoryExtractor {
  async extract(messages) {
    const s = cfg();
    if (s.llmClassifier) {
      try { return await this._extractLLM(messages); } catch (e) { log('LLM extraction failed:', e); }
    }
    return this._extractHeuristic(messages);
  }

  async _extractLLM(messages) {
    const conversation = messages
      .filter(m => !m.is_system)
      .map(m => `${m.is_user ? 'User' : 'AI'}: ${m.mes}`)
      .join('\n');

    const prompt = `Extract 1-3 important memory entries from this conversation. Only factual events, character facts, relationship changes, emotional moments.
Respond ONLY with JSON array (no markdown): [{"text":"...","type":"fact|event|emotion|relationship","importance":0.0-1.0}]
If nothing important: []

Conversation:
${conversation}`;

    const res = await fetch('/api/completions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt, max_tokens: 512, temperature: 0.3, stream: false, _skipMemoryRAG: true }),
    });
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const d = await res.json();
    const raw = d.choices?.[0]?.text ?? d.content ?? '';
    try {
      const arr = JSON.parse(raw.replace(/```json|```/g, '').trim());
      return Array.isArray(arr) ? arr.filter(e => e.text && e.type).map(e => ({
        text: e.text.trim(),
        type: this._type(e.type),
        importance: Math.min(1, Math.max(0, e.importance ?? 0.5)),
      })) : [];
    } catch { return []; }
  }

  _extractHeuristic(messages) {
    const results = [];
    for (const msg of messages) {
      if (msg.is_system) continue;
      const score = this.quickScore(msg.mes);
      if (score < cfg().minImportanceScore) continue;
      const sentences = msg.mes.split(/(?<=[.!?])\s+/).map(s => s.trim()).filter(s => s.length > 15);
      for (const s of sentences.slice(0, 2)) {
        if (this.quickScore(s) >= cfg().minImportanceScore)
          results.push({ text: s, type: this.classifyType(s), importance: score });
      }
    }
    return results.slice(0, 3);
  }

  quickScore(text) {
    let score = 0.1;
    const lower = text.toLowerCase();
    const words = text.split(/\s+/).length;
    if (words > 20) score += 0.1;
    if (words > 50) score += 0.1;
    const signals = ['killed','died','found','discovered','betrayed','promised','revealed','love','hate','fear',
      'friend','enemy','named','called','my name','my secret','my past','my goal'];
    score += Math.min(0.4, signals.filter(s => lower.includes(s)).length * 0.07);
    if (/\bI (am|will|won't|can't|must)\b/i.test(text)) score += 0.1;
    if (/\bmy (name|goal|secret|past|family|home)\b/i.test(text)) score += 0.15;
    return Math.min(1.0, score);
  }

  classifyType(text) {
    const l = text.toLowerCase();
    if (/\b(friend|enemy|ally|lover|partner|hate|love|trust|betray)\b/.test(l)) return 'relationship';
    if (/\b(feel|felt|cry|tears|angry|sad|happy|afraid|joy|hurt)\b/.test(l)) return 'emotion';
    if (/\b(killed|died|found|discovered|arrived|attacked|escaped|left|came)\b/.test(l)) return 'event';
    return 'fact';
  }

  _type(raw) {
    return { fact:'fact', event:'event', emotion:'emotion', emotional:'emotion', relationship:'relationship' }[raw?.toLowerCase()] ?? 'fact';
  }
}

// ════════════════════════════════════════════════════════════
// EMBEDDING MODULE
// ════════════════════════════════════════════════════════════

class EmbeddingModule {
  constructor() { this._pipeline = null; this._cache = new Map(); }

  async init() {
    const s = cfg();
    if (s.embeddingProvider !== 'local') return;
    try {
      const { pipeline, env } = await import('https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2/dist/transformers.min.js');
      env.allowLocalModels = false;
      env.allowRemoteModels = true;
      log('Loading embedding model:', s.embeddingModel);
      this._pipeline = await pipeline('feature-extraction', s.embeddingModel, { quantized: true });
      log('Embedding model ready ✓');
    } catch (e) {
      console.error('[MemoryRAG] Model load failed:', e);
      toastr.warning('Memory RAG: не удалось загрузить модель embeddings. Используется заглушка.', 'Memory RAG');
    }
  }

  async embed(text) {
    const key = text.trim().slice(0, 512);
    if (this._cache.has(key)) return this._cache.get(key);
    const s = cfg();
    let v;
    try {
      switch (s.embeddingProvider) {
        case 'local':  v = await this._local(key);  break;
        case 'openai': v = await this._openai(key); break;
        case 'custom': v = await this._custom(key); break;
      }
    } catch (e) {
      log('Embed error:', e);
      v = this._random(s.embeddingDimension ?? 384);
    }
    if (this._cache.size > 200) this._cache.delete(this._cache.keys().next().value);
    this._cache.set(key, v);
    return v;
  }

  async _local(text) {
    if (!this._pipeline) return this._random(cfg().embeddingDimension ?? 384);
    const out = await this._pipeline(text, { pooling: 'mean', normalize: true });
    return Array.from(out.data);
  }

  async _openai(text) {
    const s = cfg();
    const res = await fetch('https://api.openai.com/v1/embeddings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${s.customEmbeddingKey}` },
      body: JSON.stringify({ model: s.embeddingModel || 'text-embedding-3-small', input: text }),
    });
    return (await res.json()).data[0].embedding;
  }

  async _custom(text) {
    const s = cfg();
    const headers = { 'Content-Type': 'application/json' };
    if (s.customEmbeddingKey) headers['Authorization'] = `Bearer ${s.customEmbeddingKey}`;
    const res = await fetch(s.customEmbeddingUrl, { method: 'POST', headers, body: JSON.stringify({ input: text }) });
    const d = await res.json();
    return d.embedding ?? d.data?.[0]?.embedding ?? d;
  }

  _random(dim) { return Array.from({ length: dim }, () => Math.random() * 2 - 1); }

  static cosine(a, b) {
    if (!a?.length || !b?.length || a.length !== b.length) return 0;
    let dot = 0, na = 0, nb = 0;
    for (let i = 0; i < a.length; i++) { dot += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i]; }
    const d = Math.sqrt(na) * Math.sqrt(nb);
    return d === 0 ? 0 : dot / d;
  }
}

// ════════════════════════════════════════════════════════════
// VECTOR DB  (JSON + IndexedDB)
// ════════════════════════════════════════════════════════════

class VectorDB {
  constructor() { this._memories = []; this._key = 'default'; }

  async init(cid) {
    this._key = `mrag:${cid}`;
    this._memories = await this._load() ?? [];
    log('VectorDB loaded:', this._memories.length, 'memories');
  }

  async switchNamespace(cid) {
    await this._save();
    this._key = `mrag:${cid}`;
    this._memories = await this._load() ?? [];
  }

  async addMemory(entry) {
    const s = cfg();
    if (this._memories.length >= s.maxMemories) this._evict();
    this._memories.push(entry);
    await this._save();
  }

  async search(qEmb, topK, threshold) {
    if (!this._memories.length) return [];
    const now = Date.now(), maxAge = 30*24*60*60*1000;
    const w = cfg().rankingWeights;
    return this._memories
      .map(m => {
        const sim = EmbeddingModule.cosine(qEmb, m.embedding);
        if (sim < threshold) return null;
        const recency = 1 - Math.min(1, (now - m.timestamp) / maxAge);
        return { ...m, _score: sim * w.similarity + m.importance * w.importance + recency * w.recency, _sim: sim };
      })
      .filter(Boolean)
      .sort((a, b) => b._score - a._score)
      .slice(0, topK);
  }

  async deleteMemory(id) { this._memories = this._memories.filter(m => m.id !== id); await this._save(); }
  async clear(cid)       { this._memories = []; await this._save(); }

  getStats() {
    const byType = {};
    for (const m of this._memories) byType[m.type] = (byType[m.type] ?? 0) + 1;
    return { total: this._memories.length, byType, backend: 'IndexedDB' };
  }

  async exportAll() { return JSON.parse(JSON.stringify(this._memories)); }
  async importAll(data) { this._memories = data; await this._save(); }

  _evict() {
    this._memories.sort((a, b) => a.importance !== b.importance ? a.importance - b.importance : a.timestamp - b.timestamp);
    this._memories.shift();
  }

  _openDB() {
    return new Promise((res, rej) => {
      const r = indexedDB.open('mrag-db', 1);
      r.onupgradeneeded = e => { if (!e.target.result.objectStoreNames.contains('m')) e.target.result.createObjectStore('m'); };
      r.onsuccess = () => res(r.result);
      r.onerror   = () => rej(r.error);
    });
  }

  async _save() {
    try {
      const db = await this._openDB();
      await new Promise((res, rej) => {
        const tx = db.transaction('m', 'readwrite');
        tx.objectStore('m').put(this._memories, this._key);
        tx.oncomplete = res; tx.onerror = () => rej(tx.error);
      });
    } catch (e) { log('DB save error:', e); }
  }

  async _load() {
    try {
      const db = await this._openDB();
      return await new Promise((res, rej) => {
        const tx = db.transaction('m', 'readonly');
        const r  = tx.objectStore('m').get(this._key);
        r.onsuccess = () => res(r.result ?? []); r.onerror = () => rej(r.error);
      });
    } catch { return []; }
  }
}

// ════════════════════════════════════════════════════════════
// PROMPT INJECTOR
// ════════════════════════════════════════════════════════════

function buildMemoryBlock(memories) {
  const icons = { fact:'📌', event:'⚡', emotion:'💭', relationship:'🔗' };
  const s = cfg();
  const budget = s.maxMemoryTokens * 4;
  let total = 0;
  const lines = [];
  for (const m of memories) {
    const line = `${icons[m.type] ?? '•'} ${m.text}`;
    if (total + line.length > budget) break;
    lines.push(line); total += line.length;
  }
  return lines.length ? s.memoryHeader + lines.join('\n') + '\n' : null;
}

function injectMemories(data, memories) {
  const block = buildMemoryBlock(memories);
  if (!block) return;

  const pos = cfg().injectPosition;

  if (data.messages?.length) {
    if (pos === 'before_chat') {
      const idx = data.messages.findIndex(m => m.role !== 'system');
      if (idx >= 0) { data.messages.splice(idx, 0, { role: 'system', content: block }); return; }
    }
    if (pos === 'after_system') {
      let last = -1;
      for (let i = 0; i < data.messages.length; i++) { if (data.messages[i].role === 'system') last = i; else break; }
      if (last >= 0) { data.messages.splice(last + 1, 0, { role: 'system', content: block }); return; }
    }
    // default: inject into system
    const sys = data.messages.find(m => m.role === 'system');
    if (sys) { sys.content += '\n\n' + block; return; }
    data.messages.unshift({ role: 'system', content: block });
    return;
  }

  if (typeof data.prompt === 'string') { data.prompt = block + '\n' + data.prompt; }
}

// ════════════════════════════════════════════════════════════
// STATE & HELPERS
// ════════════════════════════════════════════════════════════

let extractor = new MemoryExtractor();
let embedder  = new EmbeddingModule();
let vectorDB  = new VectorDB();
let msgCount  = 0;

function log(...a) { if (cfg()?.debugMode) console.log('[MemoryRAG]', ...a); }

function charId() {
  try {
    const ctx = SillyTavern.getContext();
    return String(ctx.characterId ?? ctx.groupId ?? 'default');
  } catch { return 'default'; }
}

function genId() { return `m_${Date.now()}_${Math.random().toString(36).slice(2,8)}`; }

async function reInitEmbedder() {
  embedder = new EmbeddingModule();
  await embedder.init();
}

async function reInitDB() {
  vectorDB = new VectorDB();
  await vectorDB.init(charId());
}

// ════════════════════════════════════════════════════════════
// EVENT HOOKS
// ════════════════════════════════════════════════════════════

async function onBeforePrompt(data) {
  if (!cfg()?.enabled || data._skipMemoryRAG) return;
  try {
    const ctx = SillyTavern.getContext();
    let lastMsg = null;
    for (let i = ctx.chat.length - 1; i >= 0; i--) {
      if (ctx.chat[i].is_user && !ctx.chat[i].is_system) { lastMsg = ctx.chat[i].mes; break; }
    }
    if (!lastMsg) return;
    const qEmb = await embedder.embed(lastMsg);
    const s = cfg();
    const mems = await vectorDB.search(qEmb, s.topK, s.similarityThreshold);
    if (!mems.length) { log('No memories found'); return; }
    injectMemories(data, mems);
    log(`Injected ${mems.length} memories`);
  } catch (e) { console.error('[MemoryRAG] onBeforePrompt:', e); }
}

async function onMessageReceived(msgId) {
  if (!cfg()?.enabled) return;
  try {
    const ctx = SillyTavern.getContext();
    if (!ctx.chat[msgId]) return;
    msgCount++;
    const s = cfg();
    if (msgCount % s.autoExtractEvery !== 0) return;
    const start = Math.max(0, msgId - s.autoExtractEvery + 1);
    const msgs  = ctx.chat.slice(start, msgId + 1);
    const found = await extractor.extract(msgs);
    for (const e of found) {
      if (e.importance < s.minImportanceScore) continue;
      const embedding = await embedder.embed(e.text);
      await vectorDB.addMemory({ id: genId(), text: e.text, embedding, type: e.type, importance: e.importance, timestamp: Date.now() });
      log(`Stored [${e.type}] ${e.importance.toFixed(2)} — ${e.text.slice(0, 60)}`);
    }
  } catch (e) { console.error('[MemoryRAG] onMessageReceived:', e); }
}

async function onChatChanged() {
  try { await vectorDB.switchNamespace(charId()); msgCount = 0; log('Namespace switched:', charId()); }
  catch (e) { console.error('[MemoryRAG] onChatChanged:', e); }
}

// ════════════════════════════════════════════════════════════
// INIT
// ════════════════════════════════════════════════════════════

jQuery(async () => {
  try {
    // Init settings
    if (!extension_settings[EXT_NAME]) extension_settings[EXT_NAME] = structuredClone(defaultSettings);
    for (const [k, v] of Object.entries(defaultSettings)) {
      if (extension_settings[EXT_NAME][k] === undefined) extension_settings[EXT_NAME][k] = v;
    }
    if (!extension_settings[EXT_NAME].rankingWeights) {
      extension_settings[EXT_NAME].rankingWeights = { ...defaultSettings.rankingWeights };
    }

    // Inject styles & panel — same pattern as love-score
    injectStyles();
    $('#extensions_settings').append(settingsPanelHTML());
    bindEvents();

    // Init core modules
    await embedder.init();
    await vectorDB.init(charId());

    // Register hooks
    eventSource.on(event_types.GENERATE_BEFORE_COMBINE_PROMPTS, onBeforePrompt);
    eventSource.on(event_types.MESSAGE_RECEIVED, onMessageReceived);
    eventSource.on(event_types.CHAT_CHANGED, onChatChanged);

    // Public debug API
    window.MemoryRAG = {
      getStats:       ()     => vectorDB.getStats(),
      clearMemories:  ()     => vectorDB.clear(charId()),
      exportMemories: ()     => vectorDB.exportAll(),
      importMemories: (data) => vectorDB.importAll(data),
      searchMemories: async (q) => { const e = await embedder.embed(q); return vectorDB.search(e, 10, 0); },
    };

    log('✅ Initialized');
    console.log('[MemoryRAG] ✅ Ready');
  } catch (e) {
    console.error('[MemoryRAG] Init error:', e);
    try { toastr.error('Memory RAG: ошибка инициализации — ' + e.message); } catch {}
  }
});
