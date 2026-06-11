// Malha oficial dos municípios de Goiás (IBGE via tbrugz/geodata-br, CC0).
const GEOJSON_URL =
  'https://cdn.jsdelivr.net/gh/tbrugz/geodata-br@master/geojson/geojs-52-mun.json';
const GEOJSON_CACHE_KEY = 'vigia.geojson.go.v1';

// Data de geração do boletim (fixa — é um instantâneo diário).
const GENERATED_AT = new Date('2026-06-04T06:00:00-03:00');

// Janela de previsão sintética: 5 dias.
const WEEKDAYS = ['dom', 'seg', 'ter', 'qua', 'qui', 'sex', 'sáb'];
const MONTHS = ['jan','fev','mar','abr','mai','jun','jul','ago','set','out','nov','dez'];

function buildDays() {
  const out = [];
  const start = new Date('2026-06-04T00:00:00-03:00');
  for (let i = 0; i < 5; i++) {
    const d = new Date(start);
    d.setDate(start.getDate() + i);
    out.push({
      iso: d.toISOString().slice(0, 10),
      weekday: WEEKDAYS[d.getDay()],
      day: d.getDate(),
      month: d.getMonth(),
      label: i === 0 ? 'Hoje' : (i === 1 ? 'Amanhã' : null),
    });
  }
  return out;
}
const DAYS = buildDays();

const MODEL = { accuracy_min: 0.70, accuracy_max: 0.80 };

/* ---- PRNG determinístico (mulberry32) -------------------------------------- */
function seedFromCode(code) {
  let h = 1779033703 ^ String(code).length;
  for (let i = 0; i < String(code).length; i++) {
    h = Math.imul(h ^ String(code).charCodeAt(i), 3432918353);
    h = (h << 13) | (h >>> 19);
  }
  return h >>> 0;
}
function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

/* ---- Centróide aproximado de uma feature ----------------------------------- */
function featureCentroid(feature) {
  let sx = 0, sy = 0, n = 0;
  const geom = feature.geometry;
  if (!geom) return [-49, -16];
  const polys = geom.type === 'Polygon' ? [geom.coordinates] : geom.coordinates;
  for (const poly of polys) {
    const ring = poly[0];
    for (const pt of ring) { sx += pt[0]; sy += pt[1]; n++; }
  }
  return n ? [sx / n, sy / n] : [-49, -16];
}

/* ---- Geração sintética da previsão -----------------------------------------
   Modelo plausível, NÃO real:
   - base de risco cresce para o norte (Cerrado mais seco — Chapada, Niquelândia,
     Cavalcante, Minaçu — historicamente as áreas mais quentes em queimadas);
   - "frente seca" que se desloca pelo território a cada dia (onda espacial),
     fazendo o mapa esquentar/esfriar de um dia para o outro;
   - ruído por município para granularidade.
----------------------------------------------------------------------------- */
function generateForecast(features) {
  // amplitude latitudinal de Goiás (~ -12.4 norte .. -19.5 sul)
  const LAT_N = -12.4, LAT_S = -19.5;
  const muni = {};
  for (const f of features) {
    const code = String(f.properties.id);
    const [lng, lat] = featureCentroid(f);
    const rng = mulberry32(seedFromCode(code));
    const latFactor = clamp((lat - LAT_S) / (LAT_N - LAT_S), 0, 1); // 0 sul → 1 norte
    const base = 0.20 + 0.52 * latFactor;
    const muniNoise = (rng() - 0.5) * 0.34;
    const phase = rng() * Math.PI * 2;
    const risk = [];
    for (let d = 0; d < 5; d++) {
      // frente seca deslocando-se de oeste p/ leste ao longo dos dias
      const front = 0.16 * Math.sin((lng + 49) * 0.55 - d * 0.85 + phase * 0.15);
      const daily = (mulberry32(seedFromCode(code) + d * 99991)() - 0.5) * 0.07;
      risk.push(Number(clamp(base + muniNoise + front + daily, 0.02, 0.99).toFixed(3)));
    }
    muni[code] = {
      code,
      name: f.properties.name,
      center: [lat, lng],
      risk,
    };
  }
  return { generated_at: GENERATED_AT.toISOString(), model: MODEL, days: DAYS, municipios: muni };
}

/* ---- Escala de fogo (verde → âmbar → vermelho) ----------------------------- */
// Paradas em oklch convertidas; usamos uma rampa contínua única.
const RAMP = [
  { t: 0.00, c: [31, 158, 90] },   // calmo  — verde
  { t: 0.35, c: [120, 175, 60] },  // verde-amarelado
  { t: 0.55, c: [232, 167, 53] },  // atenção — âmbar
  { t: 0.75, c: [225, 112, 40] },  // laranja forte
  { t: 1.00, c: [214, 48, 49] },   // alto — vermelho
];
function lerp(a, b, t) { return a + (b - a) * t; }
function riskColor(r) {
  r = clamp(r, 0, 1);
  for (let i = 1; i < RAMP.length; i++) {
    if (r <= RAMP[i].t) {
      const a = RAMP[i - 1], b = RAMP[i];
      const t = (r - a.t) / (b.t - a.t);
      const c = [0, 1, 2].map((k) => Math.round(lerp(a.c[k], b.c[k], t)));
      return `rgb(${c[0]}, ${c[1]}, ${c[2]})`;
    }
  }
  const last = RAMP[RAMP.length - 1].c;
  return `rgb(${last[0]}, ${last[1]}, ${last[2]})`;
}
// Três faixas semânticas (para rótulos textuais).
function riskBucket(r) {
  if (r < 0.40) return { key: 'calmo', label: 'Calmo' };
  if (r < 0.70) return { key: 'atencao', label: 'Atenção' };
  return { key: 'alto', label: 'Alto risco' };
}

function fmtPct(r) { return Math.round(r * 100) + '%'; }
function fmtDateLong(d) {
  return `${String(d.getDate()).padStart(2, '0')}/${String(d.getMonth() + 1).padStart(2, '0')}`;
}

/* ---- Carregamento da malha (com cache em localStorage) --------------------- */
async function loadGeojson() {
  try {
    const cached = localStorage.getItem(GEOJSON_CACHE_KEY);
    if (cached) return JSON.parse(cached);
  } catch (e) { /* ignore */ }
  const res = await fetch(GEOJSON_URL);
  if (!res.ok) throw new Error('Falha ao carregar a malha do IBGE (' + res.status + ')');
  const gj = await res.json();
  try { localStorage.setItem(GEOJSON_CACHE_KEY, JSON.stringify(gj)); } catch (e) { /* quota */ }
  return gj;
}

/* ---- Parse do schema real (gerado_em + dias como array de strings) --------- */
function parseRealForecast(raw, features) {
  const norm = s => s.normalize('NFD').replace(/[\u0300-\u036f]/g, '').toLowerCase().trim();
  const byName = {};
  for (const f of features) byName[norm(f.properties.name)] = f;

  const dias = raw.dias;
  const _nowBR = new Date(new Date().toLocaleString('en-US', { timeZone: 'America/Sao_Paulo' }));
  const _pad = n => String(n).padStart(2, '0');
  const _isoToday = `${_nowBR.getFullYear()}-${_pad(_nowBR.getMonth()+1)}-${_pad(_nowBR.getDate())}`;
  const _tmrw = new Date(_nowBR); _tmrw.setDate(_nowBR.getDate() + 1);
  const _isoTmrw = `${_tmrw.getFullYear()}-${_pad(_tmrw.getMonth()+1)}-${_pad(_tmrw.getDate())}`;
  const days = dias.map(iso => {
    const d = new Date(iso + 'T12:00:00-03:00');
    const label = iso === _isoToday ? 'Hoje' : iso === _isoTmrw ? 'Amanhã' : null;
    return { iso, weekday: WEEKDAYS[d.getDay()], day: d.getDate(), month: d.getMonth(), label };
  });

  // Inicializa todos os municípios do GeoJSON
  const muni = {};
  for (const f of features) {
    const code = String(f.properties.id);
    const [lng, lat] = featureCentroid(f);
    muni[code] = { code, name: f.properties.name, center: [lat, lng], risk: [0, 0, 0, 0, 0] };
  }
  // Preenche probabilidades reais por dia; rastreia quais municípios receberam previsão
  const predicted = new Set();
  dias.forEach((dia, di) => {
    for (const rec of (raw.municipios[dia] || [])) {
      const f = byName[norm(rec.nome)];
      if (!f) continue;
      const code = String(f.properties.id);
      if (muni[code]) { muni[code].risk[di] = Number(rec.prob); predicted.add(code); }
    }
  });
  // Marca municípios sem previsão (ex: 100% Mata Atlântica) como fora do escopo
  for (const code of Object.keys(muni)) {
    if (!predicted.has(code)) muni[code].outOfScope = true;
  }

  // Células por índice de dia (array de 5 listas)
  const celulas = dias.map(dia => raw.celulas[dia] || []);

  return {
    generated_at: new Date(
      raw.gerado_em.includes('T') ? raw.gerado_em : raw.gerado_em + 'T03:00:00-03:00'
    ).toISOString(),
    model: { accuracy_min: 0.816, accuracy_max: 0.816 },
    days,
    municipios: muni,
    celulas,
    source: 'real',
  };
}

/* ---- Carregamento da previsão (real se existir, senão sintética) ----------- */
async function loadForecast(features) {
  // Tenta arquivo real: mesmo dir (symlink), um nível acima, subpasta data/
  for (const url of ['forecast.json', '../forecast.json', 'data/forecast.json']) {
    try {
      const res = await fetch(url, { cache: 'no-store' });
      if (!res.ok) continue;
      const raw = await res.json();

      // Schema real: gerado_em + dias como array de strings ISO
      if (raw.gerado_em && Array.isArray(raw.dias)) {
        return parseRealForecast(raw, features);
      }

      // Schema legado: código IBGE como chave, risk como array de 5 probs
      if (raw.municipios) {
        const byCode = {};
        for (const f of features) byCode[String(f.properties.id)] = f;
        const muni = {};
        for (const [code, rec] of Object.entries(raw.municipios)) {
          const f = byCode[code];
          muni[code] = {
            code,
            name: (f && f.properties.name) || rec.name || code,
            center: f ? featureCentroid(f).slice().reverse() : (rec.center || [-16, -49]),
            risk: rec.risk,
          };
        }
        return {
          generated_at: raw.generated_at || GENERATED_AT.toISOString(),
          model: raw.model || MODEL,
          days: raw.days || DAYS,
          municipios: muni,
          source: 'real',
        };
      }
    } catch (e) { /* tenta próximo */ }
  }
  return { ...generateForecast(features), source: 'synthetic' };
}

window.VIGIA_DATA = {
  GEOJSON_URL, DAYS, MODEL, GENERATED_AT,
  loadGeojson, loadForecast, generateForecast, parseRealForecast,
  riskColor, riskBucket, fmtPct, fmtDateLong, featureCentroid,
  WEEKDAYS, MONTHS,
};
