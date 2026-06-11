/* =============================================================================
   vigIA — lógica do painel (Leaflet + interações)
============================================================================= */
(function () {
  const D = window.VIGIA_DATA;
  const GO_CENTER = [-16.0, -49.6];

  const state = {
    geojson: null,
    forecast: null,
    features: [],
    day: Number(localStorage.getItem('vigia.day') || 0),
    selected: null,      // código do município em drill-in
    expanded: false,
    drilling: false,     // true quando E2 está ativo (E1 some do mapa)
    rankTab: 'e1',       // 'e1' | 'e2'
    minProb: 0,          // limiar das células E2 (0–90)
    map: null,
    muniLayer: null,
    cellLayer: null,
    labelLayer: null,
    layerByCode: {},
  };
  if (!(state.day >= 0 && state.day < 5)) state.day = 0;

  const $ = (s, r = document) => r.querySelector(s);
  const el = (tag, cls, html) => {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (html != null) n.innerHTML = html;
    return n;
  };

  /* ----------------------------- INIT ------------------------------------- */
  async function init() {
    const loaderMsg = $('#loader-msg');
    try {
      loaderMsg.textContent = 'Carregando malha do IBGE…';
      state.geojson = await D.loadGeojson();
      state.features = state.geojson.features;
      loaderMsg.textContent = 'Gerando previsão…';
      state.forecast = await D.loadForecast(state.features);
      buildMap();
      buildDayBar();
      renderAll();
      stampMeta();
      setTimeout(() => $('#loader').classList.add('hide'), 250);
    } catch (e) {
      loaderMsg.innerHTML = 'Erro ao carregar dados.<br><span style="color:var(--txt-faint);font-size:11px">' +
        (e.message || e) + '</span>';
      console.error(e);
    }
  }

  /* ----------------------------- MAP -------------------------------------- */
  function buildMap() {
    const map = L.map('map', {
      center: GO_CENTER,
      zoom: 7,
      zoomControl: false,
      attributionControl: true,
      preferCanvas: false,
      minZoom: 6,
      maxZoom: 13,
    });
    state.map = map;
    L.control.zoom({ position: 'topright' }).addTo(map);
    map.attributionControl.setPrefix(false);

    // Base: satélite (Esri World Imagery)
    L.tileLayer(
      'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
      { maxZoom: 18, attribution: 'Imagery © Esri, Maxar, Earthstar Geographics' }
    ).addTo(map);

    // Rótulos/limites de referência (discretos)
    state.labelLayer = L.tileLayer(
      'https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}',
      { maxZoom: 18, opacity: 0.5, pane: 'shadowPane' }
    ).addTo(map);

    // Camada de municípios (coroplético)
    state.muniLayer = L.geoJSON(state.geojson, {
      style: (f) => muniStyle(f),
      onEachFeature: (f, layer) => {
        const code = String(f.properties.id);
        state.layerByCode[code] = layer;
        layer.on({
          mouseover: () => hoverMuni(code, layer),
          mouseout: () => unhoverMuni(code, layer),
          click: () => selectMuni(code, { fly: true }),
        });
      },
    }).addTo(map);

    // Ajusta enquadramento a Goiás
    map.fitBounds(state.muniLayer.getBounds(), { padding: [10, 10] });
    state.stateBounds = state.muniLayer.getBounds();
  }

  function muniStyle(f) {
    const code = String(f.properties.id);
    const m = state.forecast.municipios[code];
    if (state.drilling) {
      return { fillOpacity: 0, color: 'rgba(120,130,140,.12)', weight: 0.4 };
    }
    if (m && m.outOfScope) {
      return { fillColor: '#6b7c85', fillOpacity: 0.18, color: 'rgba(80,95,105,.45)', weight: 0.6, dashArray: '4,4' };
    }
    const r = m ? m.risk[state.day] : 0;
    const dimmed = state.selected && state.selected !== code;
    return {
      fillColor: D.riskColor(r),
      fillOpacity: dimmed ? 0.06 : 0.52,
      color: dimmed ? 'rgba(120,130,140,.25)' : 'rgba(10,12,15,.85)',
      weight: dimmed ? 0.5 : 0.7,
    };
  }

  function restyleMuni() {
    state.muniLayer.setStyle((f) => muniStyle(f));
  }

  function hoverMuni(code, layer) {
    if (state.drilling) return;
    if (state.selected && state.selected !== code) return;
    const m = state.forecast.municipios[code];
    if (m.outOfScope) {
      layer.setStyle({ weight: 1.2, color: 'rgba(80,95,105,.7)', fillOpacity: 0.28 });
      layer.bringToFront();
      layer.bindTooltip(
        `<div class="tn">${m.name}</div><div class="tr" style="color:var(--txt-dim)">Mata Atlântica · fora do escopo de previsão</div>`,
        { className: 'muni-tip', sticky: true, direction: 'top', offset: [0, -4] }
      ).openTooltip();
      return;
    }
    const r = m.risk[state.day];
    layer.setStyle({ weight: 1.8, color: '#fff', fillOpacity: 0.62 });
    layer.bringToFront();
    const b = D.riskBucket(r);
    layer.bindTooltip(
      `<div class="tn">${m.name}</div><div class="tr"><span class="dot" style="background:${D.riskColor(r)}"></span>${D.fmtPct(r)} · ${b.label}</div>`,
      { className: 'muni-tip', sticky: true, direction: 'top', offset: [0, -4] }
    ).openTooltip();
    highlightRow(code, true);
  }
  function unhoverMuni(code, layer) {
    if (state.drilling) return;
    layer.setStyle(muniStyle(layer.feature));
    layer.unbindTooltip();
    if (!state.selected || state.selected === code) highlightRow(code, false);
  }

  /* --------------------- DRILL-IN: células finas -------------------------- */
  function selectMuni(code, opts = {}) {
    const m = state.forecast.municipios[code];
    if (m && m.outOfScope) return;
    state.selected = code;
    state.drilling = true;
    state.rankTab = 'e2';
    const feature = state.layerByCode[code].feature;

    restyleMuni();
    drawCells(feature, m);

    if (opts.fly !== false) {
      state.map.flyToBounds(state.layerByCode[code].getBounds(), {
        padding: [60, 60], duration: 0.9, maxZoom: 11,
      });
    }

    $('#tab-e2').disabled = false;
    setActiveTab('e2');
    $('#legend-e2').style.display = 'block';

    renderRanking();
    renderViewState(m);
  }

  function highlightCell(key, on) {
    if (!state.cellObjects) return;
    const obj = state.cellObjects.find(c => `${c.lat},${c.lon}` === key);
    if (!obj) return;
    if (on) {
      obj.rect.setStyle({ weight: 2, color: '#fff', fillOpacity: 0.92 });
      obj.rect.bringToFront();
    } else {
      obj.rect.setStyle({ fillColor: D.riskColor(obj.prob), fillOpacity: 0.72, color: 'rgba(10,12,15,.6)', weight: 0.5 });
    }
  }

  function flyToCell(c) {
    state.map.flyTo([c.lat, c.lon], 12, { duration: 0.7 });
  }

  function clearSelection() {
    state.selected = null;
    state.drilling = false;
    state.rankTab = 'e1';
    state.cellObjects = null;
    if (state.cellLayer) { state.map.removeLayer(state.cellLayer); state.cellLayer = null; }
    restyleMuni();
    state.map.flyToBounds(state.stateBounds, { padding: [10, 10], duration: 0.8 });

    const tabE2 = $('#tab-e2');
    tabE2.disabled = true;
    setActiveTab('e1');
    $('#legend-e2').style.display = 'none';

    renderRanking();
    renderViewState(null);
  }

  // ponto-em-polígono (ray casting) sobre os anéis externos da feature
  function polygonsOf(geom) {
    if (!geom) return [];
    return geom.type === 'Polygon' ? [geom.coordinates] : geom.coordinates;
  }
  function pointInRing(x, y, ring) {
    let inside = false;
    for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
      const xi = ring[i][0], yi = ring[i][1], xj = ring[j][0], yj = ring[j][1];
      if (((yi > y) !== (yj > y)) && (x < ((xj - xi) * (y - yi)) / (yj - yi) + xi)) inside = !inside;
    }
    return inside;
  }
  function pointInFeature(x, y, polys) {
    for (const poly of polys) if (pointInRing(x, y, poly[0])) return true;
    return false;
  }

  function drawCells(feature, m) {
    if (state.cellLayer) { state.map.removeLayer(state.cellLayer); state.cellLayer = null; }

    // Usa células reais do Estágio 2 (grade 0.1°) quando disponíveis
    const realCells = state.forecast.celulas && state.forecast.celulas[state.day];
    if (realCells && realCells.length > 0) {
      const norm = s => s.normalize('NFD').replace(/[\u0300-\u036f]/g, '').toLowerCase().trim();
      const munNorm = norm(m.name);
      const STEP = 0.1;
      const cellObjects = [];
      for (const c of realCells) {
        if (norm(c.mun) !== munNorm) continue;
        if (c.prob < state.minProb / 100) continue;
        const r = c.prob;
        const b = D.riskBucket(r);
        const latStr = Math.abs(c.lat).toFixed(1) + '\u00b0' + (c.lat < 0 ? 'S' : 'N');
        const lonStr = Math.abs(c.lon).toFixed(1) + '\u00b0' + (c.lon < 0 ? 'O' : 'L');
        const rect = L.rectangle(
          [[c.lat - STEP / 2, c.lon - STEP / 2], [c.lat + STEP / 2, c.lon + STEP / 2]],
          {
            fillColor: D.riskColor(r), fillOpacity: 0.72,
            color: 'rgba(10,12,15,.6)', weight: 0.5, className: 'cell-rect',
          }
        );
        rect.bindTooltip(
          `<div class="tn">${latStr}, ${lonStr}</div><div class="tr"><span class="dot" style="background:${D.riskColor(r)}"></span>${D.fmtPct(r)} \u00b7 ${b.label}</div>`,
          { className: 'muni-tip', direction: 'top', sticky: true }
        );
        cellObjects.push({ prob: r, lat: c.lat, lon: c.lon, rect });
      }
      state.cellObjects = cellObjects;
      state.cellLayer = L.layerGroup(cellObjects.map(o => o.rect)).addTo(state.map);
      return;
    }

    // Fallback: grade sintética (modo demo sem dados reais)
    const polys = polygonsOf(feature.geometry);
    let minx = 180, miny = 90, maxx = -180, maxy = -90;
    for (const poly of polys) for (const pt of poly[0]) {
      minx = Math.min(minx, pt[0]); maxx = Math.max(maxx, pt[0]);
      miny = Math.min(miny, pt[1]); maxy = Math.max(maxy, pt[1]);
    }
    const span = Math.max(maxx - minx, maxy - miny);
    const step = Math.max(0.02, span / 16);
    const base = m.risk[state.day];

    function cellRisk(ix, iy) {
      let h = (parseInt(m.code, 10) + ix * 73856093 + iy * 19349663) >>> 0;
      h = Math.imul(h ^ (h >>> 13), 1274126177) >>> 0;
      const noise = ((h % 1000) / 1000 - 0.5) * 0.42;
      return Math.max(0.02, Math.min(0.99, base + noise));
    }

    const cells = [];
    let ix = 0;
    for (let x = minx; x < maxx; x += step, ix++) {
      let iy = 0;
      for (let y = miny; y < maxy; y += step, iy++) {
        const cx = x + step / 2, cy = y + step / 2;
        if (!pointInFeature(cx, cy, polys)) continue;
        const r = cellRisk(ix, iy);
        const rect = L.rectangle(
          [[y, x], [y + step * 0.92, x + step * 0.92]],
          {
            fillColor: D.riskColor(r), fillOpacity: 0.72,
            color: 'rgba(10,12,15,.6)', weight: 0.5, className: 'cell-rect',
          }
        );
        rect.bindTooltip(`${D.fmtPct(r)}`, { className: 'cell-tip', direction: 'top', sticky: true });
        cells.push(rect);
      }
    }
    state.cellLayer = L.layerGroup(cells).addTo(state.map);
  }

  /* --------------------------- DAY BAR ------------------------------------ */
  function dayMeanColor(di) {
    const ms = Object.values(state.forecast.municipios).filter(m => !m.outOfScope);
    let s = 0; for (const m of ms) s += m.risk[di];
    return D.riskColor(s / ms.length);
  }
  function buildDayBar() {
    const bar = $('#daybar');
    bar.innerHTML = '<span class="daybar-lead">QUANDO →</span>';
    state.forecast.days.forEach((d, i) => {
      const when = d.label || '';
      const node = el('button', 'day' + (i === state.day ? ' is-active' : ''));
      node.innerHTML =
        `<span class="dow">${d.weekday}</span>` +
        `<span class="date">${String(d.day).padStart(2, '0')}/${String(d.month + 1).padStart(2, '0')}</span>` +
        `<span class="when">${when || '\u00A0'}</span>` +
        `<span class="peak" style="background:${dayMeanColor(i)}"></span>`;
      node.addEventListener('click', () => setDay(i));
      bar.appendChild(node);
    });
  }
  function setDay(i) {
    state.day = i;
    localStorage.setItem('vigia.day', String(i));
    $$('.day').forEach((n, k) => n.classList.toggle('is-active', k === i));
    restyleMuni();
    if (state.selected) {
      const m = state.forecast.municipios[state.selected];
      drawCells(state.layerByCode[state.selected].feature, m);
      renderViewState(m);
    }
    renderRanking();
  }
  const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));

  /* --------------------------- ABAS --------------------------------------- */
  function setActiveTab(tab) {
    state.rankTab = tab;
    $('#tab-e1').classList.toggle('is-active', tab === 'e1');
    $('#tab-e2').classList.toggle('is-active', tab === 'e2');
  }

  /* --------------------------- RANKING ------------------------------------ */
  function sortedMunis() {
    return Object.values(state.forecast.municipios)
      .filter(m => !m.outOfScope)
      .sort((a, b) => b.risk[state.day] - a.risk[state.day]);
  }
  function sparkline(risk) {
    const w = 42, h = 18, n = risk.length;
    const pts = risk.map((r, i) => {
      const x = (i / (n - 1)) * (w - 2) + 1;
      const y = h - 2 - r * (h - 4);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    }).join(' ');
    const cur = risk[state.day];
    const cx = (state.day / (n - 1)) * (w - 2) + 1;
    const cy = h - 2 - cur * (h - 4);
    return `<svg class="spark" viewBox="0 0 ${w} ${h}">
      <polyline points="${pts}" fill="none" stroke="rgba(154,166,178,.5)" stroke-width="1.2"/>
      <circle cx="${cx.toFixed(1)}" cy="${cy.toFixed(1)}" r="2.4" fill="${D.riskColor(cur)}"/>
    </svg>`;
  }
  function renderRanking() {
    const d = state.forecast.days[state.day];
    $('#rh-day').textContent = (d.label ? d.label + ' · ' : '') + d.weekday + ' ' +
      String(d.day).padStart(2, '0') + '/' + String(d.month + 1).padStart(2, '0');

    if (state.rankTab === 'e2' && state.selected) {
      renderE2Ranking();
    } else {
      renderE1Ranking();
    }
  }

  function renderE1Ranking() {
    $('#rh-k').textContent = 'Municípios mais críticos';
    $('#rh-sub').textContent = 'qual região concentrar atenção · ordenado por probabilidade';
    $('#rank-foot').style.display = '';

    const list = $('#rank-list');
    const sorted = sortedMunis();
    const limit = state.expanded ? sorted.length : 10;
    list.innerHTML = '';
    sorted.slice(0, limit).forEach((m, i) => {
      const r = m.risk[state.day];
      const b = D.riskBucket(r);
      const row = el('div', 'row' + (state.selected === m.code ? ' is-sel' : ''));
      row.dataset.code = m.code;
      row.innerHTML =
        `<span class="pos">${String(i + 1).padStart(2, '0')}</span>` +
        `<span class="nm"><div class="mn">${m.name}</div><div class="bk" style="color:${D.riskColor(r)}">${b.label}</div></span>` +
        `<span class="val">${sparkline(m.risk)}<span class="pct">${D.fmtPct(r)}</span><span class="dot" style="background:${D.riskColor(r)}"></span></span>`;
      row.addEventListener('click', () => selectMuni(m.code, { fly: true }));
      row.addEventListener('mouseenter', () => peekMuni(m.code, true));
      row.addEventListener('mouseleave', () => peekMuni(m.code, false));
      list.appendChild(row);
    });

    const btn = $('#btn-expand');
    btn.textContent = state.expanded
      ? '— recolher'
      : `+ ver todos os ${sorted.length} municípios`;
  }

  function renderE2Ranking() {
    const m = state.forecast.municipios[state.selected];
    if (!m) return;
    $('#rh-k').textContent = m.name;
    $('#rh-sub').textContent = 'zonas de risco dentro do município · por probabilidade';
    $('#rank-foot').style.display = 'none';

    const norm = s => s.normalize('NFD').replace(/[\u0300-\u036f]/g, '').toLowerCase().trim();
    const munNorm = norm(m.name);
    const allCells = (state.forecast.celulas && state.forecast.celulas[state.day]) || [];
    const cells = allCells
      .filter(c => norm(c.mun) === munNorm && c.prob >= state.minProb / 100)
      .sort((a, b) => b.prob - a.prob);

    const list = $('#rank-list');
    list.innerHTML = '';

    if (cells.length === 0) {
      const msg = el('div', 'row');
      msg.style.cssText = 'justify-content:center;color:var(--txt-faint);font-family:var(--mono);font-size:11px;padding:20px;cursor:default';
      msg.textContent = state.minProb > 0
        ? `Nenhuma área acima de ${state.minProb}%`
        : 'Sem dados de zona para este município';
      list.appendChild(msg);
      return;
    }

    cells.forEach((c, i) => {
      const r = c.prob;
      const b = D.riskBucket(r);
      const latStr = Math.abs(c.lat).toFixed(1) + '°' + (c.lat < 0 ? 'S' : 'N');
      const lonStr = Math.abs(c.lon).toFixed(1) + '°' + (c.lon < 0 ? 'O' : 'L');
      const key = `${c.lat},${c.lon}`;
      const row = el('div', 'row');
      row.dataset.key = key;
      row.innerHTML =
        `<span class="pos">${String(i + 1).padStart(2, '0')}</span>` +
        `<span class="nm"><div class="mn" style="font-size:13px">${latStr}, ${lonStr}</div><div class="bk" style="color:${D.riskColor(r)}">${b.label}</div></span>` +
        `<span class="val"><span class="pct">${D.fmtPct(r)}</span><span class="dot" style="background:${D.riskColor(r)}"></span></span>`;
      row.addEventListener('click', () => flyToCell(c));
      row.addEventListener('mouseenter', () => highlightCell(key, true));
      row.addEventListener('mouseleave', () => highlightCell(key, false));
      list.appendChild(row);
    });
  }
  function peekMuni(code, on) {
    const layer = state.layerByCode[code];
    if (!layer) return;
    if (on) {
      if (state.selected && state.selected !== code) return;
      layer.setStyle({ weight: 1.8, color: '#fff' });
      layer.bringToFront();
    } else {
      layer.setStyle(muniStyle(layer.feature));
    }
  }
  function highlightRow(code, on) {
    const row = $(`.row[data-code="${code}"]`);
    if (row) row.classList.toggle('is-sel', on || state.selected === code);
  }

  /* --------------------------- VIEW STATE --------------------------------- */
  function renderViewState(m) {
    const box = $('#viewstate');
    if (!m) { box.style.display = 'none'; return; }
    const r = m.risk[state.day];
    const b = D.riskBucket(r);
    box.style.display = 'flex';
    box.innerHTML =
      `<div><div class="vk">Município · onde dentro</div>` +
      `<div class="vv">${m.name}</div>` +
      `<div class="vsub" style="color:${D.riskColor(r)}">${D.fmtPct(r)} · ${b.label} · células de risco</div></div>` +
      `<button id="vs-close" title="Voltar a Goiás">×</button>`;
    $('#vs-close').addEventListener('click', clearSelection);
  }

  /* --------------------------- META / STAMP ------------------------------- */
  function stampMeta() {
    const dt = new Date(state.forecast.generated_at);
    const p = Object.fromEntries(
      new Intl.DateTimeFormat('pt-BR', {
        timeZone: 'America/Sao_Paulo',
        day: '2-digit', month: '2-digit',
        hour: '2-digit', minute: '2-digit', hour12: false,
      }).formatToParts(dt).map(x => [x.type, x.value])
    );
    const txt = `${p.day}/${p.month} ${p.hour}:${p.minute}`;
    $('#stamp-val').textContent = txt;
    const src = state.forecast.source === 'real' ? 'dados reais' : 'dados sintéticos · demo';
    $('#stamp-src').textContent = src;
    // AUC no modal: real → decimal ("0,816"), sintético → faixa percentual ("70–80%")
    const mm = state.forecast.model || {};
    const el = $('#acc-range');
    if (el && mm.accuracy_min) {
      if (state.forecast.source === 'real') {
        el.textContent = mm.accuracy_min.toFixed(3).replace('.', ',');
      } else if (mm.accuracy_max) {
        el.textContent = Math.round(mm.accuracy_min * 100) + '–' + Math.round(mm.accuracy_max * 100) + '%';
      }
    }
  }

  /* --------------------------- RENDER ALL --------------------------------- */
  function renderAll() {
    renderRanking();
    renderViewState(null);

    $('#btn-expand').addEventListener('click', () => {
      state.expanded = !state.expanded;
      renderRanking();
    });

    $('#tab-e1').addEventListener('click', () => {
      if (state.rankTab === 'e1') return;
      setActiveTab('e1');
      renderRanking();
    });
    $('#tab-e2').addEventListener('click', () => {
      if (state.rankTab === 'e2' || !state.selected) return;
      setActiveTab('e2');
      renderRanking();
    });

    const slider = $('#prob-slider');
    slider.addEventListener('input', () => {
      state.minProb = Number(slider.value);
      $('#prob-val').textContent = '≥ ' + state.minProb + '%';
      if (state.selected) {
        const m = state.forecast.municipios[state.selected];
        drawCells(state.layerByCode[state.selected].feature, m);
        if (state.rankTab === 'e2') renderRanking();
      }
    });

    $('#btn-about').addEventListener('click', () => $('#modal').classList.add('show'));
    $('#modal-close').addEventListener('click', () => $('#modal').classList.remove('show'));
    $('#modal').addEventListener('click', (e) => { if (e.target.id === 'modal') $('#modal').classList.remove('show'); });
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') {
        if ($('#modal').classList.contains('show')) $('#modal').classList.remove('show');
        else if (state.selected) clearSelection();
      }
    });
  }

  window.addEventListener('DOMContentLoaded', init);
})();
