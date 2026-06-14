/* ------------------------------------------------------------------ *
 *  Application controller (SPA)
 *
 *  Wires up report discovery, the dropdown / prev-next navigation,
 *  manual upload / URL paste, and delegates rendering to Renderer.
 *  No page reloads occur when switching reports.
 * ------------------------------------------------------------------ */

(function () {
  const cfg = window.APP_CONFIG;

  const state = {
    keys: [],        // discovered report object keys
    index: -1,       // current selection index into keys (-1 = none / ad hoc)
    source: null,    // where the listing came from
  };

  // ---- element refs -----------------------------------------------------
  const elSelect = document.getElementById("reportSelect");
  const elPrev = document.getElementById("prevBtn");
  const elNext = document.getElementById("nextBtn");
  const elReload = document.getElementById("reloadBtn");
  const elStatus = document.getElementById("statusLabel");
  const elContent = document.getElementById("content");
  const elPlaceholder = document.getElementById("placeholder");
  const elFile = document.getElementById("fileInput");
  const elUrl = document.getElementById("urlInput");
  const elUrlLoad = document.getElementById("urlLoadBtn");
  const elFooter = document.getElementById("footerInfo");
  const elSetName = document.getElementById("setNameLabel");

  // ---- ui helpers -------------------------------------------------------
  function setStatus(html, busy) {
    elStatus.innerHTML = (busy ? '<span class="spinner"></span>' : "") + html;
  }

  function toast(msg, ok) {
    const t = document.createElement("div");
    t.className = "toast" + (ok ? " toast--ok" : "");
    t.textContent = msg;
    document.body.appendChild(t);
    setTimeout(() => t.remove(), 4200);
  }

  function shortName(key) {
    // strip the {study}/reports/ path + the "{SET}_" filename prefix.
    let file = key.split("/").pop().replace(cfg.REPORT_SUFFIX, "");
    const set = setName(key);
    if (set && set !== "(root)" && file.startsWith(set + "_")) {
      file = file.slice(set.length + 1);
    }
    return file;
  }

  function setName(key) {
    // The study folder is the path up to (but excluding) the "reports/"
    // segment, e.g. "Study/reports/Study_obs.json" -> "Study".
    const parts = key.split("/");
    const ri = parts.indexOf("reports");
    if (ri > 0) return parts.slice(0, ri).join("/");
    return parts.length > 1 ? parts[0] : "(root)";
  }

  function showContentArea() {
    if (elPlaceholder) elPlaceholder.style.display = "none";
  }

  function showError(message) {
    showContentArea();
    elContent.innerHTML =
      '<div class="error-card" style="color:var(--danger,#ff6b6b);' +
      'background:rgba(255,107,107,0.08);border:1px solid var(--danger,#ff6b6b);' +
      'border-radius:8px;padding:1.5rem 2rem;margin:2rem auto;max-width:720px;' +
      'font-family:monospace;white-space:pre-wrap">' +
      '<strong>Error</strong>\n' + message + "</div>";
  }

  // ---- dropdown population ---------------------------------------------
  function populateSelect() {
    elSelect.innerHTML = "";
    if (!state.keys.length) {
      const opt = document.createElement("option");
      opt.value = "";
      opt.textContent = "No reports found";
      elSelect.appendChild(opt);
      return;
    }
    // group options by observation set folder
    const groups = {};
    state.keys.forEach((key, i) => {
      const g = setName(key);
      (groups[g] = groups[g] || []).push({ key, i });
    });
    Object.keys(groups)
      .sort()
      .forEach((g) => {
        const og = document.createElement("optgroup");
        og.label = g;
        groups[g].forEach(({ key, i }) => {
          const opt = document.createElement("option");
          opt.value = String(i);
          opt.textContent = shortName(key);
          og.appendChild(opt);
        });
        elSelect.appendChild(og);
      });
  }

  function syncNavButtons() {
    elPrev.disabled = !(state.index > 0);
    elNext.disabled = !(state.index >= 0 && state.index < state.keys.length - 1);
  }

  // ---- report loading ---------------------------------------------------
  async function loadByIndex(i) {
    if (i < 0 || i >= state.keys.length) return;
    const key = state.keys[i];
    state.index = i;
    elSelect.value = String(i);
    setStatus(`Loading <code>${key}</code>`, true);
    try {
      const report = await window.GCS.fetchReportByKey(key);
      renderInto(report, key);
      setStatus(`Loaded <code>${key}</code>`);
    } catch (err) {
      setStatus(`Error loading report: ${err.message}`, false);
      showError(`Failed to load report:\n  Key: ${key}\n  Error: ${err.message}`);
      toast(`Failed to load ${key}: ${err.message}`);
    }
    syncNavButtons();
  }

  async function loadByUrl(url) {
    setStatus(`Loading <code>${url}</code>`, true);
    try {
      const report = await window.GCS.fetchReportByUrl(url);
      state.index = -1;
      renderInto(report, url);
      setStatus(`Loaded external report`);
    } catch (err) {
      setStatus(`Error loading URL: ${err.message}`, false);
      showError(`Failed to load URL:\n  URL: ${url}\n  Error: ${err.message}`);
      toast(`Failed to load URL: ${err.message}`);
    }
    syncNavButtons();
  }

  function renderInto(report, sourceLabel) {
    showContentArea();
    window.Renderer.renderReport(elContent, report);
    const md = report.metadata || {};
    if (md.observation_set_name) elSetName.textContent = md.observation_set_name;
    elFooter.textContent = `Source: ${sourceLabel}`;
  }

  // ---- discovery --------------------------------------------------------
  async function discover() {
    setStatus("Discovering reports…", true);
    elSelect.innerHTML = '<option value="">Loading reports…</option>';
    try {
      const { keys, source } = await window.GCS.discoverReports();
      state.keys = keys;
      state.source = source;
      populateSelect();
      setStatus(`Found ${keys.length} report(s) via ${source}.`);
      if (keys.length) {
        await loadByIndex(0);
      } else {
        syncNavButtons();
      }
    } catch (err) {
      state.keys = [];
      populateSelect();
      syncNavButtons();
      setStatus(
        "Could not list bucket. Use upload / URL paste to load a report.",
        false
      );
      showError(
        "Could not discover reports from the bucket.\n" + err.message +
        "\n\nUse the 'Upload JSON' button or 'Paste a report URL' below."
      );
      toast(err.message);
    }
  }

  // ---- events -----------------------------------------------------------
  elSelect.addEventListener("change", (e) => {
    const v = e.target.value;
    if (v === "") return;
    loadByIndex(parseInt(v, 10));
  });

  elPrev.addEventListener("click", () => loadByIndex(state.index - 1));
  elNext.addEventListener("click", () => loadByIndex(state.index + 1));
  elReload.addEventListener("click", () => discover());

  elFile.addEventListener("change", (e) => {
    const file = e.target.files && e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const report = JSON.parse(reader.result);
        state.index = -1;
        renderInto(report, `local file: ${file.name}`);
        setStatus(`Loaded local file <code>${file.name}</code>`);
        syncNavButtons();
      } catch (err) {
        toast(`Invalid JSON: ${err.message}`);
      }
    };
    reader.onerror = () => toast("Could not read file");
    reader.readAsText(file);
  });

  elUrlLoad.addEventListener("click", () => {
    const url = elUrl.value.trim();
    if (url) loadByUrl(url);
  });
  elUrl.addEventListener("keydown", (e) => {
    if (e.key === "Enter") elUrlLoad.click();
  });

  // keyboard navigation: left / right arrows
  document.addEventListener("keydown", (e) => {
    if (e.target && /INPUT|SELECT|TEXTAREA/.test(e.target.tagName)) return;
    if (e.key === "ArrowLeft" && !elPrev.disabled) loadByIndex(state.index - 1);
    if (e.key === "ArrowRight" && !elNext.disabled) loadByIndex(state.index + 1);
  });

  // ---- boot -------------------------------------------------------------
  let booted = false;
  function boot() {
    if (booted) return;
    booted = true;
    // Show the active study (derived from config) before any report loads.
    if (elSetName && cfg.STUDY_PREFIX) {
      elSetName.textContent = cfg.STUDY_PREFIX;
    }
    discover();
  }
  // Scripts are at the end of <body>; DOMContentLoaded may have already
  // fired. Cover both cases without double-booting.
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
