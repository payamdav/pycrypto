/* ------------------------------------------------------------------ *
 *  Report renderer
 *
 *  Turns a parsed master-JSON report into DOM and renders every Plotly
 *  visualization. Tolerant of missing optional fields.
 * ------------------------------------------------------------------ */

(function () {
  // ---- small DOM helpers ------------------------------------------------
  function el(tag, cls, text) {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined && text !== null) node.textContent = String(text);
    return node;
  }

  function section(title, badge) {
    const sec = el("section", "section");
    const head = el("div", "section__head");
    head.appendChild(el("h2", null, title));
    if (badge) head.appendChild(el("span", "badge", badge));
    const body = el("div", "section__body");
    sec.appendChild(head);
    sec.appendChild(body);
    sec._body = body;
    return sec;
  }

  function kv(key, value) {
    const wrap = el("div", "kv");
    wrap.appendChild(el("span", "kv__k", key));
    wrap.appendChild(el("span", "kv__v", value));
    return wrap;
  }

  function fmtNum(v, digits = 6) {
    if (v === null || v === undefined || v === "") return "—";
    const n = Number(v);
    if (!Number.isFinite(n)) return String(v);
    if (Number.isInteger(n)) return String(n);
    return n.toFixed(digits);
  }

  // ---- section builders -------------------------------------------------
  function buildBanner(report) {
    const md = report.metadata || {};
    const banner = el("div", "obs-banner");
    const left = el("div");
    left.appendChild(el("div", "obs-set", md.observation_set_name || "—"));
    left.appendChild(el("h2", null, md.observation_name || "Untitled observation"));
    if (md.dry_run) {
      const flag = el("span", "dry-run-flag", "NO-TRAIN DRY RUN");
      left.appendChild(flag);
    }
    banner.appendChild(left);

    const meta = el("div", "obs-meta");
    meta.appendChild(el("div", null, `Asset: ${md.asset_identifier || "—"}`));
    meta.appendChild(el("div", null, `Run: ${md.execution_timestamp || "—"}`));
    banner.appendChild(meta);
    return banner;
  }

  function buildMetadata(report) {
    const md = report.metadata || {};
    const sec = section("Metadata");
    const grid = el("div", "kv-grid");
    grid.appendChild(kv("Observation Set", md.observation_set_name));
    grid.appendChild(kv("Observation Name", md.observation_name));
    grid.appendChild(kv("Asset", md.asset_identifier));
    grid.appendChild(kv("Execution Timestamp", md.execution_timestamp));
    const range = md.data_date_range || {};
    grid.appendChild(kv("Data Start", range.start));
    grid.appendChild(kv("Data End", range.end));
    if (md.dry_run) {
      grid.appendChild(kv("Dry Run", "true"));
    }
    sec._body.appendChild(grid);
    if (md.dry_run_note) {
      const note = el("p", null, md.dry_run_note);
      note.style.color = "var(--warn)";
      note.style.marginTop = "0.8rem";
      sec._body.appendChild(note);
    }
    return sec;
  }

  function buildArchitecture(report) {
    const a = report.model_architecture || {};
    const sec = section("Model Architecture");
    const grid = el("div", "kv-grid");
    grid.appendChild(kv("Model Type", a.model_type));
    grid.appendChild(kv("Sequence Length", a.sequence_length));
    sec._body.appendChild(grid);

    if (Array.isArray(a.input_features)) {
      sec._body.appendChild(el("div", "kv__k", "Input Features"));
      const chips = el("div", "chips");
      chips.style.margin = "0.4rem 0 0.9rem";
      a.input_features.forEach((f) => chips.appendChild(el("span", "chip", f)));
      sec._body.appendChild(chips);
    }
    if (Array.isArray(a.target_labels)) {
      sec._body.appendChild(el("div", "kv__k", "Target Labels"));
      const chips = el("div", "chips");
      chips.style.marginTop = "0.4rem";
      a.target_labels.forEach((f) => chips.appendChild(el("span", "chip", f)));
      sec._body.appendChild(chips);
    }
    return sec;
  }

  function buildTelemetry(report) {
    const t = report.training_telemetry || {};
    const sec = section("Training Telemetry");
    const grid = el("div", "kv-grid");
    const fields = [
      ["Total Parameters", t.total_parameters],
      ["Epochs Completed", t.epochs_completed],
      ["Batch Size", t.batch_size],
      ["Hardware", t.hardware_utilized],
      ["Train Time (s)", t.total_train_time_seconds],
      ["Rows Train", t.rows_train],
      ["Rows Val", t.rows_val],
      ["Rows Test", t.rows_test],
      ["Feature Count", t.feature_count],
      ["No-Train Dry Run", t.no_train_dry_run],
    ];
    fields.forEach(([k, v]) => {
      if (v !== undefined) grid.appendChild(kv(k, v));
    });
    sec._body.appendChild(grid);
    return sec;
  }

  function buildEvaluation(report) {
    const m = report.evaluation_metrics || {};
    const sec = section("Evaluation Metrics");

    // Global metrics as stat tiles.
    const g = m.global_metrics || {};
    if (Object.keys(g).length) {
      sec._body.appendChild(el("h3", null, "Global Metrics"));
      const row = el("div", "stat-row");
      const add = (label, val, mod) => {
        const s = el("div", "stat" + (mod ? " " + mod : ""));
        s.appendChild(el("div", "stat__v", fmtNum(val)));
        s.appendChild(el("div", "stat__k", label));
        row.appendChild(s);
      };
      add("Test Huber Loss", g.test_huber_loss, "stat--accent");
      add("Test MSE", g.test_mse);
      sec._body.appendChild(row);
    }

    // Per-head metrics.
    const heads = m.per_head_metrics || {};
    const headNames = Object.keys(heads);
    if (headNames.length) {
      const h3 = el("h3", null, "Per-Head Metrics");
      h3.style.marginTop = "1.2rem";
      sec._body.appendChild(h3);
      headNames.forEach((name) => {
        const h = heads[name] || {};
        const card = el("div", "head-card");
        card.appendChild(el("div", "head-card__title", name));
        const row = el("div", "stat-row");
        const add = (label, val, mod) => {
          const s = el("div", "stat" + (mod ? " " + mod : ""));
          s.appendChild(el("div", "stat__v", fmtNum(val, 4)));
          s.appendChild(el("div", "stat__k", label));
          row.appendChild(s);
        };
        add("MAE", h.mae);
        add("MSE", h.mse);
        add("Directional Acc %", h.directional_accuracy_pct, "stat--good");
        card.appendChild(row);
        sec._body.appendChild(card);
      });
    }
    return sec;
  }

  function buildVisualizations(report) {
    const vis = report.visualizations || {};
    const names = Object.keys(vis);
    const sec = section("Visualizations", `${names.length} plots`);
    if (!names.length) {
      sec._body.appendChild(el("p", null, "No visualizations in this report."));
      return { sec, plots: [] };
    }
    const plots = [];
    names.forEach((name) => {
      sec._body.appendChild(el("p", "chart-title", name));
      const host = el("div", "chart");
      const id = "chart_" + name.replace(/[^a-z0-9]/gi, "_");
      host.id = id;
      sec._body.appendChild(host);
      plots.push({ id, name, raw: vis[name] });
    });
    return { sec, plots };
  }

  /** Parse a Plotly serialization (string or already-parsed object). */
  function parsePlotly(raw) {
    if (raw == null) return null;
    let obj = raw;
    if (typeof raw === "string") {
      obj = JSON.parse(raw);
    }
    return { data: obj.data || [], layout: obj.layout || {} };
  }

  /** Render all collected Plotly charts after the DOM is attached. */
  function renderPlots(plots) {
    const darkLayout = {
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: { color: "#e6edf3" },
      margin: { t: 40, r: 20, b: 50, l: 60 },
    };
    plots.forEach(({ id, name, raw }) => {
      const host = document.getElementById(id);
      if (!host) return;
      try {
        const fig = parsePlotly(raw);
        if (!fig) throw new Error("empty figure");
        const layout = Object.assign({}, darkLayout, fig.layout || {});
        layout.font = Object.assign({ color: "#e6edf3" }, layout.font || {});
        Plotly.newPlot(host, fig.data, layout, {
          responsive: true,
          displaylogo: false,
        });
      } catch (err) {
        host.innerHTML = "";
        const e = el("p", null, `Failed to render "${name}": ${err.message}`);
        e.style.color = "var(--danger)";
        host.appendChild(e);
      }
    });
  }

  /** Main entry: render a full report into the given container element. */
  function renderReport(container, report) {
    container.innerHTML = "";
    container.appendChild(buildBanner(report));
    container.appendChild(buildMetadata(report));
    container.appendChild(buildArchitecture(report));
    container.appendChild(buildTelemetry(report));
    container.appendChild(buildEvaluation(report));

    const { sec, plots } = buildVisualizations(report);
    container.appendChild(sec);

    // Plotly needs the hosts attached to the DOM before drawing.
    renderPlots(plots);
  }

  window.Renderer = { renderReport };
})();
