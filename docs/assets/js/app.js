/* tabalign project page - data comes from the paper's own results table. */
(function () {
  "use strict";

  // Paper's publication palette (create_figures.py PALETTE)
  // TabPFN: Imperial Violet (#592D86, slightly lifted to #723BAE for dark theme)
  // TabFM:  Warm Gold / Ochre (#D1A11F)
  var COLOR = { pfn: "#763fa8", fm: "#D1A11F" };
  var D = null;                       // loaded payload

  var $ = function (s) { return document.querySelector(s); };

  if (window.TABALIGN_RESULTS) {
    loadData(window.TABALIGN_RESULTS);
  } else {
    fetch("./data/results.json?v=" + Date.now(), { cache: "no-store" })
      .then(function (r) { return r.json(); })
      .then(function (json) {
        loadData(json);
      })
      .catch(function (e) {
        $("#chart-a").innerHTML =
          '<p style="padding:24px;color:#8b9bb4">Could not load results.json - ' + e + "</p>";
      });
  }

  function loadData(json) {
    json.alphas = json.alphas.filter(function (a) { return a <= 0.501; });
    D = json;
    init();
  }

  function init() {
    initDatasetPicker();
    initAlphaSlider();

    window.addEventListener("resize", debounce(render, 180));
    render();
  }

  function initDatasetPicker() {
    var picker = $("#dataset-picker");
    var btn = $("#dataset-btn");
    var cur = $("#dataset-cur");
    var list = $("#dataset-list");
    var input = $("#dataset");

    var options = [
      { val: "__avg__", label: "Average (all 38 datasets)", badge: "38 datasets" }
    ].concat(D.datasets.map(function (d, i) {
      return { val: String(i), label: d.name, badge: null };
    }));

    function renderList() {
      var curVal = input.value;
      list.innerHTML = options.map(function (opt) {
        var isSel = opt.val === curVal;
        return '<li class="cs-item' + (isSel ? " selected" : "") + '" data-val="' + opt.val + '" role="option" tabindex="0">' +
          '<span class="cs-item-name">' + escapeHtml(opt.label) + '</span>' +
          (opt.badge ? '<span class="cs-badge">' + opt.badge + '</span>' : '') +
          '<svg class="cs-check" width="12" height="12" viewBox="0 0 12 12" fill="none"><path d="M2.5 6.5L4.8 9L9.5 3.5" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>' +
          '</li>';
      }).join("");

      var items = list.querySelectorAll(".cs-item");
      [].forEach.call(items, function (it) {
        it.addEventListener("click", function () {
          selectVal(this.dataset.val);
        });
      });
    }

    function selectVal(val) {
      var found = options.find(function (o) { return o.val === val; });
      if (!found) return;
      input.value = found.val;
      cur.textContent = found.label;
      close();
      renderList();
      render();
    }

    function open() {
      picker.classList.add("open");
      btn.setAttribute("aria-expanded", "true");
      renderList();
      var sel = list.querySelector(".cs-item.selected") || list.querySelector(".cs-item");
      if (sel) {
        setTimeout(function () {
          sel.focus();
          sel.scrollIntoView({ block: "nearest" });
        }, 20);
      }
    }

    function close() {
      picker.classList.remove("open");
      btn.setAttribute("aria-expanded", "false");
    }

    btn.addEventListener("click", function (e) {
      e.stopPropagation();
      picker.classList.contains("open") ? close() : open();
    });

    btn.addEventListener("keydown", function (e) {
      if (e.key === "ArrowDown" || e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        open();
      }
    });

    list.addEventListener("keydown", function (e) {
      var active = document.activeElement;
      if (e.key === "ArrowDown") {
        e.preventDefault();
        var next = active.nextElementSibling;
        if (next && next.classList.contains("cs-item")) {
          next.focus();
          next.scrollIntoView({ block: "nearest" });
        }
      } else if (e.key === "ArrowUp") {
        e.preventDefault();
        var prev = active.previousElementSibling;
        if (prev && prev.classList.contains("cs-item")) {
          prev.focus();
          prev.scrollIntoView({ block: "nearest" });
        }
      } else if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        if (active && active.dataset.val) selectVal(active.dataset.val);
      } else if (e.key === "Escape") {
        close();
        btn.focus();
      }
    });

    document.addEventListener("click", function (e) {
      if (!picker.contains(e.target)) close();
    });

    renderList();
  }

  function initAlphaSlider() {
    var wrap = $("#alpha-slider");
    var hit = $("#as-hit");
    var track = $(".as-track");
    var fill = $("#as-fill");
    var thumb = $("#as-thumb");
    var input = $("#alpha");
    var valDisplay = $("#alphaVal");
    var ticks = wrap.querySelectorAll(".as-tick");
    var labels = wrap.querySelectorAll(".as-lbl");

    var N = D.alphas.length; // 5 steps
    var currentStep = +input.value || 0;
    var isDragging = false;

    function setStep(step, animate) {
      step = Math.max(0, Math.min(N - 1, Math.round(step)));
      currentStep = step;
      input.value = step;
      valDisplay.textContent = Math.round(D.alphas[step] * 100) + "%";
      wrap.setAttribute("aria-valuenow", step);

      var pct = (step / (N - 1)) * 100;

      if (!animate) {
        thumb.style.transition = "none";
        fill.style.transition = "none";
      } else {
        thumb.style.transition = "left 0.25s cubic-bezier(0.2, 0.9, 0.3, 1)";
        fill.style.transition = "width 0.25s cubic-bezier(0.2, 0.9, 0.3, 1)";
      }

      thumb.style.left = pct + "%";
      fill.style.width = pct + "%";
      ticks.forEach(function (t, i) {
        t.classList.toggle("on", i <= step);
      });
      labels.forEach(function (l, i) {
        l.classList.toggle("on", i === step);
      });

      updateBothGuideLines(D.alphas[step], animate);
      render(animate);
    }

    function stepFromPointer(clientX) {
      var rect = track.getBoundingClientRect();
      var x = Math.max(0, Math.min(rect.width, clientX - rect.left));
      var frac = x / rect.width;
      return Math.round(frac * (N - 1));
    }

    var hoveredStep = -1;

    function setHover(step) {
      if (hoveredStep === step) return;
      hoveredStep = step;
      labels.forEach(function (l, i) {
        l.classList.toggle("hovered", i === step);
      });
      ticks.forEach(function (t, i) {
        t.classList.toggle("hovered", i === step);
      });
    }

    function clearHover() {
      if (hoveredStep === -1) return;
      hoveredStep = -1;
      labels.forEach(function (l) {
        l.classList.remove("hovered");
      });
      ticks.forEach(function (t) {
        t.classList.remove("hovered");
      });
    }

    hit.addEventListener("pointerdown", function (e) {
      isDragging = true;
      hit.setPointerCapture(e.pointerId);
      var s = stepFromPointer(e.clientX);
      setStep(s, true); // smooth glide animation!
      setHover(s);
    });

    hit.addEventListener("pointermove", function (e) {
      var s = stepFromPointer(e.clientX);
      if (isDragging) {
        if (s !== currentStep) {
          setStep(s, true);
        }
      }
      setHover(s);
    });

    hit.addEventListener("pointerup", function (e) {
      if (isDragging) {
        isDragging = false;
        try { hit.releasePointerCapture(e.pointerId); } catch (_) {}
      }
      var s = stepFromPointer(e.clientX);
      setHover(s);
    });

    hit.addEventListener("pointerleave", function () {
      if (!isDragging) {
        clearHover();
      }
    });

    hit.addEventListener("pointercancel", function () {
      isDragging = false;
      clearHover();
    });

    wrap.addEventListener("pointerleave", function () {
      if (!isDragging) {
        clearHover();
      }
    });

    labels.forEach(function (l, i) {
      l.addEventListener("pointerenter", function () {
        setHover(i);
      });
      l.addEventListener("pointerleave", function () {
        clearHover();
      });
      l.addEventListener("click", function () {
        setStep(+this.dataset.val, true);
      });
    });

    ticks.forEach(function (t, i) {
      t.addEventListener("pointerenter", function () {
        setHover(i);
      });
      t.addEventListener("pointerleave", function () {
        clearHover();
      });
      t.addEventListener("click", function (e) {
        e.stopPropagation();
        setStep(+this.dataset.val, true);
      });
    });

    wrap.addEventListener("keydown", function (e) {
      if (e.key === "ArrowRight" || e.key === "ArrowUp") {
        e.preventDefault();
        setStep(currentStep + 1, true);
      } else if (e.key === "ArrowLeft" || e.key === "ArrowDown") {
        e.preventDefault();
        setStep(currentStep - 1, true);
      } else if (e.key === "Home") {
        e.preventDefault();
        setStep(0, true);
      } else if (e.key === "End") {
        e.preventDefault();
        setStep(N - 1, true);
      }
    });

    setStep(currentStep, false);
  }

  function escapeHtml(str) {
    return str.replace(/[&<>"']/g, function (m) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[m];
    });
  }

  var guideLines = {};

  function ensureGuideLine(chartId) {
    var chart = $("#" + chartId);
    if (!guideLines[chartId] && chart) {
      var svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
      svg.setAttribute("class", "chart-guide-svg");
      var line = document.createElementNS("http://www.w3.org/2000/svg", "line");
      line.setAttribute("class", "chart-guide-line");
      svg.appendChild(line);
      chart.appendChild(svg);
      guideLines[chartId] = { svg: svg, line: line };
    }
  }

  function updateGuideLine(chartId, alpha, animate) {
    var chart = $("#" + chartId);
    if (!chart || !chart._fullLayout || !chart._fullLayout.xaxis) return;
    ensureGuideLine(chartId);
    var xa = chart._fullLayout.xaxis;
    var ya = chart._fullLayout.yaxis;
    var targetX = xa._offset + xa.d2p(alpha);
    var y1 = ya._offset;
    var y2 = ya._offset + ya._length;

    var line = guideLines[chartId].line;
    line.setAttribute("x1", "0");
    line.setAttribute("y1", y1);
    line.setAttribute("x2", "0");
    line.setAttribute("y2", y2);

    if (animate) {
      line.style.transition = "transform 0.25s cubic-bezier(0.2, 0.9, 0.3, 1)";
    } else {
      line.style.transition = "none";
    }
    line.style.transform = "translateX(" + targetX + "px)";
  }

  function updateBothGuideLines(alpha, animate) {
    updateGuideLine("chart-a", alpha, animate);
    updateGuideLine("chart-b", alpha, animate);
  }

  /* ---------- series for the selected dataset ---------- */
  function seriesFor(model, metric) {
    var sel = $("#dataset").value;
    var row = sel === "__avg__" ? D.average_row : D.datasets[+sel];
    var xs = [], ys = [], wins = [];
    var key = metric === "gc" ? "gc" : "e";
    D.alphas.forEach(function (a) {
      var c = row[model][a.toFixed(1)];
      if (c && c[key] !== null && c[key] !== undefined) {
        xs.push(a);
        ys.push(c[key]);
        wins.push(c.win);
      }
    });
    return { x: xs, y: ys, wins: wins, name: row.name };
  }

  function render(isAnimated) {
    if (!D) return;
    var ai = +$("#alpha").value;
    var alpha = D.alphas[ai];
    var sel = $("#dataset").value;
    var isAvg = sel === "__avg__";

    var plotH = 440;
    var plotW = 220;
    var marginPlot = { l: 48, r: 16, t: 64, b: 48 };
    var chartW = plotW + marginPlot.l + marginPlot.r;
    var chartH = plotH + marginPlot.t + marginPlot.b;

    // ----------------- Chart A: Sample Efficiency (E_alpha) -----------------
    var tracesA = [];
    // Reference: parity line y = x (starts at 0,0 and ends at 0.5,0.5 at 45 degrees)
    tracesA.push({
      x: [0, 0.50], y: [0, 0.50], mode: "lines", name: "No benefit",
      line: { color: "#64748b", width: 1.5, dash: "dash" }, hoverinfo: "skip"
    });

    if (isAvg && D.aggregate_stats) {
      ["pfn", "fm"].forEach(function (m) {
        var stat = D.aggregate_stats[m].sample_efficiency;
        var xs = D.alphas.slice();
        var q25 = xs.map(function (a) { return stat[a.toFixed(1)].q25; });
        var q75 = xs.map(function (a) { return stat[a.toFixed(1)].q75; });
        var bandColor = m === "pfn" ? "rgba(118,63,168,0.14)" : "rgba(209,161,31,0.14)";
        tracesA.push({
          x: xs, y: q75, mode: "lines", line: { color: "transparent" },
          showlegend: false, hoverinfo: "skip"
        });
        tracesA.push({
          x: xs, y: q25, mode: "lines", line: { color: "transparent" },
          fill: "tonexty", fillcolor: bandColor, showlegend: false, hoverinfo: "skip"
        });
      });
    }

    ["pfn", "fm"].forEach(function (m) {
      var s = seriesFor(m, "e");
      tracesA.push({
        x: s.x, y: s.y, mode: "lines+markers", name: D.models[m],
        line: { color: COLOR[m], width: 2.6, shape: "spline", smoothing: 0.6 },
        marker: {
          symbol: m === "pfn" ? "circle" : "square",
          size: s.x.map(function (v) { return v === alpha ? 12 : 7; }),
          color: COLOR[m],
          line: { color: "#ffffff", width: s.x.map(function (v) { return v === alpha ? 2.0 : 0.8; }) }
        },
        hoverinfo: "skip"
      });
    });

    var layoutA = {
      width: chartW,
      height: chartH,
      paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
      font: { color: "#8b9bb4", size: 11 },
      margin: marginPlot,
      xaxis: {
        title: { text: "Context budget (% of training data)", standoff: 10 },
        range: [0, 0.50], gridcolor: "#232c39", zeroline: false,
        tickvals: [0.1, 0.2, 0.3, 0.4, 0.5],
        ticktext: ["10%", "20%", "30%", "40%", "50%"],
        fixedrange: true
      },
      yaxis: {
        title: { text: "Effective training data (%)", standoff: 8 },
        range: [0, 1.00], gridcolor: "#232c39", zeroline: false,
        tickvals: [0, 0.2, 0.4, 0.6, 0.8, 1.0],
        ticktext: ["0%", "20%", "40%", "60%", "80%", "100%"],
        scaleanchor: "x", scaleratio: 1,
        fixedrange: true
      },
      legend: { orientation: "h", y: 1.03, yanchor: "bottom", x: 0, font: { size: 10.5 } },
      hovermode: false
    };

    // ----------------- Chart B: Distillation Efficiency (% gap closed) -----------------
    var tracesB = [];

    if (isAvg && D.aggregate_stats) {
      ["pfn", "fm"].forEach(function (m) {
        var stat = D.aggregate_stats[m].distillation_efficiency;
        var xs = D.alphas.slice();
        var q25 = xs.map(function (a) { return stat[a.toFixed(1)].q25; });
        var q75 = xs.map(function (a) { return stat[a.toFixed(1)].q75; });
        var bandColor = m === "pfn" ? "rgba(118,63,168,0.14)" : "rgba(209,161,31,0.14)";
        tracesB.push({
          x: xs, y: q75, mode: "lines", line: { color: "transparent" },
          showlegend: false, hoverinfo: "skip"
        });
        tracesB.push({
          x: xs, y: q25, mode: "lines", line: { color: "transparent" },
          fill: "tonexty", fillcolor: bandColor, showlegend: false, hoverinfo: "skip"
        });
      });
    }

    var allY_B = [];
    ["pfn", "fm"].forEach(function (m) {
      var s = seriesFor(m, "gc");
      allY_B = allY_B.concat(s.y);
      tracesB.push({
        x: s.x, y: s.y, mode: "lines+markers", name: D.models[m],
        line: { color: COLOR[m], width: 2.6, shape: "spline", smoothing: 0.6 },
        marker: {
          symbol: m === "pfn" ? "circle" : "square",
          size: s.x.map(function (v) { return v === alpha ? 12 : 7; }),
          color: COLOR[m],
          line: { color: "#ffffff", width: s.x.map(function (v) { return v === alpha ? 2.0 : 0.8; }) }
        },
        hoverinfo: "skip"
      });
    });

    var yRangeB = [0, 100];
    var yDtickB = 25;
    if (!isAvg && allY_B.length > 0) {
      var minY = Math.min.apply(null, allY_B);
      var maxY = Math.max.apply(null, allY_B);
      var low = Math.min(0, Math.floor(minY / 20) * 20);
      var high = Math.max(100, Math.ceil(maxY / 20) * 20);
      yRangeB = [low, high];
      yDtickB = (high - low) > 160 ? 50 : 25;
    }

    var layoutB = {
      width: chartW,
      height: chartH,
      paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
      font: { color: "#8b9bb4", size: 11 },
      margin: marginPlot,
      xaxis: {
        title: { text: "Context budget (% of training data)", standoff: 10 },
        range: [0, 0.50], gridcolor: "#232c39", zeroline: false,
        tickvals: [0.1, 0.2, 0.3, 0.4, 0.5],
        ticktext: ["10%", "20%", "30%", "40%", "50%"],
        fixedrange: true
      },
      yaxis: {
        title: { text: "Relative gap closed (%)", standoff: 8 },
        range: yRangeB, gridcolor: "#232c39", zeroline: false,
        dtick: yDtickB, fixedrange: true
      },
      legend: { orientation: "h", y: 1.03, yanchor: "bottom", x: 0, font: { size: 10.5 } },
      hovermode: false
    };

    var pA = Plotly.react("chart-a", tracesA, layoutA, {
      displayModeBar: false, responsive: true, scrollZoom: false, doubleClick: false, showAxisDragHandles: false
    });
    var pB = Plotly.react("chart-b", tracesB, layoutB, {
      displayModeBar: false, responsive: true, scrollZoom: false, doubleClick: false, showAxisDragHandles: false
    });

    Promise.all([pA, pB]).then(function () {
      updateBothGuideLines(alpha, isAnimated);
    });

    readout(alpha);
  }

  function readout(alpha) {
    var html = "";
    ["pfn", "fm"].forEach(function (m) {
      var sE = seriesFor(m, "e");
      var sGc = seriesFor(m, "gc");
      var iE = sE.x.indexOf(alpha);
      var iGc = sGc.x.indexOf(alpha);

      if (iE === -1 && iGc === -1) {
        html += '<div class="rcard ' + m + '">' +
          '<div class="rcard-head"><span class="m">' + D.models[m] + '</span></div>' +
          '<div class="v">-</div><div class="d">Not evaluated at this context budget</div></div>';
        return;
      }

      var e = iE !== -1 ? sE.y[iE] : null;
      var mult = e ? (e / alpha) : null;
      var gc = iGc !== -1 ? sGc.y[iGc] : null;

      var eHtml = e !== null
        ? (e * 100).toFixed(0) + "% <span class='x'>(" + mult.toFixed(1) + "×)</span>"
        : "-";

      var gcHtml = gc !== null
        ? gc.toFixed(0) + "%"
        : "-";

      var desc = "";
      if (e !== null && gc !== null) {
        desc = "aligned with " + Math.round(alpha * 100) + "% data ≈ unaligned with " + (e * 100).toFixed(0) + "% data; recovers " + gc.toFixed(0) + "% of teacher gap";
      } else if (e !== null) {
        desc = "aligned with " + Math.round(alpha * 100) + "% data ≈ unaligned with " + (e * 100).toFixed(0) + "% data";
      } else {
        desc = "Excluded by teacher superiority filter";
      }

      html += '<div class="rcard ' + m + '">' +
        '<div class="rcard-head"><span class="m">' + D.models[m] + '</span></div>' +
        '<div class="rcard-metrics">' +
          '<div class="metric-col"><span class="m-lbl">Effective data</span><span class="v">' + eHtml + '</span></div>' +
          '<div class="metric-col"><span class="m-lbl">Teacher gap closed</span><span class="v v-gc">' + gcHtml + '</span></div>' +
        '</div>' +
        '<div class="d">' + escapeHtml(desc) + '</div>' +
        '</div>';
    });
    $("#readout").innerHTML = html;
  }

  function debounce(fn, ms) {
    var t; return function () { clearTimeout(t); t = setTimeout(fn, ms); };
  }
})();
