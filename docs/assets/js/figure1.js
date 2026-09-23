/* Interactive version of Figure 1 (method overview) from the paper.
 *
 * Geometry is ported 1:1 from paper/figures/method_overview_body.tex so the
 * layout matches the published figure. TikZ y-axis points up, SVG's points
 * down, hence the Y() flip. Palette keeps the paper's hues but lifts the
 * luminance so it reads on the dark page background.
 */
(function () {
  "use strict";

  var SVGNS = "http://www.w3.org/2000/svg";
  var S = 40;                                   // px per TikZ unit
  var X = function (x) { return (x + 1.15) * S; };
  var Y = function (y) { return (2.90 - y) * S; };

  var C = {
    teal: "#4f9fd8", amber: "#cf6d17", green: "#2fb37e",
    gray: "#8b9bb4", ice: "#7cc0f0", flame: "#e87e22", core: "#f7c948",
    text: "#e6edf3", dim: "#5d6b80", panel: "rgba(255,255,255,.028)"
  };

  var yT = 1.50, yS = -0.55, BSHIFT = -3.34;    // teacher row, student row, panel (b)

  var STEPS = [
    { id: "syn",     label: "1 · Synthetic queries",
      text: "Unlabeled synthetic queries <em>X</em><sub>syn</sub> are generated from the feature distribution (using TabPFN's built-in unsupervised capabilities) and appended to both the full teacher context <em>C</em><sub>full</sub> and the reduced student context <em>C</em><sub>student</sub>." },
    { id: "teacher", label: "2 · Teacher pass",
      text: "The teacher evaluates the frozen foundation model <em>M</em> over the full context <em>C</em><sub>full</sub> together with the synthetic queries. Intermediate activations are tapped at layer <em>k</em> to produce the high-capacity target representation <em>h</em><sup>(<em>k</em>)</sup><sub>teacher</sub>, capturing rich in-context information across all available labeled training examples." },
    { id: "student", label: "3 · Student pass",
      text: "The student evaluates the exact same frozen model <em>M</em>, but only over a downsampled context <em>C</em><sub>student</sub> containing a fraction <em>α</em>·<em>N</em><sub>full</sub> of labeled examples (e.g. 10% context). Tapping activations at layer <em>k</em> yields <em>h</em><sup>(<em>k</em>)</sup><sub>student</sub> - substantially faster to compute due to shorter context, but lacking task context compared to the teacher." },
    { id: "align",   label: "4 · Fit the aligner",
      text: "A lightweight residual aligner <em>f</em><sub>θ</sub>(<em>h</em>) = <em>Wh</em> + <em>b</em> is trained by minimizing mean squared error against the teacher's activation: <em>L</em><sub>MSE</sub> = ‖<em>ĥ</em><sup>(<em>k</em>)</sup> − <em>h</em><sup>(<em>k</em>)</sup><sub>teacher</sub>‖<sup>2</sup>, where <em>ĥ</em><sup>(<em>k</em>)</sup> = <em>h</em> + <em>f</em><sub>θ</sub>(<em>h</em>). Only the aligner parameters θ are learned - solving in seconds on a single CPU - while the foundation model <em>M</em> remains completely frozen." },
    { id: "infer",   label: "5 · Inference",
      text: "At deployment time, only the fast student runs on real test queries <em>x</em><sub>q</sub> with the compact context <em>C</em><sub>student</sub>. Its layer-<em>k</em> activation is corrected in-place by the aligner (<em>ĥ</em><sup>(<em>k</em>)</sup> = <em>h</em><sup>(<em>k</em>)</sup> + <em>f</em><sub>θ</sub>(<em>h</em><sup>(<em>k</em>)</sup>)), and the remaining frozen layers <em>k</em>+1…<em>L</em> run unchanged to produce final predictions <em>ŷ</em><sub>q</sub>." }
  ];

  var svg, alphaRows = 2, currentStep = -1;

  /* ---------------- tiny SVG helpers ---------------- */
  function el(tag, attrs, parent) {
    var n = document.createElementNS(SVGNS, tag);
    for (var k in attrs) if (attrs[k] !== undefined) n.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(n);
    return n;
  }
  function rect(p, x, y, w, h, o) {
    o = o || {};
    return el("rect", {
      x: X(x), y: Y(y), width: w * S, height: h * S,
      rx: o.rx === undefined ? 1.5 : o.rx,
      fill: o.fill || "none", stroke: o.stroke || "none",
      "stroke-width": o.sw || 1
    }, p);
  }
  function text(p, x, y, str, o) {
    o = o || {};
    var n = el("text", {
      x: X(x), y: Y(y), fill: o.fill || C.text,
      "font-size": o.size || 8.5,
      "font-family": o.mono ? "ui-monospace,Menlo,monospace" : "inherit",
      "font-style": o.italic ? "italic" : null,
      "font-weight": o.bold ? 650 : null,
      "text-anchor": o.anchor || "middle"
    }, p);
    n.textContent = str;
    return n;
  }
  /* math label: base + optional superscript + optional subscript */
  function mathLabel(p, x, y, base, sup, sub, o) {
    o = o || {};
    var size = o.size || 8.5;
    var n = el("text", {
      x: X(x), y: Y(y), fill: o.fill || C.text, "font-size": size,
      "font-family": "Georgia,'Times New Roman',serif", "font-style": "italic",
      "text-anchor": o.anchor || "middle"
    }, p);
    var t1 = el("tspan", {}, n); t1.textContent = base;
    if (sup) {
      var t2 = el("tspan", { dy: -size * 0.42, "font-size": size * 0.72, "font-style": "normal" }, n);
      t2.textContent = sup;
    }
    if (sub) {
      var t3 = el("tspan", {
        dy: sup ? size * 0.42 + size * 0.28 : size * 0.28,
        "font-size": size * 0.72, "font-style": "normal"
      }, n);
      t3.textContent = sub;
    }
    return n;
  }
  function arrow(p, pts, color, o) {
    o = o || {};
    var d = pts.map(function (pt, i) {
      return (i ? "L" : "M") + X(pt[0]).toFixed(1) + "," + Y(pt[1]).toFixed(1);
    }).join(" ");
    return el("path", {
      d: d, fill: "none", stroke: color, "stroke-width": o.sw || 1.15,
      "stroke-dasharray": o.dash || null,
      "marker-end": o.noHead ? null : "url(#f1-head-" + o.head + ")",
      "stroke-linejoin": "round"
    }, p);
  }

  /* ---------------- pics ported from the TikZ styles ---------------- */
  function snowflake(p, x, y) {
    var g = el("g", { transform: "translate(" + X(x) + "," + Y(y) + ")" }, p);
    for (var a = 0; a < 360; a += 60) {
      var r = a * Math.PI / 180, L = 7;
      el("line", {
        x1: 0, y1: 0, x2: Math.cos(r) * L, y2: -Math.sin(r) * L,
        stroke: C.ice, "stroke-width": .8, "stroke-linecap": "round"
      }, g);
      [4.7, 2.4].forEach(function (d) {
        [50, -50].forEach(function (off) {
          var br = (a + off) * Math.PI / 180, bl = d === 4.7 ? 2.4 : 1.6;
          el("line", {
            x1: Math.cos(r) * d, y1: -Math.sin(r) * d,
            x2: Math.cos(r) * d + Math.cos(br) * bl,
            y2: -Math.sin(r) * d - Math.sin(br) * bl,
            stroke: C.ice, "stroke-width": .65, "stroke-linecap": "round"
          }, g);
        });
      });
    }
    return g;
  }
  function flame(p, x, y) {
    var g = el("g", { transform: "translate(" + X(x) + "," + Y(y) + ")" }, p);
    el("path", {
      d: "M0,-8 C6,-2.4 5.4,2.2 0,6.4 C-5.4,2.2 -6,-2.4 0,-8 Z", fill: C.flame
    }, g);
    el("path", {
      d: "M0,-2.9 C3.1,-0.2 2.8,2.5 0,5.8 C-2.8,2.5 -3.1,-0.2 0,-2.9 Z", fill: C.core
    }, g);
    return g;
  }

  /* data table: nCtx labeled rows + nQry unlabeled query rows */
  function dtable(p, cx, cy, nCtx, nQry, accent, ctxLab, qryLab) {
    var g = el("g", {}, p);
    var hL = 0.2 * nCtx, hQ = 0.2 * nQry;
    var total = hL + 0.09 + hQ;
    var oy = cy + total / 2;                      // pic origin (top edge)

    for (var i = 1; i <= nCtx; i++) {
      var ry = oy - 0.2 * i + 0.2;
      rect(g, 0, ry, 0.95, 0.2, { fill: accent + "1f", stroke: accent + "66", sw: .5, rx: 0 });
      rect(g, 1.0, ry - 0.025, 0.2, 0.15, { fill: accent + "b3", rx: 1 });
    }
    [0.317, 0.633].forEach(function (sx) {
      el("line", { x1: X(sx), y1: Y(oy), x2: X(sx), y2: Y(oy - hL),
        stroke: accent + "4d", "stroke-width": .5 }, g);
    });

    var qTop = oy - hL - 0.09;
    for (var j = 1; j <= nQry; j++) {
      var qy = qTop - 0.2 * j + 0.2;
      rect(g, 0, qy, 0.95, 0.2, { fill: C.gray + "14", stroke: C.gray + "73", sw: .5, rx: 0 });
      var yb = qy - 0.175;                        // dotted "unknown label" box
      [0, .05, .10, .15, .20].forEach(function (dx) {
        el("circle", { cx: X(1.0 + dx), cy: Y(yb), r: .7, fill: C.gray }, g);
        el("circle", { cx: X(1.0 + dx), cy: Y(yb + .15), r: .7, fill: C.gray }, g);
      });
      [.05, .10].forEach(function (dy) {
        el("circle", { cx: X(1.0), cy: Y(yb + dy), r: .7, fill: C.gray }, g);
        el("circle", { cx: X(1.2), cy: Y(yb + dy), r: .7, fill: C.gray }, g);
      });
    }
    [0.317, 0.633].forEach(function (sx) {
      el("line", { x1: X(sx), y1: Y(qTop), x2: X(sx), y2: Y(qTop - hQ),
        stroke: C.gray + "4d", "stroke-width": .5 }, g);
    });

    ctxLab(g, -0.06, oy - hL / 2);
    qryLab(g, -0.06, qTop - hQ / 2);
    return g;
  }

  /* transformer stack, layers 1..k with layer k highlighted */
  function stackk(p, x, y, accent) {
    var g = el("g", {}, p);
    [0, 0.50, 1.00].forEach(function (bx) {
      rect(g, x + bx, y + 0.5, 0.40, 1.0, { fill: C.gray + "17", stroke: C.gray + "73", sw: .7, rx: 1.5 });
    });
    text(g, x + 1.68, y - 0.09, "···", { fill: C.gray, size: 9 });
    rect(g, x + 1.96, y + 0.5, 0.40, 1.0, { fill: accent + "2b", stroke: accent, sw: 1.3, rx: 1.5 });
    return g;
  }
  /* transformer stack, layers k+1..L */
  function stackr(p, x, y) {
    var g = el("g", {}, p);
    rect(g, x, y + 0.5, 0.40, 1.0, { fill: C.gray + "17", stroke: C.gray + "73", sw: .7, rx: 1.5 });
    text(g, x + 0.675, y - 0.09, "···", { fill: C.gray, size: 9 });
    [0.95, 1.45].forEach(function (bx) {
      rect(g, x + bx, y + 0.5, 0.40, 1.0, { fill: C.gray + "17", stroke: C.gray + "73", sw: .7, rx: 1.5 });
    });
    return g;
  }
  /* activation vector strip */
  function strip(p, x, y, accent) {
    var g = el("g", {}, p);
    var shades = ["9e", "33", "e0", "61", "21"];
    shades.forEach(function (a, i) {
      rect(g, x, y + 0.5 - i * 0.2, 0.3, 0.2, { fill: accent + a, rx: 0 });
    });
    rect(g, x, y + 0.5, 0.3, 1.0, { stroke: accent + "bf", sw: .75, rx: 0 });
    return g;
  }

  /* ---------------- build ---------------- */
  function build() {
    var host = document.getElementById("fig1");
    host.innerHTML = "";
    svg = el("svg", {
      viewBox: "0 -14 610 320", role: "img",
      "aria-label": "Activation alignment method overview"
    }, host);

    var defs = el("defs", {}, svg);
    [["t", C.teal], ["s", C.amber], ["g", C.green], ["n", C.gray]].forEach(function (h) {
      var m = el("marker", {
        id: "f1-head-" + h[0], markerWidth: 6, markerHeight: 6,
        refX: 5.2, refY: 2, orient: "auto"
      }, defs);
      el("path", { d: "M0,0 L0,4 L5.2,2 z", fill: h[1] }, m);
    });

    /* panel backgrounds + titles */
    rect(svg, -1.15, 2.76, 15.25, 4.26, { fill: C.panel, rx: 5 });
    rect(svg, -1.15, -1.78, 15.25, 2.45, { fill: C.panel, rx: 5 });
    text(svg, -1.15, 2.84, "Offline · Fit the aligner",
      { anchor: "start", bold: true, size: 10.5 });
    text(svg, -1.15, -1.86, "Inference · Correct the activation",
      { anchor: "start", bold: true, size: 10.5 });

    /* ============ panel (a) - teacher row ============ */
    var gTTable = group("teacher-table");
    dtable(gTTable, 0, yT, 6, 2, C.teal,
      function (g, x, y) { mathLabel(g, x, y - 0.03, "C", null, "full", { anchor: "end", fill: C.gray }); },
      function (g, x, y) { synLabel(g, x, y); });

    var gT = group("teacher");
    arrow(gT, [[1.30, yT], [2.08, yT]], C.teal, { head: "t" });
    stackk(gT, 2.15, yT, C.teal);
    frozenM(gT, 3.33, yT + 0.74);
    text(gT, 4.31, yT - 0.78, "layer k", { fill: C.gray, size: 8 });
    arrow(gT, [[4.58, yT], [5.30, yT]], C.teal, { head: "t" });
    var gHT = group("hk-teacher");
    strip(gHT, 5.35, yT, C.teal);
    mathLabel(gHT, 5.50, yT + 0.60, "h", "(k)", "teacher", { fill: C.teal });

    /* ============ panel (a) - student row ============ */
    var gSTable = group("student-table");
    studentTable(gSTable, yS, alphaRows, 2, false);

    var gS = group("student");
    arrow(gS, [[1.30, yS], [2.08, yS]], C.amber, { head: "s" });
    stackk(gS, 2.15, yS, C.amber);
    frozenM(gS, 3.33, yS + 0.74);
    text(gS, 4.31, yS - 0.78, "layer k", { fill: C.gray, size: 8 });
    arrow(gS, [[4.58, yS], [5.30, yS]], C.amber, { head: "s" });
    var gHS = group("hk-student");
    strip(gHS, 5.35, yS, C.amber);
    mathLabel(gHS, 5.50, yS + 0.60, "h", "(k)", "student", { fill: C.amber });

    /* ============ panel (a) - aligner chain ============ */
    var gA = group("align");
    arrow(gA, [[5.65, yS], [6.56, yS]], C.amber, { head: "s" });
    el("circle", { cx: X(6.15), cy: Y(yS), r: 2, fill: C.amber }, gA);
    alignerBox(gA, 6.60, yS, true);
    arrow(gA, [[8.85, yS], [9.22, yS]], C.green, { head: "g" });
    oplus(gA, 9.40, yS);
    arrow(gA, [[6.15, yS], [6.15, yS - 0.80], [9.40, yS - 0.80], [9.40, yS - 0.20]],
      C.amber, { head: "s" });
    arrow(gA, [[9.58, yS], [9.95, yS]], C.green, { head: "g" });
    strip(gA, 10.00, yS, C.green);
    mathLabel(gA, 10.15, yS + 0.60, "ĥ", "(k)", null, { fill: C.green });

    /* MSE loss + dashed target paths */
    var gL = group("align");
    arrow(gL, [[5.65, yT], [12.10, yT], [12.10, 0.72]], C.gray, { dash: "3 3", sw: .9, head: "n" });
    arrow(gL, [[10.30, yS], [12.10, yS], [12.10, 0.22]], C.gray, { dash: "3 3", sw: .9, head: "n" });
    rect(gL, 11.62, 0.72, 0.96, 0.5, { fill: "#161d27", stroke: C.gray + "99", sw: .7, rx: 3 });
    mathLabel(gL, 12.10, 0.40, "L", null, "MSE", { fill: C.text, size: 9 });
    text(gL, 8.55, yT + 0.07, "target", { fill: C.gray, size: 7.5 });

    /* ============ panel (b) ============ */
    var gB = group("infer");
    studentTable(gB, BSHIFT, alphaRows, 1, true);
    arrow(gB, [[1.30, BSHIFT], [2.08, BSHIFT]], C.amber, { head: "s" });
    stackk(gB, 2.15, BSHIFT, C.amber);
    frozenM(gB, 3.33, BSHIFT + 0.74);
    text(gB, 3.33, BSHIFT - 0.78, "layers 1…k", { fill: C.gray, size: 8 });
    arrow(gB, [[4.58, BSHIFT], [5.30, BSHIFT]], C.amber, { head: "s" });
    strip(gB, 5.35, BSHIFT, C.amber);
    mathLabel(gB, 5.50, BSHIFT + 0.60, "h", "(k)", null, { fill: C.amber });
    arrow(gB, [[5.65, BSHIFT], [6.56, BSHIFT]], C.amber, { head: "s" });
    el("circle", { cx: X(6.15), cy: Y(BSHIFT), r: 2, fill: C.amber }, gB);
    alignerBox(gB, 6.60, BSHIFT, false);
    arrow(gB, [[8.85, BSHIFT], [9.22, BSHIFT]], C.green, { head: "g" });
    oplus(gB, 9.40, BSHIFT);
    arrow(gB, [[6.15, BSHIFT], [6.15, BSHIFT - 0.80], [9.40, BSHIFT - 0.80], [9.40, BSHIFT - 0.20]],
      C.amber, { head: "s" });
    arrow(gB, [[9.58, BSHIFT], [9.95, BSHIFT]], C.green, { head: "g" });
    strip(gB, 10.00, BSHIFT, C.green);
    mathLabel(gB, 10.15, BSHIFT + 0.60, "ĥ", "(k)", null, { fill: C.green });
    arrow(gB, [[10.30, BSHIFT], [10.88, BSHIFT]], C.green, { head: "g" });
    stackr(gB, 10.93, BSHIFT);
    frozenM(gB, 11.86, BSHIFT + 0.74);
    text(gB, 11.90, BSHIFT - 0.78, "layers k+1…L", { fill: C.gray, size: 8 });
    arrow(gB, [[12.86, BSHIFT], [13.32, BSHIFT]], C.gray, { head: "n" });
    rect(gB, 13.34, BSHIFT + 0.24, 0.66, 0.48, { fill: C.gray + "14", stroke: C.gray + "8c", sw: .6, rx: 4 });
    mathLabel(gB, 13.67, BSHIFT - 0.09, "ŷ", null, "q", { fill: C.text, size: 9 });

    applyStep();
  }

  /* shared sub-builders ------------------------------------------------ */
  function group(step) {
    return el("g", { class: "f1-g", "data-step": step }, svg);
  }
  function synLabel(g, x, y) {
    var n = mathLabel(g, x, y - 0.03, "X", null, "syn", { anchor: "end", fill: C.gray });
    n.setAttribute("class", "f1-syn");
  }
  function frozenM(g, x, y) {
    mathLabel(g, x + 0.12, y, "M", null, null, { fill: C.gray, size: 10 });
    snowflake(g, x - 0.52, y + 0.07);
  }
  function alignerLabel(p, x, y, anchor) {
    var size = 8;
    var n = el("text", {
      x: X(x), y: Y(y), fill: C.green, "font-size": size,
      "text-anchor": anchor || "middle"
    }, p);
    var t1 = el("tspan", {}, n); t1.textContent = "aligner f";
    var t2 = el("tspan", { dy: size * 0.28, "font-size": size * 0.72 }, n); t2.textContent = "θ";
    return n;
  }
  function alignerBox(g, x, y, withFlame) {
    rect(g, x, y + 0.31, 2.25, 0.62, { fill: C.green + "1f", stroke: C.green, sw: 1.15, rx: 3 });
    mathLabel(g, x + 1.125, y - 0.09, "h ↦ Wh + b", null, null, { size: 9, fill: C.text });
    var lbl;
    if (withFlame) {
      flame(g, x + 0.52, y + 0.49);
      lbl = alignerLabel(g, x + 0.88, y + 0.42, "start");
    } else {
      lbl = alignerLabel(g, x + 1.125, y + 0.42, "middle");
    }
    return lbl;
  }
  function oplus(g, x, y) {
    el("circle", { cx: X(x), cy: Y(y), r: 7, fill: "#161d27", stroke: C.green, "stroke-width": 1.2 }, g);
    text(g, x, y - 0.075, "+", { fill: C.green, size: 10, bold: true });
  }
  function studentTable(g, cy, nCtx, nQry, isInfer) {
    return dtable(g, 0, cy, nCtx, nQry, C.amber,
      function (gg, x, y) { mathLabel(gg, x, y - 0.03, "C", null, "student", { anchor: "end", fill: C.gray }); },
      function (gg, x, y) {
        if (isInfer) mathLabel(gg, x, y - 0.03, "x", null, "q", { anchor: "end", fill: C.gray });
        else synLabel(gg, x, y);
      });
  }

  /* ---------------- interactivity ---------------- */
  function applyStep() {
    var isAll = (currentStep < 0 || currentStep >= STEPS.length);
    var groups = svg.querySelectorAll(".f1-g");
    [].forEach.call(groups, function (g) {
      var s = g.dataset.step;
      var on = isAll;

      if (!isAll) {
        if (currentStep === 0) {
          // Phase 1: input tables (teacher and student tables with synthetic queries) are colored
          on = (s === "teacher-table" || s === "student-table");
        } else if (currentStep === 1) {
          // Phase 2: teacher pass (table, transformer stack, and layer-k activation)
          on = (s === "teacher-table" || s === "teacher" || s === "hk-teacher");
        } else if (currentStep === 2) {
          // Phase 3: student pass (table, transformer stack, layer-k activation)
          on = (s === "student-table" || s === "student" || s === "hk-student");
        } else if (currentStep === 3) {
          // Phase 4: aligner, MSE loss, plus both teacher and student activations
          on = (s === "align" || s === "hk-teacher" || s === "hk-student");
        } else if (currentStep === 4) {
          // Phase 5: inference
          on = (s === "infer");
        }
      }

      g.style.opacity = on ? 1 : 0.16;
    });

    [].forEach.call(svg.querySelectorAll(".f1-syn"), function (n) {
      var lit = (currentStep === 0 || isAll);
      n.style.fill = lit ? C.text : C.gray;
      n.style.fontWeight = lit ? 700 : 400;
    });
    var cap = document.getElementById("fig1-cap");
    if (isAll) {
      cap.innerHTML = "";
    } else {
      cap.innerHTML = STEPS[currentStep].text;
    }
    [].forEach.call(document.querySelectorAll("#fig1-steps .step-nav-item"), function (b, i) {
      b.classList.toggle("on", i === currentStep);
    });
  }
  function setStep(i) { currentStep = i; applyStep(); }
 
  var STEP_BOUNDS = [
    { min: 0.05, max: 0.23, center: 0.14 },
    { min: 0.23, max: 0.42, center: 0.32 },
    { min: 0.42, max: 0.61, center: 0.51 },
    { min: 0.61, max: 0.80, center: 0.70 },
    { min: 0.80, max: 0.95, center: 0.87 }
  ];

  /* ---------------- init ---------------- */
  document.addEventListener("DOMContentLoaded", function () {
    var bar = document.getElementById("fig1-steps");
    bar.innerHTML = "";
    STEPS.forEach(function (s, i) {
      var item = document.createElement("div");
      item.className = "step-nav-item" + (i === currentStep ? " on" : "");
      item.setAttribute("role", "button");
      item.setAttribute("tabindex", "0");
      item.setAttribute("aria-label", s.label);

      var cleanTitle = s.label.replace(/^\d+\s*·\s*/, "");
      var idxStr = "0" + (i + 1);

      item.innerHTML =
        '<span class="step-nav-pip"></span>' +
        '<div class="step-nav-text">' +
          '<span class="step-nav-idx">' + idxStr + '</span>' +
          '<span class="step-nav-title">' + cleanTitle + '</span>' +
        '</div>';

      item.addEventListener("click", function () {
        var track = document.getElementById("method-track");
        if (track) {
          var rect = track.getBoundingClientRect();
          var maxScroll = track.offsetHeight - window.innerHeight;
          var trackTop = rect.top + window.scrollY;
          var targetFrac = STEP_BOUNDS[i].center;
          var targetY = trackTop + targetFrac * maxScroll;
          window.scrollTo({ top: targetY, behavior: "smooth" });
        } else {
          setStep(i);
        }
      });
      bar.appendChild(item);
    });

    build();
    initScrolly();
  });

  function initScrolly() {
    var track = document.getElementById("method-track");
    if (!track) return;

    function getStepFromScroll() {
      var rect = track.getBoundingClientRect();
      var maxScroll = track.offsetHeight - window.innerHeight;
      if (maxScroll <= 0) return -1;

      var scrolled = -rect.top;
      var frac = scrolled / maxScroll;

      if (frac < 0.05) return -1;
      if (frac >= 0.95) return 5;

      for (var i = 0; i < STEP_BOUNDS.length; i++) {
        if (frac < STEP_BOUNDS[i].max) return i;
      }
      return 5;
    }

    var ticking = false;
    function onScroll() {
      if (!ticking) {
        requestAnimationFrame(function () {
          var step = getStepFromScroll();
          if (step !== currentStep) {
            setStep(step);
          }
          ticking = false;
        });
        ticking = true;
      }
    }

    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", onScroll, { passive: true });
    onScroll();
  }
})();
