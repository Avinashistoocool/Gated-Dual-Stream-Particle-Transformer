// Source: https://avinashistoocool.github.io/app.js (eventDisplay, hero canvas)
(() => {
  "use strict";
  const $ = (s, r = document) => r.querySelector(s);

  /* ---------------- hero: toy collision event display (transverse view) ---------------- */
  function eventDisplay() {
    const cv = $("#jet");
    const ctx = cv.getContext("2d");
    const hero = cv.parentElement;
    const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
    const CLASSES = ["H→bb̄", "H→cc̄", "H→gg", "H→4q", "H→ℓνqq'", "t→bqq'", "t→bℓν", "W→qq'", "Z→qq̄", "QCD"];
    const PRONGS = [2, 2, 2, 4, 2, 3, 2, 2, 2, 1];
    const LEPTON = [0, 0, 0, 0, 1, 0, 1, 0, 0, 0];
    const TOWERS = 72;
    const T_FLASH = 0.35, T_TRACKS = 1.8, T_HOLD = 9, T_FADE = 10;
    let W, H, dpr, C, R, ev = null, t0 = 0, evNo = 41270 + Math.floor(Math.random() * 900);
    const rnd = (a, b) => a + Math.random() * (b - a);
    const gauss = () => { let u = 0, v = 0; while (!u) u = Math.random(); while (!v) v = Math.random(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
    const L = { pipe: 0.035, trk: [0.09, 0.15, 0.22, 0.3, 0.38], trkOut: 0.44, ecal: [0.47, 0.57], hcal: [0.61, 0.79], mu: [0.85, 0.96] };

    function layout() {
      dpr = Math.min(devicePixelRatio || 1, 2);
      W = cv.clientWidth; H = cv.clientHeight;
      cv.width = W * dpr; cv.height = H * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const landscapePhone = W > H && H < 520;
      if (W > 1024) {
        // laptop / desktop: detector on the right, text on the left
        C = { x: W * 0.7, y: H * 0.46 };
        R = Math.min(W * 0.26, H * 0.36);
      } else if (landscapePhone) {
        C = { x: W * 0.78, y: H * 0.55 };
        R = Math.min(W * 0.17, H * 0.34);
      } else if (W > 600) {
        // tablet portrait
        C = { x: W * 0.5, y: H * 0.3 };
        R = Math.min(W * 0.3, H * 0.2);
      } else {
        // phone portrait: detector above the headline
        C = { x: W * 0.5, y: Math.max(160, H * 0.28) };
        R = Math.min(W * 0.38, H * 0.19);
      }
      if (ev) ev = null; // rebuild trajectories for the new radius
    }

    function track(phi, pt, q, stopR) {
      // charged particle in a solenoid field: circle of radius rc through the origin
      const rc = pt * 0.55;
      const pts = [];
      const n = 70;
      const cx = -q * rc * Math.sin(phi), cy = q * rc * Math.cos(phi);
      const a0 = Math.atan2(-cy, -cx);
      for (let i = 0; i <= n; i++) {
        const s = (i / n) * Math.min(Math.PI * 1.6, (stopR * 1.6) / rc);
        const a = a0 + q * s;
        const x = cx + rc * Math.cos(a), y = cy + rc * Math.sin(a);
        pts.push([x, y]);
        if (Math.hypot(x, y) >= stopR) break;
      }
      const [ex, ey] = pts[pts.length - 1];
      return { pts, exit: Math.hypot(ex, ey) >= stopR * 0.98 ? Math.atan2(ey, ex) : null };
    }

    function newEvent() {
      const cls = Math.floor(Math.random() * CLASSES.length);
      const nj = Math.random() < 0.5 ? 2 : 3;
      const ecal = new Float32Array(TOWERS), hcal = new Float32Array(TOWERS), parts = [], muons = [];
      const jets = [];
      const phiLead = rnd(0, Math.PI * 2);
      for (let j = 0; j < nj; j++) {
        const axis = phiLead + (j === 0 ? 0 : j === 1 ? Math.PI + gauss() * 0.35 : rnd(1.2, 2.2) * (Math.random() < 0.5 ? 1 : -1));
        const ptJet = j === 0 ? rnd(0.9, 1.4) : rnd(0.45, 1.0);
        const prongs = j === 0 ? PRONGS[cls] : 1 + Math.floor(Math.random() * 2);
        const width = j === 0 ? 0.08 + 0.05 * prongs : rnd(0.08, 0.16);
        const subs = Array.from({ length: prongs }, () => axis + gauss() * width);
        const n = Math.round(rnd(9, 16) * (j === 0 ? 1.3 : 1));
        jets.push({ axis, pt: ptJet, width: width * 2.2 });
        for (let k = 0; k < n; k++) {
          const sub = subs[k % prongs];
          const phi = sub + gauss() * 0.045;
          const pt = ptJet * Math.exp(-rnd(0, 2.6)) * rnd(0.6, 1.4);
          const kind = Math.random();
          if (kind < 0.62) {
            const q = Math.random() < 0.5 ? -1 : 1;
            const tr = track(phi, Math.max(0.12, pt) * R, q, L.trkOut * R);
            const calo = Math.random() < 0.3 ? "e" : "h";
            parts.push({ type: "ch", ...tr, pt, q, calo, delay: rnd(0, 0.35), jet: j });
            if (tr.exit !== null) deposit(ecal, hcal, tr.exit, pt, calo);
          } else if (kind < 0.85) {
            parts.push({ type: "ph", phi, pt, r1: L.ecal[0] * R, delay: rnd(0, 0.35), jet: j });
            deposit(ecal, hcal, phi, pt, "e");
          } else {
            parts.push({ type: "nh", phi, pt, r1: L.hcal[0] * R, delay: rnd(0, 0.35) });
            deposit(ecal, hcal, phi, pt, "h");
          }
        }
      }
      if (LEPTON[cls] || Math.random() < 0.35) {
        const phi = rnd(0, Math.PI * 2);
        const q = Math.random() < 0.5 ? -1 : 1;
        const tr = track(phi, rnd(1.6, 3) * R, q, L.mu[1] * R);
        muons.push({ ...tr, q, pt: rnd(25, 80), delay: 0.1 });
      }
      for (let k = 0; k < 18; k++) {
        const tr = track(rnd(0, Math.PI * 2), rnd(0.05, 0.22) * R, Math.random() < 0.5 ? -1 : 1, L.trkOut * R);
        parts.push({ type: "soft", ...tr, pt: 0.02, delay: rnd(0, 0.6) });
      }
      // classifier read-out: the "true" class gets most of the probability
      const logits = CLASSES.map((_, i) => (i === cls ? rnd(2.4, 3.6) : gauss() * 0.7));
      const ex = logits.map(Math.exp), z = ex.reduce((a, b) => a + b, 0);
      const probs = ex.map((v, i) => ({ i, p: v / z })).sort((a, b) => b.p - a.p).slice(0, 3);
      const nTracks = parts.filter((p) => p.type !== "ph" && p.type !== "nh").length + muons.length;
      // the hardest charged tracks get labelled callouts
      const labelled = parts.filter((p) => p.type === "ch" && p.exit !== null).sort((a, b) => b.pt - a.pt).slice(0, 3);
      labelled.forEach((p, k) => {
        p.label = `${p.q > 0 ? "h⁺" : "h⁻"}  pT ${(p.pt * 95 + 8).toFixed(0)} GeV`;
        p.k = k;
      });
      // straight extrapolation from the tracker edge into the calorimeter for every exiting track
      for (const p of parts) {
        if (p.type !== "ch" || p.exit === null) continue;
        const n = p.pts.length;
        const [x1, y1] = p.pts[n - 1], [x0, y0] = p.pts[Math.max(0, n - 4)];
        const a = Math.atan2(y1 - y0, x1 - x0);
        const stop = (p.calo === "e" ? L.ecal[1] : L.hcal[1] * 0.9) * R;
        p.ext = [];
        for (let s = 0; s < 80; s++) {
          const x = x1 + Math.cos(a) * s * 4, y = y1 + Math.sin(a) * s * 4;
          p.ext.push([x, y]);
          if (Math.hypot(x, y) >= stop) break;
        }
      }
      ev = { cls, jets, parts, muons, ecal, hcal, probs, nTracks, labelled, ht: jets.reduce((s, j) => s + j.pt, 0) * 210 };
      evNo += 1 + Math.floor(Math.random() * 40);
    }

    function deposit(ecal, hcal, phi, e, where) {
      const i = ((Math.round((phi / (Math.PI * 2)) * TOWERS) % TOWERS) + TOWERS) % TOWERS;
      if (where === "e") { ecal[i] += e; ecal[(i + 1) % TOWERS] += e * 0.25; ecal[(i + TOWERS - 1) % TOWERS] += e * 0.25; }
      else { hcal[i] += e * 0.8; ecal[i] += e * 0.15; hcal[(i + 1) % TOWERS] += e * 0.3; hcal[(i + TOWERS - 1) % TOWERS] += e * 0.3; }
    }

    function ring(r, w, style) {
      ctx.strokeStyle = style; ctx.lineWidth = w;
      ctx.beginPath(); ctx.arc(C.x, C.y, r, 0, Math.PI * 2); ctx.stroke();
    }

    function drawDetector(glow) {
      // faint grid backdrop
      ctx.strokeStyle = "rgba(255,255,255,0.035)"; ctx.lineWidth = 1;
      const step = Math.max(48, Math.round(W / 26));
      for (let x = (C.x % step); x < W; x += step) { ctx.beginPath(); ctx.moveTo(x + .5, 0); ctx.lineTo(x + .5, H); ctx.stroke(); }
      for (let y = (C.y % step); y < H; y += step) { ctx.beginPath(); ctx.moveTo(0, y + .5); ctx.lineTo(W, y + .5); ctx.stroke(); }
      const g = ctx.createRadialGradient(C.x, C.y, 0, C.x, C.y, R * 1.25);
      g.addColorStop(0, `rgba(82,168,255,${0.07 + glow * 0.18})`);
      g.addColorStop(1, "rgba(82,168,255,0)");
      ctx.fillStyle = g; ctx.fillRect(0, 0, W, H);
      ring(L.pipe * R, 1, "rgba(255,255,255,0.35)");
      for (const r of L.trk) ring(r * R, 1, "rgba(255,255,255,0.10)");
      // calorimeter and muon system as segmented annuli
      const seg = (r0, r1, n, fill, stroke) => {
        for (let i = 0; i < n; i++) {
          const a0 = (i / n) * Math.PI * 2 + 0.004, a1 = ((i + 1) / n) * Math.PI * 2 - 0.004;
          ctx.beginPath(); ctx.arc(C.x, C.y, r1, a0, a1); ctx.arc(C.x, C.y, r0, a1, a0, true); ctx.closePath();
          ctx.fillStyle = fill; ctx.fill();
          if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = 1; ctx.stroke(); }
        }
      };
      seg(L.ecal[0] * R, L.ecal[1] * R, TOWERS, "rgba(255,255,255,0.025)");
      seg(L.hcal[0] * R, L.hcal[1] * R, TOWERS / 2, "rgba(255,255,255,0.03)");
      seg(L.mu[0] * R, L.mu[1] * R, 12, "rgba(255,255,255,0.02)", "rgba(255,255,255,0.08)");
      ring(L.ecal[0] * R, 1, "rgba(255,255,255,0.14)");
      ring(L.hcal[1] * R, 1, "rgba(255,255,255,0.14)");
    }

    function polyline(pts, frac) {
      const n = Math.max(1, Math.floor((pts.length - 1) * frac));
      ctx.beginPath();
      ctx.moveTo(C.x + pts[0][0], C.y + pts[0][1]);
      for (let i = 1; i <= n; i++) ctx.lineTo(C.x + pts[i][0], C.y + pts[i][1]);
      ctx.stroke();
      return pts[n];
    }

    function drawEvent(t, alpha) {
      const grow = (d) => Math.max(0, Math.min(1, (t - T_FLASH * 0.6 - d) / (T_TRACKS - T_FLASH)));
      ctx.lineCap = "round";
      // jet cones
      for (const j of ev.jets) {
        const f = grow(0.1);
        if (!f) continue;
        const r = L.hcal[1] * R * (0.35 + 0.65 * f);
        ctx.fillStyle = `rgba(82,168,255,${0.05 * alpha})`;
        ctx.beginPath(); ctx.moveTo(C.x, C.y); ctx.arc(C.x, C.y, r, j.axis - j.width / 2, j.axis + j.width / 2); ctx.closePath(); ctx.fill();
        ctx.strokeStyle = `rgba(82,168,255,${0.22 * alpha})`; ctx.lineWidth = 1; ctx.setLineDash([4, 6]);
        ctx.beginPath(); ctx.moveTo(C.x, C.y); ctx.lineTo(C.x + Math.cos(j.axis) * r * 1.08, C.y + Math.sin(j.axis) * r * 1.08); ctx.stroke();
        ctx.setLineDash([]);
      }
      // particles
      for (const p of ev.parts) {
        const f = grow(p.delay);
        if (!f) continue;
        if (p.type === "ch" || p.type === "soft") {
          const hi = p.pt > 0.35;
          ctx.strokeStyle = p.type === "soft" ? `rgba(153,153,153,${0.45 * alpha})` : hi ? `rgba(237,237,237,${0.95 * alpha})` : `rgba(82,168,255,${0.9 * alpha})`;
          ctx.lineWidth = p.type === "soft" ? 1 : hi ? 2 : 1.5;
          const head = polyline(p.pts, f);
          if (f < 1) { ctx.fillStyle = ctx.strokeStyle; ctx.beginPath(); ctx.arc(C.x + head[0], C.y + head[1], 2.6, 0, 7); ctx.fill(); }
          // dotted continuation into the calorimeter, drawn once the track reaches the tracker edge
          if (f >= 1 && p.ext) {
            const ef = Math.max(0, Math.min(1, (t - T_TRACKS) / 0.5));
            ctx.save();
            ctx.setLineDash([2, 3]); ctx.lineWidth = 1;
            ctx.strokeStyle = p.calo === "e" ? `rgba(237,237,237,${0.45 * alpha})` : `rgba(82,168,255,${0.55 * alpha})`;
            polyline(p.ext, ef);
            ctx.restore();
          }
        } else {
          const r = p.r1 * f;
          ctx.strokeStyle = p.type === "ph" ? `rgba(237,237,237,${0.5 * alpha})` : `rgba(153,153,153,${0.55 * alpha})`;
          ctx.lineWidth = 1; ctx.setLineDash(p.type === "ph" ? [2, 4] : [6, 4]);
          ctx.beginPath(); ctx.moveTo(C.x, C.y); ctx.lineTo(C.x + Math.cos(p.phi) * r, C.y + Math.sin(p.phi) * r); ctx.stroke();
          ctx.setLineDash([]);
        }
      }
      for (const m of ev.muons) {
        const f = grow(m.delay);
        if (!f) continue;
        ctx.strokeStyle = `rgba(98,192,115,${alpha})`; ctx.lineWidth = 2.2;
        polyline(m.pts, f);
        if (f >= 1 && m.exit !== null) {
          const a = m.exit, r0 = L.mu[0] * R, r1 = L.mu[1] * R;
          ctx.fillStyle = `rgba(98,192,115,${0.5 * alpha})`;
          ctx.beginPath(); ctx.arc(C.x, C.y, r1, a - 0.08, a + 0.08); ctx.arc(C.x, C.y, r0, a + 0.08, a - 0.08, true); ctx.closePath(); ctx.fill();
        }
      }
      // calorimeter towers light up after the tracks arrive
      const cf = Math.max(0, Math.min(1, (t - T_TRACKS * 0.7) / 0.6)) * alpha;
      if (cf > 0) {
        const tower = (arr, r0, r1, rgb, gain) => {
          for (let i = 0; i < TOWERS; i++) {
            const e = arr[i];
            if (e < 0.02) continue;
            const a0 = (i / TOWERS) * Math.PI * 2 - Math.PI / TOWERS, a1 = a0 + (Math.PI * 2) / TOWERS - 0.006;
            const len = Math.min(1, e * gain);
            const re = r0 + (r1 - r0) * len;
            ctx.fillStyle = `rgba(${rgb},${(0.35 + 0.6 * len) * cf})`;
            ctx.beginPath(); ctx.arc(C.x, C.y, re, a0, a1); ctx.arc(C.x, C.y, r0, a1, a0, true); ctx.closePath(); ctx.fill();
            if (len > 0.75) {
              ctx.fillStyle = `rgba(${rgb},${0.3 * cf})`;
              ctx.beginPath(); ctx.arc(C.x, C.y, re + R * 0.05 * (len - 0.6), a0, a1); ctx.arc(C.x, C.y, re, a1, a0, true); ctx.closePath(); ctx.fill();
            }
          }
        };
        tower(ev.ecal, L.ecal[0] * R, L.ecal[1] * R, "237,237,237", 2.2);
        tower(ev.hcal, L.hcal[0] * R, L.hcal[1] * R, "82,168,255", 1.6);
      }
      // vertex flash
      if (t < T_FLASH * 2) {
        const k = 1 - t / (T_FLASH * 2);
        const g = ctx.createRadialGradient(C.x, C.y, 0, C.x, C.y, R * 0.25);
        g.addColorStop(0, `rgba(255,255,255,${0.9 * k})`);
        g.addColorStop(1, "rgba(255,255,255,0)");
        ctx.fillStyle = g; ctx.beginPath(); ctx.arc(C.x, C.y, R * 0.25, 0, 7); ctx.fill();
      }
      drawLabels(t, alpha);
    }

    // leader lines + text tags pointing at the actual trajectories
    function drawLabels(t, alpha) {
      const la = Math.max(0, Math.min(1, (t - T_TRACKS - 0.3) / 0.6)) * alpha;
      if (la <= 0) return;
      const fs = W < 560 ? 10.5 : 12;
      ctx.font = `${fs}px 'Geist Mono', ui-monospace, monospace`;
      ctx.textBaseline = "middle";
      const placed = [];
      const tag = (ax, ay, ang, text, color, dist) => {
        // anchor (ax, ay) on the trajectory, tag pushed outward along ang
        let ex = C.x + Math.cos(ang) * dist, ey = C.y + Math.sin(ang) * dist;
        for (const q of placed) if (Math.abs(q - ey) < fs + 6) ey += (ey > C.y ? 1 : -1) * (fs + 8);
        placed.push(ey);
        const right = Math.cos(ang) >= 0;
        const tw = ctx.measureText(text).width;
        let hx = right ? ex + 18 : ex - 18;
        if (right && hx + tw + 10 > W) hx = W - tw - 10;
        if (!right && hx - tw - 10 < 0) hx = tw + 10;
        ctx.strokeStyle = color.replace("A", String(0.6 * la)); ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(ex, ey); ctx.lineTo(hx, ey); ctx.stroke();
        ctx.fillStyle = color.replace("A", String(la));
        ctx.beginPath(); ctx.arc(ax, ay, 2.5, 0, 7); ctx.fill();
        ctx.fillStyle = `rgba(0,0,0,${0.55 * la})`;
        ctx.fillRect(right ? hx + 2 : hx - tw - 8, ey - fs / 2 - 3, tw + 6, fs + 6);
        ctx.fillStyle = color.replace("A", String(la));
        ctx.textAlign = right ? "left" : "right";
        ctx.fillText(text, right ? hx + 5 : hx - 5, ey);
      };
      const outer = L.mu[1] * R + (W < 560 ? 10 : 28);
      const compact = W < 700 || H < 520;
      // hardest charged tracks (full labels only where there is room)
      for (const p of compact ? ev.labelled.slice(0, 1) : ev.labelled) {
        const [x, y] = p.pts[Math.floor(p.pts.length * 0.7)];
        tag(C.x + x, C.y + y, Math.atan2(y, x), compact ? p.label.split("  ")[0] + " track" : p.label, "rgba(237,237,237,A)", outer);
      }
      // jets: labelled at their axis outside the detector
      (compact ? ev.jets.slice(0, 1) : ev.jets).forEach((j, k) => {
        const r = L.hcal[1] * R;
        const name = k === 0 ? (compact ? "lead jet" : `lead jet · ${PRONGS[ev.cls]}-prong`) : `jet ${k + 1}`;
        tag(C.x + Math.cos(j.axis) * r, C.y + Math.sin(j.axis) * r, j.axis + 0.12, compact ? name : `${name} · ${(j.pt * 210).toFixed(0)} GeV`, "rgba(82,168,255,A)", outer + 14);
      });
      // muons reach the outer chambers
      for (const m of ev.muons) {
        if (m.exit === null) continue;
        const r = L.mu[1] * R;
        tag(C.x + Math.cos(m.exit) * r, C.y + Math.sin(m.exit) * r, m.exit - 0.1, compact ? (m.q > 0 ? "μ⁺" : "μ⁻") : `${m.q > 0 ? "μ⁺" : "μ⁻"} · ${m.pt.toFixed(0)} GeV`, "rgba(98,192,115,A)", outer + 4);
      }
      // primary vertex
      if (!compact) tag(C.x, C.y, -Math.PI * 0.62, "primary vertex", "rgba(153,153,153,A)", R * 0.2);
      ctx.textAlign = "left";
      ctx.textBaseline = "alphabetic";
    }

    function drawHud(t, alpha) {
      if (W <= 1024) return;
      const x = Math.max(20, Math.min(56, W * 0.04)), y = 150;
      ctx.font = "12px 'Geist Mono', ui-monospace, monospace";
      ctx.textBaseline = "alphabetic";
      ctx.fillStyle = "rgba(153,153,153,0.95)";
      ctx.fillText(`EVENT ${String(evNo).padStart(7, "0")}`, x, y);
      ctx.fillText(`${ev.nTracks} tracks · ${ev.jets.length} jets · HT ${Math.round(ev.ht)} GeV`, x, y + 20);
      const ready = Math.max(0, Math.min(1, (t - T_TRACKS) / 0.8)) * alpha;
      if (ready <= 0) { ctx.fillText("reconstructing…", x, y + 48); return; }
      ctx.fillText("jet tagger · lead jet", x, y + 48);
      ev.probs.forEach((p, k) => {
        const yy = y + 70 + k * 20;
        const top = k === 0;
        ctx.fillStyle = top ? `rgba(237,237,237,${ready})` : `rgba(153,153,153,${ready})`;
        ctx.fillText(CLASSES[p.i], x, yy);
        ctx.fillStyle = "rgba(255,255,255,0.08)";
        ctx.fillRect(x + 78, yy - 8, 120, 6);
        ctx.fillStyle = top ? `rgba(82,168,255,${ready})` : `rgba(255,255,255,${0.3 * ready})`;
        ctx.fillRect(x + 78, yy - 8, 120 * p.p * ready, 6);
        ctx.fillStyle = `rgba(153,153,153,${ready})`;
        ctx.fillText(p.p.toFixed(2), x + 210, yy);
      });
    }

    function frame(now) {
      if (!ev || (now - t0) / 1000 > T_FADE) { newEvent(); t0 = now; }
      const t = reduce ? T_HOLD - 0.1 : (now - t0) / 1000;
      const alpha = t < T_HOLD ? 1 : Math.max(0, 1 - (t - T_HOLD) / (T_FADE - T_HOLD));
      ctx.clearRect(0, 0, W, H);
      drawDetector(t < T_FLASH * 2 ? 1 - t / (T_FLASH * 2) : 0);
      drawEvent(t, alpha);
      drawHud(t, alpha);
      if (!reduce) requestAnimationFrame(frame);
    }

    layout();
    let lastW = W;
    addEventListener("resize", () => {
      // mobile browsers fire resize when the URL bar hides; only rebuild on real size changes
      if (Math.abs(cv.clientWidth - lastW) < 2 && Math.abs(cv.clientHeight - H) < 120) return;
      lastW = cv.clientWidth;
      layout();
      if (reduce) requestAnimationFrame(frame);
    });
    const fire = () => { ev = null; if (reduce) requestAnimationFrame(frame); };
    cv.addEventListener("pointerdown", (e) => { if (e.isPrimary) fire(); });
    const btn = $("#new-event");
    if (btn) btn.addEventListener("click", fire);
    requestAnimationFrame(frame);
  }
  eventDisplay();
})();
