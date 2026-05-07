// Recall — landing-page interactivity (v0.6).
//
//  1. Animated typed-edge graph in the hero card (Γ-walk traversal).
//  2. Ambient particle field drifting behind the demo graph.
//  3. Drifting demo-card footer numbers (CRC bound, λ_min, depth).
//  4. Scroll-progress bar at the very top.
//  5. Scroll-aware nav (blur intensifies + active link tracks scroll).
//  6. Reveal-on-scroll for all .reveal* elements (IntersectionObserver).
//  7. Counter tween for [data-count] when entering viewport.
//  8. Copy-to-clipboard on .copy-btn elements.
//  9. Subtle magnetic hover for .btn-primary.
//
// All vanilla — no libraries. Honors prefers-reduced-motion.

(function () {
  "use strict";

  const reduceMotion = window.matchMedia &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  // =========================================================================
  // 1 + 2 + 3 — Hero demo graph: Γ-walk + particles + drifting footer
  // =========================================================================
  function buildGraph() {
    const container = document.getElementById("demo-graph");
    if (!container) return;

    const ns = "http://www.w3.org/2000/svg";
    const W = container.clientWidth || 480;
    const H = container.clientHeight || 280;

    const svg = document.createElementNS(ns, "svg");
    svg.setAttribute("width", "100%");
    svg.setAttribute("height", "100%");
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svg.setAttribute("preserveAspectRatio", "xMidYMid meet");
    svg.style.display = "block";
    container.appendChild(svg);

    // Defs — gradients + glow filter
    const defs = document.createElementNS(ns, "defs");
    defs.innerHTML = `
      <radialGradient id="nodeGrad" cx="50%" cy="50%" r="50%">
        <stop offset="0%" stop-color="#b89cd9" />
        <stop offset="100%" stop-color="#6b4e8a" />
      </radialGradient>
      <radialGradient id="nodeGradKey" cx="50%" cy="50%" r="50%">
        <stop offset="0%" stop-color="#ede5f5" />
        <stop offset="100%" stop-color="#8b6dad" />
      </radialGradient>
      <filter id="softGlow" x="-50%" y="-50%" width="200%" height="200%">
        <feGaussianBlur in="SourceGraphic" stdDeviation="2" />
      </filter>
      <filter id="strongGlow" x="-50%" y="-50%" width="200%" height="200%">
        <feGaussianBlur in="SourceGraphic" stdDeviation="3.5" />
      </filter>
    `;
    svg.appendChild(defs);

    // Slightly richer graph than v0.5 — 8 nodes, 9 edges
    const nodes = [
      { id: "n1", x: 0.13, y: 0.34, role: "attempt",  label: "Postgres LISTEN/NOTIFY" },
      { id: "n2", x: 0.34, y: 0.16, role: "pivot",    label: "msg loss under load" },
      { id: "n3", x: 0.55, y: 0.50, role: "decision", label: "switched to Redis Streams", key: true },
      { id: "n4", x: 0.84, y: 0.32, role: "outcome",  label: "queue stable" },
      { id: "n5", x: 0.28, y: 0.74, role: "fact",     label: "PagerDuty alerts" },
      { id: "n6", x: 0.74, y: 0.78, role: "fact",     label: "consumer groups" },
      { id: "n7", x: 0.50, y: 0.88, role: "fact",     label: "Bun 1.1.4 monorepo" },
      { id: "n8", x: 0.92, y: 0.62, role: "outcome",  label: "50 K msg/sec" },
    ];

    const edges = [
      { src: "n1", dst: "n2", type: "pivots",     weight:  0.85 },
      { src: "n2", dst: "n3", type: "corrects",   weight:  0.78 },
      { src: "n5", dst: "n2", type: "supports",   weight:  0.62 },
      { src: "n3", dst: "n4", type: "supports",   weight:  0.72 },
      { src: "n6", dst: "n3", type: "supports",   weight:  0.55 },
      { src: "n1", dst: "n3", type: "superseded", weight: -0.45 },
      { src: "n7", dst: "n3", type: "supports",   weight:  0.41 },
      { src: "n3", dst: "n8", type: "supports",   weight:  0.66 },
      { src: "n4", dst: "n8", type: "agrees",     weight:  0.58 },
    ];

    const idMap = {};
    nodes.forEach(n => { idMap[n.id] = { x: n.x * W, y: n.y * H, ...n }; });

    // ---- Edges with subtle weight pulse ----
    const edgeEls = [];
    edges.forEach((e, i) => {
      const a = idMap[e.src], b = idMap[e.dst];
      const isNeg = e.weight < 0;
      const stroke = isNeg ? "#b9745f" : "#8b6dad";
      const dash   = isNeg ? "4 4"    : "0";
      const baseOpacity = isNeg ? 0.45 : 0.65;

      const line = document.createElementNS(ns, "path");
      const mx = (a.x + b.x) / 2;
      const my = (a.y + b.y) / 2 - 12;
      line.setAttribute("d", `M${a.x},${a.y} Q${mx},${my} ${b.x},${b.y}`);
      line.setAttribute("fill", "none");
      line.setAttribute("stroke", stroke);
      line.setAttribute("stroke-width", "1.5");
      line.setAttribute("stroke-dasharray", dash);
      line.setAttribute("opacity", "0");
      line.setAttribute("stroke-linecap", "round");
      svg.appendChild(line);
      setTimeout(() => {
        line.style.transition = "opacity 0.6s ease";
        line.setAttribute("opacity", baseOpacity.toString());
      }, 200 + i * 90);

      // Edge type label
      const lbl = document.createElementNS(ns, "text");
      lbl.setAttribute("x", mx);
      lbl.setAttribute("y", my - 4);
      lbl.setAttribute("font-family", "JetBrains Mono, monospace");
      lbl.setAttribute("font-size", "9");
      lbl.setAttribute("text-anchor", "middle");
      lbl.setAttribute("fill", stroke);
      lbl.setAttribute("opacity", "0");
      lbl.textContent = e.type;
      svg.appendChild(lbl);
      setTimeout(() => {
        lbl.style.transition = "opacity 0.5s";
        lbl.setAttribute("opacity", "0.8");
      }, 600 + i * 90);

      edgeEls.push({ line, lbl, baseOpacity, edge: e });
    });

    // Edge weight pulse — every ~3s, edges along the active walk path get
    // briefly brighter
    const pulsePath = ["n1", "n2", "n3", "n4"];
    const pulseEdgeKeys = new Set();
    for (let i = 0; i < pulsePath.length - 1; i++) {
      pulseEdgeKeys.add(pulsePath[i] + "→" + pulsePath[i + 1]);
    }
    if (!reduceMotion) {
      setInterval(() => {
        edgeEls.forEach(({ line, baseOpacity, edge }) => {
          const k = edge.src + "→" + edge.dst;
          if (pulseEdgeKeys.has(k)) {
            line.setAttribute("stroke-width", "2.5");
            line.setAttribute("opacity", "0.95");
            setTimeout(() => {
              line.setAttribute("stroke-width", "1.5");
              line.setAttribute("opacity", baseOpacity.toString());
            }, 700);
          }
        });
      }, 3200);
    }

    // ---- Nodes ----
    nodes.forEach((n, i) => {
      const g = document.createElementNS(ns, "g");
      g.setAttribute("transform", `translate(${n.x * W} ${n.y * H})`);
      g.style.opacity = 0;
      g.style.transition = "opacity 0.4s ease, transform 0.4s ease";

      const circle = document.createElementNS(ns, "circle");
      circle.setAttribute("r", n.key ? 11 : 8);
      circle.setAttribute("fill", n.key ? "url(#nodeGradKey)" : "url(#nodeGrad)");
      circle.setAttribute("stroke", n.key ? "#6b4e8a" : "transparent");
      circle.setAttribute("stroke-width", "1.5");
      g.appendChild(circle);

      if (n.key && !reduceMotion) {
        const pulse = document.createElementNS(ns, "circle");
        pulse.setAttribute("r", "11");
        pulse.setAttribute("fill", "transparent");
        pulse.setAttribute("stroke", "#8b6dad");
        pulse.setAttribute("stroke-width", "2");
        pulse.setAttribute("opacity", "0.6");
        const animR = document.createElementNS(ns, "animate");
        animR.setAttribute("attributeName", "r");
        animR.setAttribute("from", "11");
        animR.setAttribute("to", "22");
        animR.setAttribute("dur", "2.4s");
        animR.setAttribute("repeatCount", "indefinite");
        const animO = document.createElementNS(ns, "animate");
        animO.setAttribute("attributeName", "opacity");
        animO.setAttribute("from", "0.7");
        animO.setAttribute("to", "0");
        animO.setAttribute("dur", "2.4s");
        animO.setAttribute("repeatCount", "indefinite");
        pulse.appendChild(animR);
        pulse.appendChild(animO);
        g.appendChild(pulse);
      }

      const text = document.createElementNS(ns, "text");
      text.setAttribute("y", n.y * H > H * 0.7 ? -16 : 22);
      text.setAttribute("font-family", "Inter, sans-serif");
      text.setAttribute("font-size", "10.5");
      text.setAttribute("text-anchor", "middle");
      text.setAttribute("fill", "#2d2438");
      text.textContent = n.label;
      g.appendChild(text);

      svg.appendChild(g);
      setTimeout(() => { g.style.opacity = 1; }, 80 + i * 90);
    });

    // ---- Animated Γ-walk dot ----
    if (!reduceMotion) {
      const walkPath = ["n1", "n2", "n3", "n4"];
      const dot = document.createElementNS(ns, "circle");
      dot.setAttribute("r", "5");
      dot.setAttribute("fill", "#ede5f5");
      dot.setAttribute("stroke", "#6b4e8a");
      dot.setAttribute("stroke-width", "1.5");
      dot.setAttribute("opacity", "0");
      dot.setAttribute("filter", "url(#strongGlow)");
      svg.appendChild(dot);

      let segmentIdx = 0;
      let segmentStart = 0;
      const segmentDuration = 1100;
      const pauseBetweenLoops = 1800;

      function bezier(p0, p1, p2, t) {
        const u = 1 - t;
        return {
          x: u * u * p0.x + 2 * u * t * p1.x + t * t * p2.x,
          y: u * u * p0.y + 2 * u * t * p1.y + t * t * p2.y,
        };
      }

      function step(now) {
        if (!segmentStart) segmentStart = now;
        const t = Math.min(1, (now - segmentStart) / segmentDuration);
        const a = idMap[walkPath[segmentIdx]];
        const b = idMap[walkPath[segmentIdx + 1]];
        if (!a || !b) return;
        const mx = (a.x + b.x) / 2;
        const my = (a.y + b.y) / 2 - 12;
        const p = bezier({ x: a.x, y: a.y }, { x: mx, y: my }, { x: b.x, y: b.y }, t);
        dot.setAttribute("cx", p.x);
        dot.setAttribute("cy", p.y);
        dot.setAttribute("opacity", "0.95");
        if (t >= 1) {
          segmentIdx++;
          segmentStart = now;
          if (segmentIdx >= walkPath.length - 1) {
            dot.setAttribute("opacity", "0");
            segmentIdx = 0;
            segmentStart = now + pauseBetweenLoops;
            setTimeout(() => requestAnimationFrame(step), pauseBetweenLoops);
            return;
          }
        }
        requestAnimationFrame(step);
      }
      setTimeout(() => requestAnimationFrame(step), 1800);
    }

    // ---- Drifting footer numbers (CRC bound, λ_min, depth) ----
    if (!reduceMotion) {
      const footer = container.parentElement.querySelector(".demo-card-footer");
      if (footer) {
        const stats = footer.querySelectorAll(".demo-stat strong");
        // stats[0] = λ_min, stats[1] = depth, stats[2] = CRC bound
        let phase = 0;
        setInterval(() => {
          phase += 0.07;
          if (stats[0]) {
            const v = (Math.abs(Math.sin(phase * 0.7)) * 0.04).toFixed(2);
            stats[0].textContent = v;
          }
          if (stats[2]) {
            // wobble ±0.005 around 0.516
            const v = (0.516 + Math.sin(phase * 1.2) * 0.005).toFixed(3);
            stats[2].textContent = v;
          }
        }, 240);
      }
    }
  }

  // =========================================================================
  // Particle field drifting behind the demo graph
  // =========================================================================
  function buildParticles() {
    const host = document.getElementById("demo-graph");
    if (!host || reduceMotion) return;
    const layer = document.createElement("div");
    layer.className = "demo-particles";
    host.appendChild(layer);

    const ns = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(ns, "svg");
    svg.setAttribute("width", "100%");
    svg.setAttribute("height", "100%");
    svg.style.display = "block";
    layer.appendChild(svg);

    const W = host.clientWidth || 480;
    const H = host.clientHeight || 280;

    const N = 22;
    const particles = [];
    for (let i = 0; i < N; i++) {
      const c = document.createElementNS(ns, "circle");
      const r = 0.6 + Math.random() * 1.2;
      const x = Math.random() * W;
      const y = Math.random() * H;
      c.setAttribute("r", r.toString());
      c.setAttribute("cx", x);
      c.setAttribute("cy", y);
      c.setAttribute("fill", "#b89cd9");
      c.setAttribute("opacity", (0.18 + Math.random() * 0.18).toString());
      svg.appendChild(c);
      particles.push({
        el: c,
        x, y,
        vx: (Math.random() - 0.5) * 0.18,
        vy: -0.12 - Math.random() * 0.12, // drift up
      });
    }

    function tick() {
      particles.forEach(p => {
        p.x += p.vx;
        p.y += p.vy;
        if (p.y < -3) { p.y = H + 3; p.x = Math.random() * W; }
        if (p.x < -3) p.x = W + 3;
        if (p.x > W + 3) p.x = -3;
        p.el.setAttribute("cx", p.x);
        p.el.setAttribute("cy", p.y);
      });
      requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
  }

  // =========================================================================
  // 4 — Scroll-progress bar
  // =========================================================================
  function setupScrollProgress() {
    const bar = document.querySelector(".scroll-progress");
    if (!bar) return;
    const onScroll = () => {
      const h = document.documentElement;
      const max = h.scrollHeight - h.clientHeight;
      const pct = max > 0 ? (h.scrollTop / max) * 100 : 0;
      bar.style.width = pct + "%";
    };
    window.addEventListener("scroll", onScroll, { passive: true });
    onScroll();
  }

  // =========================================================================
  // 5 — Scroll-aware nav (blur + active link)
  // =========================================================================
  function setupNavScroll() {
    const nav = document.querySelector(".nav");
    if (!nav) return;
    const onScroll = () => {
      if (window.scrollY > 24) nav.classList.add("is-scrolled");
      else nav.classList.remove("is-scrolled");
    };
    window.addEventListener("scroll", onScroll, { passive: true });
    onScroll();

    // Active link tracking
    const sections = document.querySelectorAll("section[id]");
    const navLinks = document.querySelectorAll('.nav-links a[href^="#"]');
    if (!sections.length) return;

    const observer = new IntersectionObserver(entries => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          const id = entry.target.id;
          navLinks.forEach(a => {
            const isActive = a.getAttribute("href") === `#${id}`;
            a.classList.toggle("is-active", isActive);
          });
        }
      });
    }, { rootMargin: "-30% 0px -60% 0px" });
    sections.forEach(s => observer.observe(s));
  }

  // =========================================================================
  // 6 — Reveal on scroll (.reveal*, .reveal-stagger)
  // =========================================================================
  function setupReveals() {
    const targets = document.querySelectorAll(
      ".reveal, .reveal-up, .reveal-left, .reveal-right, .reveal-scale, .reveal-stagger"
    );
    if (!targets.length) return;

    // Stagger: assign a custom property to each child
    document.querySelectorAll(".reveal-stagger").forEach(parent => {
      Array.from(parent.children).forEach((child, i) => {
        child.style.setProperty("--stagger-i", i.toString());
      });
    });

    const observer = new IntersectionObserver(entries => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.classList.add("is-visible");
          observer.unobserve(entry.target);
        }
      });
    }, { rootMargin: "0px 0px -10% 0px", threshold: 0.05 });

    targets.forEach(t => observer.observe(t));
  }

  // =========================================================================
  // 7 — Counter animation for [data-count]
  // =========================================================================
  function setupCounters() {
    const targets = document.querySelectorAll("[data-count]");
    if (!targets.length) return;

    function easeOutCubic(t) { return 1 - Math.pow(1 - t, 3); }

    function animate(el) {
      const target = parseFloat(el.getAttribute("data-count"));
      const decimals = parseInt(el.getAttribute("data-decimals") || "0", 10);
      const prefix = el.getAttribute("data-prefix") || "";
      const suffix = el.getAttribute("data-suffix") || "";
      const duration = parseInt(el.getAttribute("data-duration") || "1400", 10);
      const start = performance.now();
      el.classList.add("is-counting");

      function step(now) {
        const t = Math.min(1, (now - start) / duration);
        const v = target * easeOutCubic(t);
        el.textContent = prefix + v.toFixed(decimals) + suffix;
        if (t < 1) {
          requestAnimationFrame(step);
        } else {
          el.classList.remove("is-counting");
          el.textContent = prefix + target.toFixed(decimals) + suffix;
        }
      }
      requestAnimationFrame(step);
    }

    if (reduceMotion) {
      // Just write the final value
      targets.forEach(el => {
        const target = parseFloat(el.getAttribute("data-count"));
        const decimals = parseInt(el.getAttribute("data-decimals") || "0", 10);
        const prefix = el.getAttribute("data-prefix") || "";
        const suffix = el.getAttribute("data-suffix") || "";
        el.textContent = prefix + target.toFixed(decimals) + suffix;
      });
      return;
    }

    const observer = new IntersectionObserver(entries => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          animate(entry.target);
          observer.unobserve(entry.target);
        }
      });
    }, { rootMargin: "0px 0px -10% 0px", threshold: 0.4 });

    targets.forEach(t => observer.observe(t));
  }

  // =========================================================================
  // 8 — Copy buttons
  // =========================================================================
  function setupCopyButtons() {
    document.querySelectorAll(".copy-btn").forEach(btn => {
      btn.addEventListener("click", async () => {
        const text = btn.dataset.copy;
        try {
          await navigator.clipboard.writeText(text);
        } catch {
          const ta = document.createElement("textarea");
          ta.value = text;
          document.body.appendChild(ta);
          ta.select();
          document.execCommand("copy");
          document.body.removeChild(ta);
        }
        const orig = btn.textContent;
        btn.classList.add("copied");
        btn.textContent = "✓ copied";
        setTimeout(() => {
          btn.classList.remove("copied");
          btn.textContent = orig;
        }, 1500);
      });
    });
  }

  // =========================================================================
  // 9 — Magnetic hover for the primary CTA
  // =========================================================================
  function setupMagneticButtons() {
    if (reduceMotion) return;
    document.querySelectorAll(".btn-primary").forEach(btn => {
      btn.addEventListener("mousemove", e => {
        const rect = btn.getBoundingClientRect();
        const x = e.clientX - rect.left - rect.width / 2;
        const y = e.clientY - rect.top - rect.height / 2;
        btn.style.transform = `translate(${x * 0.12}px, ${y * 0.12 - 2}px) scale(1.02)`;
      });
      btn.addEventListener("mouseleave", () => {
        btn.style.transform = "";
      });
    });
  }

  // =========================================================================
  // init
  // =========================================================================
  function init() {
    buildGraph();
    buildParticles();
    setupScrollProgress();
    setupNavScroll();
    setupReveals();
    setupCounters();
    setupCopyButtons();
    setupMagneticButtons();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
