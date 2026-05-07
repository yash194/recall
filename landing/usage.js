// Recall — usage page interactivity (v0.6).
//
//  1. Scroll-progress bar at the top.
//  2. Scroll-aware nav (blur on scroll).
//  3. Reveal-on-scroll for .reveal-up etc.
//  4. Tab switching for MCP-client config blocks.
//  5. TOC scrollspy — highlight current section in the sidebar.
//  6. Per-<pre> copy buttons (auto-injected on every code block).
//
// All vanilla. Honors prefers-reduced-motion.

(function () {
  "use strict";

  const reduceMotion = window.matchMedia &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  // ---------- Scroll progress ----------
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

  // ---------- Nav scroll-state ----------
  function setupNavScroll() {
    const nav = document.querySelector(".nav");
    if (!nav) return;
    const onScroll = () => {
      if (window.scrollY > 24) nav.classList.add("is-scrolled");
      else nav.classList.remove("is-scrolled");
    };
    window.addEventListener("scroll", onScroll, { passive: true });
    onScroll();
  }

  // ---------- Reveal-on-scroll ----------
  function setupReveals() {
    const targets = document.querySelectorAll(
      ".reveal, .reveal-up, .reveal-left, .reveal-right, .reveal-scale, .reveal-stagger"
    );
    if (!targets.length) return;

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

  // ---------- Tabs ----------
  function setupTabs() {
    document.querySelectorAll("[data-tabs]").forEach(group => {
      const buttons = group.querySelectorAll(".tab-btn");
      const panels = group.querySelectorAll(".tab-panel");
      buttons.forEach(btn => {
        btn.addEventListener("click", () => {
          const target = btn.getAttribute("data-tab");
          buttons.forEach(b => b.classList.toggle("is-active", b === btn));
          panels.forEach(p => p.classList.toggle(
            "is-active", p.getAttribute("data-tab") === target,
          ));
        });
      });
    });
  }

  // ---------- TOC scrollspy ----------
  function setupTocSpy() {
    const tocLinks = document.querySelectorAll(".toc a[href^='#']");
    if (!tocLinks.length) return;
    const idToLink = {};
    tocLinks.forEach(a => {
      const id = a.getAttribute("href").slice(1);
      idToLink[id] = a;
    });
    const sections = Array.from(document.querySelectorAll(
      ".doc-content h2[id], .doc-content [id].path-picker"
    ));

    const observer = new IntersectionObserver(entries => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          const id = entry.target.id;
          tocLinks.forEach(a => a.classList.remove("is-active"));
          if (idToLink[id]) idToLink[id].classList.add("is-active");
        }
      });
    }, { rootMargin: "-20% 0px -70% 0px" });

    sections.forEach(s => observer.observe(s));

    // Smooth-scroll on TOC click (small offset for sticky nav)
    tocLinks.forEach(a => {
      a.addEventListener("click", e => {
        const href = a.getAttribute("href");
        const target = document.querySelector(href);
        if (!target) return;
        e.preventDefault();
        const top = target.getBoundingClientRect().top + window.scrollY - 90;
        window.scrollTo({
          top,
          behavior: reduceMotion ? "auto" : "smooth",
        });
        history.replaceState(null, "", href);
      });
    });
  }

  // ---------- Auto copy buttons on every <pre> ----------
  function setupCopyButtons() {
    document.querySelectorAll(".doc-content pre").forEach(pre => {
      const code = pre.querySelector("code");
      if (!code) return;
      const btn = document.createElement("button");
      btn.className = "pre-copy";
      btn.textContent = "copy";
      btn.setAttribute("aria-label", "Copy code to clipboard");
      btn.addEventListener("click", async () => {
        const text = code.textContent;
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
        btn.classList.add("copied");
        btn.textContent = "✓ copied";
        setTimeout(() => {
          btn.classList.remove("copied");
          btn.textContent = "copy";
        }, 1500);
      });
      pre.appendChild(btn);
    });
  }

  // ---------- init ----------
  function init() {
    setupScrollProgress();
    setupNavScroll();
    setupReveals();
    setupTabs();
    setupTocSpy();
    setupCopyButtons();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
