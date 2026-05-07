# Recall — Landing Page

Static landing page for `recall.dev`.

## Structure

```
landing/
├── index.html      # all markup, semantic HTML5
├── style.css       # cream/beige theme with aubergine purple accents
└── script.js       # animated typed-edge graph + copy-to-clipboard + nav highlight
```

No build step. Pure HTML/CSS/JS. Two web fonts loaded from Google Fonts
(Source Serif 4, Inter, JetBrains Mono).

## Local preview

```bash
cd landing
python3 -m http.server 8080
open http://localhost:8080
```

## Deployment

### Cloudflare Pages (recommended, free tier)

1. Connect the repo to Cloudflare Pages
2. Set build output directory: `landing/`
3. No build command needed
4. Custom domain: `recall.dev` → CNAME to your Pages URL

### Vercel (free tier)

1. Import repo
2. Set root directory: `landing/`
3. Framework preset: "Other"
4. No build command, no output directory override

### Netlify (free tier)

1. New site from Git
2. Base directory: `landing/`
3. Publish directory: `landing/`
4. No build command

### GitHub Pages

```bash
git subtree push --prefix landing origin gh-pages
```

Or copy the three files to a `gh-pages` branch root.

## Design

### Color palette

| Variable | Value | Use |
|---|---|---|
| `--cream` | #faf6ed | Page background |
| `--cream-deep` | #f3ecdc | Section alternating bg |
| `--cream-soft` | #fdfaf3 | Card backgrounds |
| `--tan` | #d6c4a8 | Borders |
| `--aubergine` | #6b4e8a | Primary accent |
| `--aubergine-dark` | #4d3868 | Hover state |
| `--aubergine-mid` | #8b6dad | Secondary accent |
| `--lavender` | #b89cd9 | Tertiary accent / gradients |
| `--mist` | #ede5f5 | Soft purple wash |
| `--ink` | #2d2438 | Primary text (deep purple-grey) |
| `--ink-soft` | #4a3f55 | Body text |
| `--ink-faint` | #75687f | Captions |

### Typography

- **Headings**: Source Serif 4 (700) — refined, academic, slightly literary
- **Body**: Inter (400/500/600) — clean, modern, readable
- **Code/eyebrow**: JetBrains Mono — technical, deliberate

### Sections

1. **Nav** — sticky, glass-blur background
2. **Hero** — title + tagline + animated typed-edge graph
3. **What** — 3 cards explaining the data model
4. **How** — 4 numbered steps with code snippets
5. **Benchmarks** — 6 cards with real reproducible numbers
6. **Research** — 4 cards crediting professors + papers
7. **Install** — 4 install paths (lib / MCP / CLI / Docker)
8. **Distinguished** — 6-bullet "what nothing else does"
9. **Commitment** — Apache-2.0 forever pledge
10. **Footer** — links + legal

### Responsive

- Desktop: 2-column hero, 3-card grids
- Tablet (≤800px): 2-card grids
- Mobile (≤500px): single column

### Accessibility

- Semantic HTML5 (`<main>`, `<nav>`, `<section>`, `<article>`)
- Prefers-reduced-motion respected via CSS animation-only effects
- ARIA labels on copy buttons
- Color contrast WCAG AA on all body text

## Updating

When the README changes substantively:
1. Update hero tagline if the project description shifts
2. Update benchmark numbers in the bench-grid section
3. Update the install snippets in install-grid
4. Re-deploy (one `git push` if connected to Pages/Vercel)
