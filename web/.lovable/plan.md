# Crucible — Direction Proposal (Section 1)

Per your brief, I'm proposing the token system _first_, before touching pages. Nothing below gets built until you confirm or redirect.

## Direction: "Signal Readout"

Grounded in what Crucible actually is: a graph executor that emits measurable signals — node order, tensor shapes, per-op latency, byte counts. The site should feel like a **lab instrument's readout panel**, not a landing page. Think oscilloscope trace, spectrum analyzer, wire-format hex dump — but rendered light, precise, and calm, not neon or "cyberpunk."

The signature isn't a color or a font — it's that **live measurements are the primary visual element on every page**, and static marketing text is subordinate to them.

### Color (6 tokens, all light-first)

| Token       | Hex       | Role                                                                                             |
| ----------- | --------- | ------------------------------------------------------------------------------------------------ |
| `--surface` | `#FAFAF7` | Page background — bone white, faint warm cast, not cream                                         |
| `--panel`   | `#FFFFFF` | Instrument panel / card surface                                                                  |
| `--rule`    | `#DCDCD3` | 1px hairlines, tick marks, grid                                                                  |
| `--ink`     | `#0E0E10` | Primary text, axis labels                                                                        |
| `--ink-dim` | `#5A5A55` | Secondary text, units                                                                            |
| `--trace`   | `#1F3A8A` | The one accent — deep indigo signal trace, used for live values, active graph edges, primary CTA |
| `--warn`    | `#B45309` | Amber, only for warn/review states                                                               |
| `--ok`      | `#166534` | Deep green, only for pass/legitimate states                                                      |

No terracotta. No cream. No dark surfaces except a **single monospace readout strip** where hex/binary bytes are shown (that's the wire-format decoder's native habitat — it earns the dark treatment).

### Type (3 faces, one role each)

- **JetBrains Mono** — all numbers, all data, all axis labels, all node IDs. Tabular numerals globally.
- **Söhne / Inter Tight** (whichever loads clean) — UI, nav, body. Tight tracking, no serif.
- **No display serif.** The "editorial serif headline" is exactly the AI-generated pattern to avoid. Headlines are set in the same sans as body, one size up, medium weight, tight leading. The mono numbers do the visual heavy lifting.

### Layout

**One-sentence concept:** every page is an instrument panel — a labeled bezel (route name, unit, timestamp) wrapping a live readout area, with static prose set as marginalia in a narrow left gutter.

Rough wireframe:

```text
┌─ CRUCIBLE ──────── /architecture ── 14:32:07 UTC ─┐
│ gutter │  READOUT AREA                            │
│ 200px  │  (graph, chart, hex dump, live number —  │
│ prose  │   the actual thing this page is about)   │
│        │                                          │
└────────┴──────────────────────────────────────────┘
```

Hairline rules at 1px `--rule`. No rounded corners above 2px. No shadows — depth comes from rules and whitespace only.

### Signature element

A persistent **top-of-page bezel strip** on every route: monospace, shows route name, a live-updating tick counter (real `performance.now()`), and — where relevant — the current page's headline metric (e.g. on `/benchmark` it shows `p50 = 14.2ms`, on `/fraud` it shows the last inference's probability). The bezel is the one thing that recurs everywhere and makes the site instantly recognizable as Crucible.

## Critique against the three AI-tell patterns

- **Cream + serif + terracotta?** No — bone white, no serif, indigo accent.
- **Near-black + acid accent?** No — light-first, dark reserved only for the hex-dump strip where it's diegetically correct.
- **Broadsheet columns?** No — instrument-panel bezel, not newspaper. Hairlines are present but they frame live data, not stacked prose columns.
- **Generic AI infra startup?** The signature (live bezel + mono-first typography + prose-as-marginalia) is specific to a runtime that emits measurements. It would look wrong for a generic SaaS.

## Build order once confirmed

1. **Tokens** — rewrite `src/styles/crucible.css` to the palette above; swap fonts in `__root.tsx`.
2. **Bugs first** (Section 3): fix `§` glyph in Fraud empty state; correct "Eigen / WASM-SIMD" mislabel on `/benchmark` (split into **Crucible Native (C++/Eigen)** and **Crucible WASM (Rust/SIMD128)** rows/cards); add GitHub link to nav; unify primary button color; kill duplicate three-card rows on home.
3. **Bezel component** — the signature strip, mounted in `CrucibleLayout`.
4. **Homepage hero** — replace the two 3-card rows with a live graph-execution visualization (reuse the Playground's real node list, animate Kahn's-order execution with per-node latency ticking in).
5. **New routes**: `/architecture` (diagram + NCHW + Kahn's + data-path comparison), `/story` (first-person, the 50MB/750MB constraint, protobuf decoder, C++/Rust parity), `/roadmap` (typed `RoadmapItem[]`, Shipped/In Progress/Planned).
6. **Footer rebuild** per Section 4E.
7. **Favicon + OG** from the bezel mark.
8. **Responsive pass** on Fraud + Playground two-column layouts.

All placeholder data lands as typed objects in `src/data/*.ts` so real numbers swap in cleanly.

---

Confirm the "Signal Readout" direction (or point me at what to change) and I'll start with tokens + Section 3 bugs.
