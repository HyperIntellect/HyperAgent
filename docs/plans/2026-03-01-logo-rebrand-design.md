# HyperAgent Logo Rebrand Design

**Date:** 2026-03-01
**Status:** Approved

## Summary

Rebrand the HyperAgent logo from the current 2x2 geometric grid (3 squares + 1 circle) to a modern geometric "H" monogram with an agent node accent on a violet background.

## Design Specification

### Logo Mark

A geometric "H" lettermark on a rounded-square background, with a small filled circle ("agent node") at the crossbar center.

**Construction:**
- 128x128 viewBox, rounded-square container with rx="28"
- Two vertical bars (H pillars) — filled rectangles
- Horizontal crossbar — slightly above center for visual balance
- Small filled circle (agent node) at crossbar center

**Colors (Light theme):**
- Background: `#1E293B` (slate-800)
- H letterform: `#FFFFFF` (white)
- Agent node: `#0891B2` (cyan-600 — brand teal)

**Colors (Dark theme):**
- Background: `#334155` (slate-700)
- H letterform: `#FFFFFF` (white)
- Agent node: `#22D3EE` (cyan-400 — brand bright teal)

### Files Updated

1. `web/public/images/logo-dark.svg`
2. `web/public/images/logo-light.svg`
3. `public/images/logo-dark.svg`
4. `public/images/logo-light.svg`

### What Stays the Same

- Product name "HyperAgent"
- Brand typography (DM Sans 700, -0.04em letter-spacing)
- UI theme color architecture (semantic token system)
- Logo dimensions and rendering in components
- Component code (all reference existing SVG paths)
