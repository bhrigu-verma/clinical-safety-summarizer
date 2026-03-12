---
name: "ClinicalSafe Frontend Design"
description: "Design system, component patterns, animation library, and page architecture reference for the ClinicalSafe clinical research showcase frontend."
---

# ClinicalSafe Frontend Design Skill

## Overview
This skill provides the complete design system, animation patterns, and architectural conventions for building frontend pages in the ClinicalSafe clinical safety table summarization project.

---

## 1. Technology Stack

| Layer | Technology | Version |
|-------|-----------|---------|
| Framework | Next.js (App Router) | 16.1.6 |
| UI Library | React | 19.2.3 |
| Language | TypeScript | 5.x |
| Styling | Tailwind CSS | 4.x |
| Animation | Framer Motion | 12.34.2 |
| Icons | Lucide React | 0.575.0 |
| Primitives | Radix UI | latest |
| Component Kit | shadcn/ui | latest |
| Utilities | clsx, tailwind-merge, class-variance-authority | latest |

---

## 2. Design Tokens

### Colors
```
Background:          #0a0a0f (near-black)
Surface:             white/5 (glassmorphism base)
Border:              white/10
Text Primary:        white
Text Secondary:      white/60
Text Muted:          white/40

Accent Blue:         blue-500 → blue-600 (ML pipeline)
Accent Purple:       purple-500 → purple-600 (DL pipeline)
Accent Emerald:      emerald-500 → emerald-600 (Fine-tuned)
Accent Amber:        amber-500 → amber-600 (Compare mode)
Success:             green-500
Danger:              red-500
```

### Typography
```
Font Family:         "Outfit", sans-serif
Heading XL:          text-4xl md:text-5xl font-bold tracking-tight
Heading LG:          text-2xl md:text-3xl font-bold
Heading MD:          text-xl font-semibold
Body:                text-base text-white/80 leading-relaxed
Caption:             text-sm text-white/50
Mono:                font-mono text-sm (for numbers, code snippets)
```

### Spacing
```
Section gap:         space-y-24 / py-24
Card padding:        p-6 (default), p-8 (featured)
Grid gap:            gap-4 (tight), gap-6 (default), gap-8 (spacious)
Container:           max-w-7xl mx-auto px-4
Border radius:       rounded-xl (buttons), rounded-2xl (cards), rounded-3xl (hero)
```

---

## 3. Core CSS Utilities

These are defined in `frontend/app/globals.css`:

```css
@utility glass {
  background: rgba(255, 255, 255, 0.03);
  backdrop-filter: blur(20px);
  border: 1px solid rgba(255, 255, 255, 0.08);
}

@utility gradient-text {
  background: linear-gradient(135deg, #fff 0%, rgba(255,255,255,0.6) 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

@utility minimal-card {
  background: rgba(255, 255, 255, 0.03);
  backdrop-filter: blur(12px);
  border: 1px solid rgba(255, 255, 255, 0.06);
  border-radius: 1rem;
  padding: 1.5rem;
}
```

---

## 4. Component Patterns

### Glassmorphism Card
```tsx
<motion.div
  className="bg-white/5 backdrop-blur-xl border border-white/10 rounded-2xl p-6"
  initial={{ opacity: 0, y: 20 }}
  whileInView={{ opacity: 1, y: 0 }}
  viewport={{ once: true }}
  transition={{ duration: 0.5 }}
>
  {children}
</motion.div>
```

### Gradient CTA Button
```tsx
<motion.button
  className="bg-gradient-to-r from-blue-600 to-purple-600 text-white px-6 py-3 rounded-xl font-medium"
  whileHover={{ scale: 1.02 }}
  whileTap={{ scale: 0.98 }}
>
  {label}
</motion.button>
```

### Stat Card
```tsx
<motion.div
  className="minimal-card text-center"
  initial={{ opacity: 0, scale: 0.9 }}
  whileInView={{ opacity: 1, scale: 1 }}
  viewport={{ once: true }}
  transition={{ type: "spring", stiffness: 200, delay }}
>
  <div className="text-3xl font-bold gradient-text">{value}</div>
  <div className="text-sm text-white/50 mt-1">{label}</div>
</motion.div>
```

### Icon Badge
```tsx
<div className={cn(
  "w-10 h-10 rounded-xl flex items-center justify-center",
  `bg-${color}-500/10 text-${color}-400`
)}>
  <Icon className="w-5 h-5" />
</div>
```

---

## 5. Animation Conventions

### Entry Animations
All content sections animate on scroll:
```tsx
// Standard fade-up
initial={{ opacity: 0, y: 20 }}
whileInView={{ opacity: 1, y: 0 }}
viewport={{ once: true, margin: "-80px" }}
transition={{ duration: 0.5, ease: "easeOut" }}
```

### Stagger Pattern
For lists and grids, use container + children variants:
```tsx
const container = {
  hidden: {},
  visible: { transition: { staggerChildren: 0.1 } }
};

const item = {
  hidden: { opacity: 0, y: 20 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.5 } }
};

<motion.div variants={container} initial="hidden" whileInView="visible" viewport={{ once: true }}>
  {items.map(i => <motion.div key={i} variants={item} />)}
</motion.div>
```

### Page Transitions
Use `AnimatePresence mode="wait"` for content swaps:
```tsx
<AnimatePresence mode="wait">
  <motion.div
    key={activeTab}
    initial={{ opacity: 0, x: 20 }}
    animate={{ opacity: 1, x: 0 }}
    exit={{ opacity: 0, x: -20 }}
    transition={{ duration: 0.3 }}
  />
</AnimatePresence>
```

### Hover Effects
Interactive elements use subtle scale transitions:
```tsx
whileHover={{ scale: 1.02, borderColor: "rgba(255,255,255,0.2)" }}
whileTap={{ scale: 0.98 }}
transition={{ type: "spring", stiffness: 300, damping: 20 }}
```

### Number Counters
Animate numeric values with `useMotionValue`:
```tsx
const count = useMotionValue(0);
const rounded = useTransform(count, v => Math.round(v));
const isInView = useInView(ref, { once: true });

useEffect(() => {
  if (isInView) {
    animate(count, targetValue, { duration: 1.5 });
  }
}, [isInView]);
```

### Rules
1. **Always** set `viewport={{ once: true }}` — animations fire once
2. **Never** animate layout-affecting properties (width, height) without `layout` prop
3. **Stagger** grid items by 80-120ms
4. **Keep** durations between 0.3s-0.6s for UI elements
5. **Use spring** for interactive elements, **easeOut** for reveals
6. **Respect** `prefers-reduced-motion` (Framer Motion handles this automatically)

---

## 6. Page Architecture

### File Organization
```
frontend/app/
├── layout.tsx                    # Root layout (nav, footer, background)
├── page.tsx                      # Home — summarization tool
├── globals.css                   # Design tokens + utilities
├── research/
│   └── page.tsx                  # Research showcase
├── pipeline/
│   └── page.tsx                  # Interactive pipeline visualizer
├── components/
│   ├── ui/                       # Reusable animated primitives
│   │   ├── AnimatedCounter.tsx
│   │   ├── GlassCard.tsx
│   │   ├── ScrollReveal.tsx
│   │   ├── StaggerContainer.tsx
│   │   ├── TimelineNode.tsx
│   │   ├── PipelineFlowDiagram.tsx
│   │   ├── MetricBar.tsx
│   │   ├── FloatingOrb.tsx
│   │   └── index.ts
│   ├── research/                 # Research page sections
│   │   ├── HeroSection.tsx
│   │   ├── DataTimeline.tsx
│   │   ├── PipelineComparison.tsx
│   │   ├── HallucinationGuard.tsx
│   │   ├── MetricsDashboard.tsx
│   │   └── TechStackGrid.tsx
│   └── pipeline/                 # Pipeline page sections
│       ├── PipelineNode.tsx
│       ├── PipelineArrow.tsx
│       └── ComparisonTable.tsx
└── lib/
    └── utils.ts                  # cn() helper
```

### Server vs Client Components
- **Page files** (`page.tsx`): Server Components (export metadata for SEO)
- **Section components**: `"use client"` (they use Framer Motion)
- **Layout**: Server Component (static shell)
- **UI primitives**: `"use client"` (interactive)

### Navigation
All routes appear in the header nav in `layout.tsx`:
```
ClinicalSafe | Summarizer | Research | Pipeline | Documentation | Safety & Ethics
```

---

## 7. Approach Colors

Each pipeline approach has a consistent color identity used everywhere:

| Approach | Tailwind Prefix | Icon Background | Text Color | Border Hover |
|----------|----------------|----------------|------------|-------------|
| ML Pipeline | `blue` | `bg-blue-500/10` | `text-blue-400` | `border-blue-500/30` |
| DL Base | `purple` | `bg-purple-500/10` | `text-purple-400` | `border-purple-500/30` |
| DL Fine-tuned | `emerald` | `bg-emerald-500/10` | `text-emerald-400` | `border-emerald-500/30` |
| Compare | `amber` | `bg-amber-500/10` | `text-amber-400` | `border-amber-500/30` |
| Success | `green` | `bg-green-500/10` | `text-green-400` | — |
| Error | `red` | `bg-red-500/10` | `text-red-400` | — |

---

## 8. API Endpoints

The frontend communicates with two backend servers:

| Endpoint | Method | Port | Pipeline | Request Body |
|----------|--------|------|----------|-------------|
| `/summarize` | POST | 8000 | ML | `{ "table_text": string }` |
| `/summarize` | POST | 8001 | DL Base | `{ "table_text": string }` |
| `/summarize` | POST | 8001 | DL Fine-tuned | `{ "table_text": string, "use_finetuned": true }` |

Response shape:
```ts
interface ResultData {
  summary: string;
  model_used: string;
  verified: boolean;
  numeric_accuracy: number;
  inference_time_ms: number;
  warnings: string[];
  tokens_generated: number;
}
```

---

## 9. Accessibility Requirements
- All interactive elements must be keyboard accessible
- Use semantic HTML (`<section>`, `<article>`, `<nav>`, proper heading hierarchy)
- Maintain WCAG AA contrast ratios (white text on dark background meets this)
- Provide `aria-label` for icon-only buttons
- Animations respect `prefers-reduced-motion`
