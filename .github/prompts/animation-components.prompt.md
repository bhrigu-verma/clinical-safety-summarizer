---
description: "Create reusable animated UI components for ClinicalSafe — glassmorphism cards, animated counters, scroll-reveal wrappers, timeline nodes, and pipeline flow diagrams using Framer Motion + Tailwind."
mode: "agent"
agent: "ui-specialist"
---

# Animated Component Library

Build a set of reusable animated components in `frontend/app/components/ui/` that follow the ClinicalSafe design system.

## Components to Create

### 1. `AnimatedCounter.tsx`
Smoothly animates a number from 0 to a target value.

```tsx
interface AnimatedCounterProps {
  value: number;
  suffix?: string;       // e.g., "%", "ms", "+"
  prefix?: string;       // e.g., "$", "~"
  duration?: number;     // seconds, default 1.5
  decimals?: number;     // decimal places, default 0
  className?: string;
}
```

- Uses `useMotionValue` + `useTransform` + `useInView` from Framer Motion
- Only starts counting when scrolled into view
- Displays with monospace font (`font-mono`)

### 2. `GlassCard.tsx`
Standard glassmorphism card with hover animation.

```tsx
interface GlassCardProps {
  children: React.ReactNode;
  className?: string;
  hoverGlow?: "blue" | "purple" | "emerald" | "amber";
  delay?: number;        // stagger delay in seconds
  onClick?: () => void;
}
```

- Base: `bg-white/5 backdrop-blur-xl border border-white/10 rounded-2xl p-6`
- Hover: scale 1.02, border brightens, optional color glow in top-right corner
- Entry animation: fade up with configurable delay

### 3. `ScrollReveal.tsx`
Wrapper that animates children when scrolled into view.

```tsx
interface ScrollRevealProps {
  children: React.ReactNode;
  direction?: "up" | "down" | "left" | "right";
  delay?: number;
  className?: string;
}
```

- Uses `whileInView` with `viewport={{ once: true, margin: "-80px" }}`
- Translates 30px from the specified direction
- Duration: 0.5s with easeOut

### 4. `StaggerContainer.tsx`
Container that staggers the animation of its direct children.

```tsx
interface StaggerContainerProps {
  children: React.ReactNode;
  staggerDelay?: number; // default 0.1
  className?: string;
}
```

- Uses Framer Motion `variants` with `staggerChildren`
- Wraps children in `motion.div` with proper variant inheritance

### 5. `TimelineNode.tsx`
A single node in an animated vertical timeline.

```tsx
interface TimelineNodeProps {
  step: number;
  title: string;
  description: string;
  detail?: React.ReactNode; // expandable content
  color?: "blue" | "purple" | "emerald" | "amber";
  isLast?: boolean;
}
```

- Vertical line connecting dots
- Dot pulses when `whileInView`
- Content card slides in from the left
- Optional expand/collapse for detail content

### 6. `PipelineFlowDiagram.tsx`
Animated horizontal flow diagram showing pipeline steps.

```tsx
interface PipelineStep {
  icon: React.ReactNode;
  label: string;
  sublabel: string;
  color: string;
}

interface PipelineFlowDiagramProps {
  steps: PipelineStep[];
  title: string;
  accentColor: "blue" | "purple";
}
```

- Steps laid out horizontally with animated connecting arrows
- Each step animates in sequence (left to right)
- Arrows animate with a drawing effect (SVG path animation)
- On mobile: stack vertically

### 7. `MetricBar.tsx`
Animated horizontal bar for showing metric values.

```tsx
interface MetricBarProps {
  label: string;
  value: number;
  maxValue: number;
  color: "blue" | "purple" | "emerald" | "amber";
  suffix?: string;
}
```

- Bar fills from left with spring animation on scroll
- Label on left, value on right
- Subtle gradient fill matching the color

### 8. `FloatingOrb.tsx`
Background decorative animated orb.

```tsx
interface FloatingOrbProps {
  color: string;       // Tailwind color class like "blue-500"
  size: "sm" | "md" | "lg";
  position: { top?: string; left?: string; right?: string; bottom?: string };
}
```

- Absolutely positioned gradient circle with blur
- Infinite slow drift animation (translate + scale)
- Low opacity (5-10%) so it doesn't distract

## Export
Create an `index.ts` barrel file in `frontend/app/components/ui/` that re-exports all components.

## Rules
- Every component must be `"use client"`
- Full TypeScript with exported interfaces
- Use `cn()` from `@/lib/utils` for className merging (create the util if it doesn't exist)
- No hardcoded colors — always use the color prop mapped to Tailwind classes
- All animations respect `prefers-reduced-motion` via Framer Motion's built-in support
