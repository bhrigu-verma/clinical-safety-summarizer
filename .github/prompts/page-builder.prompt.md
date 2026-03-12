---
description: "Scaffold a new page in the ClinicalSafe frontend following the established design system. Provide the page name and purpose, and this prompt generates the complete page structure with proper animations, layout, and navigation integration."
mode: "agent"
agent: "ui-specialist"
---

# New Page Builder

Create a new page in the ClinicalSafe frontend application.

## Inputs Required
When invoking this prompt, specify:
1. **Page route** — e.g., `/about`, `/benchmarks`, `/data`
2. **Page purpose** — what content this page displays
3. **Key sections** — list the main content sections

## Scaffolding Checklist

For every new page, do the following:

### Step 1: Read Current State
- Read `frontend/app/layout.tsx` to understand existing navigation
- Read `frontend/app/globals.css` to use existing utility classes
- Check `frontend/app/components/` for reusable components

### Step 2: Create the Page File
Create `frontend/app/{route}/page.tsx` with:

```tsx
import { Metadata } from "next";
// Import section components

export const metadata: Metadata = {
  title: "{Page Title} | ClinicalSafe",
  description: "{Page description for SEO}",
};

export default function PageName() {
  return (
    <div className="space-y-24 py-8">
      {/* Import and render section components here */}
    </div>
  );
}
```

### Step 3: Create Section Components
For each section, create a separate `"use client"` component in `frontend/app/components/{route}/`:
- Full TypeScript interfaces for any data structures
- Framer Motion `whileInView` animations
- Staggered card reveals
- Responsive layout (mobile-first)
- Glassmorphism card styling

### Step 4: Update Navigation
Add the new route to `frontend/app/layout.tsx` in the navigation header alongside existing links (Documentation, Safety & Ethics).

### Step 5: Verify
- Ensure all imports resolve
- No TypeScript errors
- Responsive at 375px, 768px, 1024px, 1440px breakpoints
- Animations are smooth and don't cause layout shifts

## Design System Quick Reference
- Background: `bg-[#0a0a0f]`
- Cards: `minimal-card` or `bg-white/5 backdrop-blur-xl border border-white/10 rounded-2xl p-6`
- Text: `text-white` (primary), `text-white/60` (secondary), `text-muted-foreground` (tertiary)
- Headings: `gradient-text` utility or `bg-clip-text text-transparent bg-gradient-to-r from-white to-white/60`
- Accent colors: blue (ML), purple (DL), emerald (fine-tuned), amber (compare), green (success), red (danger)
- Spacing between sections: `space-y-24` or `py-24`
- Max content width: `max-w-7xl mx-auto`
