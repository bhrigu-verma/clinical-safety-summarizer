import type { Metadata } from "next";
import { Outfit } from "next/font/google";
import "./globals.css";

const outfit = Outfit({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "ClinicalSafe | 2026 AI Table Summarization",
  description: "State-of-the-art clinical trial safety table summarization using ML & DL approaches.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className={outfit.className}>
        <main className="min-h-screen relative overflow-hidden">
          {/* Subtle background glow */}
          <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-blue-500/10 rounded-full blur-[120px] pointer-events-none" />
          <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-purple-500/10 rounded-full blur-[120px] pointer-events-none" />

          <div className="relative z-10 mx-auto max-w-7xl px-4 py-8 md:py-16">
            <header className="mb-12 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <div className="w-8 h-8 rounded-lg bg-gradient-to-tr from-blue-600 to-purple-600 flex items-center justify-center font-bold text-white">C</div>
                <h1 className="text-xl font-semibold tracking-tight">ClinicalSafe</h1>
              </div>
              <nav className="flex items-center gap-6 text-sm font-medium text-muted-foreground">
                <a href="#" className="hover:text-foreground transition-colors">Documentation</a>
                <a href="#" className="hover:text-foreground transition-colors">Safety Ethics</a>
                <div className="h-4 w-px bg-border" />
                <button className="px-4 py-1.5 rounded-full bg-white text-black hover:bg-white/90 transition-all font-medium">Log out</button>
              </nav>
            </header>
            {children}
          </div>
        </main>
      </body>
    </html>
  );
}
