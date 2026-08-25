import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Phytosense AI 🌱 | HXAI Tomato Disease Diagnostic & Decision Support System',
  description: 'Publication-grade Human-Centric Explainable AI (HXAI) for Tomato Pathology: Grad-CAM visual heatmaps, counterfactual reasoning, and contextual agronomic treatment protocols.',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link
          href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap"
          rel="stylesheet"
        />
      </head>
      <body className="font-sans antialiased bg-[#070c09] text-slate-100 min-h-screen selection:bg-emerald-500/30 selection:text-emerald-200">
        {children}
      </body>
    </html>
  );
}
