import type { Metadata, Viewport } from "next";
import { Mona_Sans } from "next/font/google";
import "./globals.css";

const monaSans = Mona_Sans({
  variable: "--font-mona-sans",
  subsets: ["latin", "latin-ext"],
  axes: ["wdth"],
});

export const metadata: Metadata = {
  title: "Tuteur NLP",
  description:
    "Assistant de révision pour les cours NLP et Transformers d'ING3 IA (CY Tech) : Qwen2.5-0.5B-Instruct adapté par LoRA.",
};

export const viewport: Viewport = {
  themeColor: "#000000",
  colorScheme: "dark",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="fr" className={`${monaSans.variable} antialiased`}>
      <body className="min-h-dvh">{children}</body>
    </html>
  );
}
