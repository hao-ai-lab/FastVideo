import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "FastVideo",
  description: "A lightweight UI for running video-generation jobs.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="antialiased">
        <header>
          <Link href="/">
            <h1>FastVideo Job Runner</h1>
          </Link>
          <p className="subtitle">Create and manage video generation jobs</p>
        </header>
        {children}
      </body>
    </html>
  );
}
