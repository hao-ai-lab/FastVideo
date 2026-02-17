import type { Metadata } from "next";
import Image from "next/image";
import "./globals.css";
import headerStyles from "@styles/Header.module.css";

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
        <header className={headerStyles.header}>
          <Image src="/logo.svg" alt="FastVideo Logo" width={252} height={105} />
        </header>
        {children}
      </body>
    </html>
  );
}
