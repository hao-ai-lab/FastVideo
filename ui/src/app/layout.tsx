import type { Metadata } from "next";
import "./globals.css";
import { DefaultOptionsProvider } from "@/contexts/DefaultOptionsContext";
import { JobsRefreshProvider } from "@/contexts/JobsRefreshContext";
import Header from "@/components/Header";

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
        <DefaultOptionsProvider>
          <JobsRefreshProvider>
            <Header />
            {children}
          </JobsRefreshProvider>
        </DefaultOptionsProvider>
      </body>
    </html>
  );
}
