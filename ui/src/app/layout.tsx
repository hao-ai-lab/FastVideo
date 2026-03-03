import type { Metadata } from "next";
import "./globals.css";
import { DefaultOptionsProvider } from "@/contexts/DefaultOptionsContext";
import { JobsRefreshProvider } from "@/contexts/JobsRefreshContext";
import { ActiveTabProvider } from "@/contexts/ActiveTabContext";
import { ActiveJobProvider } from "@/contexts/ActiveJobContext";
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
            <ActiveTabProvider>
              <ActiveJobProvider>
                <Header />
                {children}
              </ActiveJobProvider>
            </ActiveTabProvider>
          </JobsRefreshProvider>
        </DefaultOptionsProvider>
      </body>
    </html>
  );
}
