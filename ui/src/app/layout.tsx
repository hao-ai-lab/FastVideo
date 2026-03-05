import type { Metadata } from "next";
import "./globals.css";
import { DefaultOptionsProvider } from "@/contexts/DefaultOptionsContext";
import { JobsRefreshProvider } from "@/contexts/JobsRefreshContext";
import { ActiveTabProvider } from "@/contexts/ActiveTabContext";
import { ActiveJobProvider } from "@/contexts/ActiveJobContext";
import { ActiveDatasetProvider } from "@/contexts/ActiveDatasetContext";
import Header from "@/components/Header";

export const metadata: Metadata = {
  title: "FastVideo",
  description: "A lightweight UI for running video-generation jobs.",
  icons: {
    icon: "/fastvideo.ico",
  },
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
                <ActiveDatasetProvider>
                <Header />
                <main
                  style={{
                    flex: 1,
                    marginTop: "var(--header-height)",
                    display: "flex",
                    flexDirection: "column",
                    minHeight: 0,
                  }}
                >
                  {children}
                </main>
                </ActiveDatasetProvider>
              </ActiveJobProvider>
            </ActiveTabProvider>
          </JobsRefreshProvider>
        </DefaultOptionsProvider>
      </body>
    </html>
  );
}
