import type { Metadata } from 'next';
import { IBM_Plex_Mono, IBM_Plex_Sans } from 'next/font/google';

import { AppShell } from '@/components/shell/AppShell';
import './globals.css';

const plexSans = IBM_Plex_Sans({
  subsets: ['latin'],
  weight: ['400', '500', '600', '700'],
  variable: '--font-plex-sans',
  display: 'swap',
});

const plexMono = IBM_Plex_Mono({
  subsets: ['latin'],
  weight: ['400', '500', '600'],
  variable: '--font-plex-mono',
  display: 'swap',
});

export const metadata: Metadata = {
  title: 'FastVideo Studio',
  icons: { icon: '/fastvideo.ico' },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html
      lang="en"
      suppressHydrationWarning
      className={`${plexSans.variable} ${plexMono.variable}`}
    >
      <head>
        {/* The studio defaults to dark; apply the stored choice before paint. */}
        <script
          id="theme-init-script"
          suppressHydrationWarning
          dangerouslySetInnerHTML={{
            __html: `(function(){var dark=true;try{dark=localStorage.getItem('theme')!=='light'}catch(e){}if(dark)document.documentElement.classList.add('dark')})()`,
          }}
        />
      </head>
      <body className="antialiased">
        <AppShell>{children}</AppShell>
      </body>
    </html>
  );
}
