import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'FastVideo Config Generator',
  description: 'Choose a maintained FastVideo recipe and generate an executable configuration',
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en">
      <body className="font-sans antialiased">
        {children}
      </body>
    </html>
  )
}
