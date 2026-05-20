'use client';

import { Toaster as Sonner, type ToasterProps } from 'sonner';

export function Toaster(props: ToasterProps) {
  return (
    <Sonner
      position="bottom-center"
      richColors={true}
      toastOptions={{
        classNames: {
          toast:
            'border border-border bg-card text-card-foreground shadow-lg',
          description: 'text-muted-foreground',
          actionButton:
            'bg-primary text-primary-foreground hover:bg-primary/90',
          cancelButton:
            'bg-secondary text-secondary-foreground hover:bg-secondary/80',
        },
      }}
      {...props}
    />
  );
}
