'use client';

import * as React from 'react';
import * as SwitchPrimitives from '@radix-ui/react-switch';

import { cn } from '@/lib/utils';

const Switch = React.forwardRef<
  React.ElementRef<typeof SwitchPrimitives.Root>,
  React.ComponentPropsWithoutRef<typeof SwitchPrimitives.Root>
>(({ className, ...props }, ref) => (
  <SwitchPrimitives.Root
    ref={ref}
    className={cn(
      'peer relative inline-flex size-11 shrink-0 cursor-pointer items-center justify-center rounded-xl bg-transparent transition-colors before:absolute before:h-5 before:w-9 before:rounded-full before:border before:border-border before:bg-background data-[state=checked]:before:border-accent-blue data-[state=checked]:before:bg-accent-blue disabled:cursor-not-allowed disabled:opacity-50',
      className,
    )}
    {...props}
  >
    <SwitchPrimitives.Thumb
      className={cn(
        'pointer-events-none absolute left-1.5 top-[15px] z-[1] block h-3.5 w-3.5 rounded-full bg-muted-foreground shadow-lg ring-0 transition-transform data-[state=checked]:translate-x-4 data-[state=checked]:bg-white',
      )}
    />
  </SwitchPrimitives.Root>
));
Switch.displayName = SwitchPrimitives.Root.displayName;

export { Switch };
