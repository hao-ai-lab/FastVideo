"use client";

import * as React from "react";
import { Slot } from "@radix-ui/react-slot";
import { cva, type VariantProps } from "class-variance-authority";

import { cn } from "@/lib/utils";

const buttonVariants = cva(
	"inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-xl border !text-sm !font-semibold transition-colors duration-150 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400/40 disabled:pointer-events-none disabled:cursor-not-allowed disabled:opacity-50",
	{
		variants: {
			variant: {
				default: "border-slate-700 bg-slate-800 text-white shadow-md hover:bg-slate-700 dark:border-slate-300 dark:bg-slate-300 dark:text-slate-900 dark:hover:bg-slate-200",
				secondary: "border-input bg-card text-card-foreground hover:bg-accent",
				outline: "border-input bg-secondary text-secondary-foreground hover:bg-accent",
				ghost: "border-transparent bg-transparent text-foreground hover:bg-accent",
				destructive: "border-rose-500/60 bg-rose-600/90 text-white hover:bg-rose-500",
			},
			size: {
				default: "h-10 px-4 py-2",
				sm: "h-9 rounded-lg px-3 !text-xs",
				lg: "h-11 px-5 !text-sm",
				icon: "size-10",
				"icon-sm": "size-8",
			},
		},
		defaultVariants: {
			variant: "default",
			size: "default",
		},
	},
);

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement>, VariantProps<typeof buttonVariants> {
	asChild?: boolean;
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(({ className, variant, size, asChild = false, ...props }, ref) => {
	const Comp = asChild ? Slot : "button";
	return <Comp className={cn(buttonVariants({ variant, size, className }))} ref={ref} {...props} />;
});
Button.displayName = "Button";

export { Button, buttonVariants };
