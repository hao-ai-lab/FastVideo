"use client";

import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";

import { cn } from "@/lib/utils";

const badgeVariants = cva("inline-flex items-center rounded-full border px-2.5 py-1 text-[11px] font-semibold uppercase tracking-[0.16em] transition-colors", {
	variants: {
		variant: {
			default: "border-blue-400/35 bg-blue-500/15 text-blue-900 dark:text-blue-100",
			secondary: "border-border bg-secondary text-secondary-foreground",
			outline: "border-border bg-transparent text-muted-foreground",
			success: "border-emerald-400/30 bg-emerald-500/15 text-emerald-100",
			warning: "border-amber-400/30 bg-amber-500/15 text-amber-800 dark:text-amber-100",
			destructive: "border-rose-400/30 bg-rose-500/15 text-rose-100",
		},
	},
	defaultVariants: {
		variant: "default",
	},
});

export interface BadgeProps extends React.HTMLAttributes<HTMLDivElement>, VariantProps<typeof badgeVariants> {}

function Badge({ className, variant, ...props }: BadgeProps) {
	return <div className={cn(badgeVariants({ variant }), className)} {...props} />;
}

export { Badge, badgeVariants };
