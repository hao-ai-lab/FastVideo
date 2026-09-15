"use client";

import React from "react";

import { cn } from "@/lib/utils";

export default function ConfigPill({
	children,
	className,
	...props
}: React.ButtonHTMLAttributes<HTMLButtonElement>) {
	return (
		<button
			type="button"
			className={cn(
				"studio-control studio-control-press studio-hover-surface inline-flex h-9 min-h-9 shrink-0 items-center gap-1 rounded-full border border-border/50 bg-background/80 px-2.5 text-[11px] font-medium text-foreground/90",
				className,
			)}
			{...props}
		>
			{children}
		</button>
	);
}
