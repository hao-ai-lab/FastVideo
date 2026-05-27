"use client";

import * as React from "react";

import { cn } from "@/lib/utils";

const Input = React.forwardRef<HTMLInputElement, React.ComponentProps<"input">>(({ className, type, ...props }, ref) => {
	return (
		<input
			type={type}
			className={cn(
				"flex h-10 w-full rounded-xl border border-input bg-card/60 backdrop-blur-md px-3 py-2 text-sm text-foreground shadow-sm transition-colors placeholder:text-muted-foreground focus-visible:border-sky-400/70 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400/25 disabled:cursor-not-allowed disabled:opacity-50",
				className,
			)}
			ref={ref}
			{...props}
		/>
	);
});
Input.displayName = "Input";

export { Input };
