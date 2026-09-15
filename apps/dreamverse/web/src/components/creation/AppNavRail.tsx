"use client";

import React from "react";
import { FolderOpen, Home, Sparkles } from "lucide-react";

import { cn } from "@/lib/utils";

export type AppNavSection = "explore" | "create" | "assets";

interface AppNavRailProps {
	activeSection?: AppNavSection;
	onSectionChange?: (section: AppNavSection) => void;
	onOpenProjects?: () => void;
	className?: string;
}

const NAV_ITEMS: Array<{ id: AppNavSection; label: string; icon: typeof Home }> = [
	{ id: "explore", label: "Explore", icon: Home },
	{ id: "create", label: "Create", icon: Sparkles },
	{ id: "assets", label: "Assets", icon: FolderOpen },
];

export default function AppNavRail({
	activeSection = "create",
	onSectionChange = () => {},
	onOpenProjects,
	className,
}: AppNavRailProps) {
	return (
		<aside
			className={cn(
				"hidden shrink-0 flex-col items-center gap-2 border-r border-border/40 bg-background/30 px-2.5 py-5 lg:flex",
				className,
			)}
			aria-label="Primary navigation"
		>
			{NAV_ITEMS.map((item) => {
				const Icon = item.icon;
				const isActive = item.id === activeSection;
				return (
					<button
						key={item.id}
						type="button"
						aria-label={item.label}
						aria-current={isActive ? "page" : undefined}
						onClick={() => {
							if (item.id === "assets") {
								onOpenProjects?.();
							}
							onSectionChange(item.id);
						}}
						className={cn(
							"studio-control studio-control-press flex w-[4.5rem] min-h-11 flex-col items-center gap-1 rounded-xl px-2 py-2.5 text-[10px] font-medium tracking-wide",
							isActive
								? "bg-secondary/90 text-foreground shadow-sm ring-1 ring-border/60"
								: "text-muted-foreground hover-capable:hover:bg-secondary/50 hover-capable:hover:text-foreground",
						)}
					>
						<Icon className={cn("size-[18px]", isActive && "text-accent-blue")} />
						{item.label}
					</button>
				);
			})}
		</aside>
	);
}
