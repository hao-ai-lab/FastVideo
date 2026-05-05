"use client";

import React from "react";
import { SidePanelOpenFilled } from "@carbon/icons-react";
import { ExternalLink } from "lucide-react";
import Image from "next/image";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ThemeToggle } from "@/components/ui/theme-toggle";

const FASTVIDEO_REPO_URL = "https://haoailab.com/blogs/dreamverse/";
const FASTVIDEO_BLOG_URL = "https://haoailab.com/blogs/dreamverse/";

interface Props {
	timeLeft?: number | null;
	formatTime?: (seconds: number) => string;
	onToggleSidebar?: () => void;
}

export default function Header({ timeLeft = null, formatTime = (seconds) => `${seconds}`, onToggleSidebar }: Props) {
	const timeVariant = timeLeft !== null && timeLeft <= 30 ? "warning" : "secondary";

	return (
		<header className="relative z-30 shrink-0">
			<div className="flex flex-wrap items-center justify-between gap-y-2 px-4 pt-3 pb-2 sm:pt-4 sm:pb-3 sm:px-6">
				<div className="flex items-center gap-3">
					{onToggleSidebar && (
						<Button variant="outline" size="icon" onClick={onToggleSidebar} aria-label="Toggle sidebar">
							<SidePanelOpenFilled size={20} />
						</Button>
					)}
					<a href={FASTVIDEO_REPO_URL} target="_blank" rel="noopener noreferrer" title="FastVideo on GitHub">
						<Image src="/logo.svg" alt="FastVideo" width={32} height={32} className="h-8 w-auto sm:h-9 transition-opacity hover:opacity-70" />
					</a>
					<div className="hidden sm:flex items-center gap-3">
						<a href="https://docs.google.com/forms/d/e/1FAIpQLSe5zpO1iD8Ds-Ih-fOLm64qd7YZVvuvAyHuJaAfw1hkRHTe_A/viewform?usp=publish-editor" target="_blank" rel="noopener noreferrer">
							<Button variant="outline" size="sm" className="gap-1.5 rounded-full px-3 text-xs">
								Join Waitlist
								<ExternalLink className="size-3 opacity-60" />
							</Button>
						</a>
					</div>
				</div>

				<div className="flex items-center gap-3">
					{timeLeft !== null && (
						<Badge variant={timeVariant} className="rounded-xl px-3 py-1 text-xs font-medium normal-case tracking-normal">
							Time left: {formatTime(timeLeft)}
						</Badge>
					)}
					<ThemeToggle />
				</div>
			</div>

			<div className="flex sm:hidden items-center gap-2 px-4 pb-3">
				<a href="https://docs.google.com/forms/d/e/1FAIpQLSe5zpO1iD8Ds-Ih-fOLm64qd7YZVvuvAyHuJaAfw1hkRHTe_A/viewform?usp=publish-editor" target="_blank" rel="noopener noreferrer">
					<Button variant="outline" size="sm" className="gap-1.5 rounded-full px-3 text-xs">
						Join Waitlist
						<ExternalLink className="size-3 opacity-60" />
					</Button>
				</a>
			</div>
		</header>
	);
}
