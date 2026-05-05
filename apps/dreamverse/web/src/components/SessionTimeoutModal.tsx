"use client";

import { Button } from "@/components/ui/button";

interface SessionTimeoutModalProps {
	open?: boolean;
	onClose?: () => void;
	onStartNewProject?: () => void;
	repoUrl?: string;
	blogUrl?: string;
}

export default function SessionTimeoutModal({
	open = false,
	onClose = () => {},
	onStartNewProject = () => {},
	repoUrl = "",
	blogUrl = "",
}: SessionTimeoutModalProps) {
	if (!open) return null;

	return (
		<div className="fixed inset-0 z-[70] flex items-center justify-center bg-black/45 px-4 backdrop-blur-[3px]">
			<div
				role="dialog"
				aria-modal="true"
				aria-labelledby="session-timeout-title"
				className="w-full max-w-md rounded-3xl border border-border/70 bg-card/95 p-6 shadow-2xl"
			>
				<div className="space-y-3">
					<div className="space-y-1">
						<h2 id="session-timeout-title" className="text-lg font-semibold text-foreground">
							Session ended
						</h2>
						<p className="text-sm text-muted-foreground">
							This project hit the current 5-minute session limit. Your latest video stays on screen, and the project is being kept in the archive so you can come back to it.
						</p>
					</div>
					<p className="text-sm text-muted-foreground">
						Start a new project to keep creating, or keep viewing this one while you decide what to do next.
					</p>
					<div className="flex flex-wrap gap-2 text-sm">
						<a
							href={repoUrl}
							target="_blank"
							rel="noreferrer"
							className="font-medium text-sky-700 underline underline-offset-4 transition-colors hover:text-sky-600 dark:text-sky-300 dark:hover:text-sky-200"
						>
							Open repo
						</a>
						<a
							href={blogUrl}
							target="_blank"
							rel="noreferrer"
							className="font-medium text-sky-700 underline underline-offset-4 transition-colors hover:text-sky-600 dark:text-sky-300 dark:hover:text-sky-200"
						>
							Read blog
						</a>
					</div>
				</div>
				<div className="mt-5 flex flex-col-reverse gap-2 sm:flex-row sm:justify-end">
					<Button variant="outline" onClick={onClose}>
						Keep viewing
					</Button>
					<Button onClick={onStartNewProject}>New project</Button>
				</div>
			</div>
		</div>
	);
}
