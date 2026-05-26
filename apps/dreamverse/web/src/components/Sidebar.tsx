"use client";

import React, { useState, useRef, useCallback, useEffect } from "react";
import { SidePanelCloseFilled } from "@carbon/icons-react";
import { Plus, Trash2, Clock, Film } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import type { StoredProject } from "@/lib/projectStorage";

function resolveProjectTitle(project: StoredProject): string {
	if (project.originalLabel) return project.originalLabel;
	const events = project.promptEvents || [];
	for (let i = events.length - 1; i >= 0; i--) {
		const e = events[i];
		if (String(e?.source || "") === "user_rewrite" && typeof e?.text === "string" && e.text.trim()) {
			return e.text.trim();
		}
	}
	return "Untitled project";
}

function formatRelativeTime(timestamp: number): string {
	const diff = Date.now() - timestamp;
	const seconds = Math.floor(diff / 1000);
	if (seconds < 60) return "just now";
	const minutes = Math.floor(seconds / 60);
	if (minutes < 60) return `${minutes}m ago`;
	const hours = Math.floor(minutes / 60);
	if (hours < 24) return `${hours}h ago`;
	const days = Math.floor(hours / 24);
	if (days < 7) return `${days}d ago`;
	return new Date(timestamp).toLocaleDateString();
}

interface SidebarProps {
	open?: boolean;
	currentProjectId?: string;
	currentProjectLabel?: string;
	sessionActive?: boolean;
	sessionExpired?: boolean;
	projectResetPending?: boolean;
	savedProjects?: StoredProject[];
	viewingProjectId?: string | null;
	isViewingPastProject?: boolean;
	onClose?: () => void;
	onSelectProject?: (projectId: string) => void;
	onSelectCurrentProject?: () => void;
	onDeleteProject?: (projectId: string) => void;
	onNewProject?: () => void;
}

export default function Sidebar({
	open = false,
	currentProjectId = "",
	currentProjectLabel = "",
	sessionActive = false,
	sessionExpired = false,
	projectResetPending = false,
	savedProjects = [],
	viewingProjectId = null,
	isViewingPastProject = false,
	onClose = () => {},
	onSelectProject = () => {},
	onSelectCurrentProject = () => {},
	onDeleteProject = () => {},
	onNewProject = () => {},
}: SidebarProps) {
	const hasCurrentProject = sessionActive || sessionExpired;
	const previousProjects = currentProjectId ? savedProjects.filter((p) => p.id !== currentProjectId) : savedProjects;

	const [pendingDeleteId, setPendingDeleteId] = useState<string | null>(null);
	const deleteTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

	const clearPendingDelete = useCallback(() => {
		setPendingDeleteId(null);
		if (deleteTimerRef.current) {
			clearTimeout(deleteTimerRef.current);
			deleteTimerRef.current = null;
		}
	}, []);

	useEffect(() => {
		return () => {
			if (deleteTimerRef.current) clearTimeout(deleteTimerRef.current);
		};
	}, []);

	useEffect(() => {
		if (!open) clearPendingDelete();
	}, [open, clearPendingDelete]);

	function handleDeleteClick(e: React.MouseEvent, projectId: string) {
		e.stopPropagation();
		if (pendingDeleteId === projectId) {
			clearPendingDelete();
			onDeleteProject(projectId);
		} else {
			setPendingDeleteId(projectId);
			if (deleteTimerRef.current) clearTimeout(deleteTimerRef.current);
			deleteTimerRef.current = setTimeout(() => setPendingDeleteId(null), 3000);
		}
	}

	return (
		<>
			<div
				className={cn("fixed inset-0 z-40 bg-black/40 backdrop-blur-[2px] transition-opacity duration-200", open ? "opacity-100" : "pointer-events-none opacity-0")}
				onClick={onClose}
				aria-hidden="true"
			/>

			<aside
				className={cn(
					"fixed inset-y-0 left-0 z-50 flex w-[280px] max-w-[calc(100vw-3rem)] flex-col border-r border-border/60 bg-card/95 backdrop-blur-xl transition-transform duration-200 ease-out",
					open ? "translate-x-0" : "-translate-x-full",
				)}
				aria-label="Project history"
			>
				<div className="flex items-center justify-between px-5 pt-4 pb-2">
					<span className="text-lg font-semibold text-foreground">Projects</span>
					<Button variant="ghost" size="icon" onClick={onClose} aria-label="Close sidebar">
						<SidePanelCloseFilled size={18} />
					</Button>
				</div>

				<div className="px-4 pb-3">
					<Button
						variant="outline"
						size="sm"
						className="w-full gap-2 rounded-lg font-medium"
						onClick={onNewProject}
						disabled={projectResetPending}
					>
						<Plus className="size-4" />
						{projectResetPending ? "Starting new project..." : "New project"}
					</Button>
				</div>

				<nav className="flex-1 overflow-y-auto px-3 pb-4">
					{hasCurrentProject && (
						<div className="mb-3">
							<p className="mb-1.5 px-2 text-[11px] font-medium uppercase tracking-wider text-muted-foreground">Current</p>
							<div
								className={cn("rounded-xl px-3 py-2.5", isViewingPastProject ? "cursor-pointer bg-accent/40 hover:bg-accent/60 transition-colors" : "bg-accent/80")}
								onClick={isViewingPastProject ? onSelectCurrentProject : undefined}
								role={isViewingPastProject ? "button" : undefined}
								tabIndex={isViewingPastProject ? 0 : undefined}
								onKeyDown={isViewingPastProject ? (e) => e.key === "Enter" && onSelectCurrentProject() : undefined}
							>
								<div className="flex items-center gap-2">
									<Film className="size-3.5 shrink-0 text-muted-foreground" />
									<span className="min-w-0 flex-1 truncate text-[13px] font-medium text-foreground">{currentProjectLabel || "Untitled project"}</span>
								</div>
								<div className="mt-1.5 flex items-center gap-1.5">
									{sessionExpired ? (
										<Badge variant="secondary" className="rounded-md px-1.5 py-0 text-[10px]">
											Expired
										</Badge>
									) : (
										<Badge variant="default" className="rounded-md px-1.5 py-0 text-[10px]">
											Active
										</Badge>
									)}
								</div>
							</div>
						</div>
					)}

					{previousProjects.length > 0 && (
						<div>
							<p className="mb-1.5 px-2 text-[11px] font-medium uppercase tracking-wider text-muted-foreground">Previous</p>
							<div className="flex flex-col gap-1">
								{previousProjects.map((project) => {
									const isViewing = viewingProjectId === project.id;
									return (
										<div
											key={project.id}
											className={cn("group flex items-start gap-2 rounded-xl px-3 py-2.5 transition-colors cursor-pointer", isViewing ? "bg-accent/60" : "hover:bg-accent/40")}
											onClick={() => onSelectProject(project.id)}
											role="button"
											tabIndex={0}
											onKeyDown={(e) => e.key === "Enter" && onSelectProject(project.id)}
										>
											{project.lastThumbnail ? (
												<img src={project.lastThumbnail} alt="" className="mt-0.5 h-8 w-auto shrink-0 rounded border border-border object-cover" />
											) : (
												<Film className="mt-0.5 size-3.5 shrink-0 text-muted-foreground" />
											)}
											<div className="min-w-0 flex-1">
												<p className="truncate text-[13px] font-medium text-foreground">{resolveProjectTitle(project)}</p>
												<div className="flex items-center gap-1 text-[11px] text-muted-foreground">
													<Clock className="size-3" />
													<span>{formatRelativeTime(project.createdAt)}</span>
												</div>
											</div>
											{pendingDeleteId === project.id ? (
												<button
													type="button"
													className="mt-0.5 shrink-0 rounded bg-destructive/15 px-1.5 py-0.5 !text-xs !font-medium text-destructive transition-colors hover:bg-destructive/25"
													onClick={(e) => handleDeleteClick(e, project.id)}
													aria-label="Confirm delete project"
												>
													Delete?
												</button>
											) : (
												<button
													type="button"
													className="mt-0.5 shrink-0 rounded p-1 text-muted-foreground opacity-0 transition-opacity hover:bg-destructive/10 hover:text-destructive group-hover:opacity-100"
													onClick={(e) => handleDeleteClick(e, project.id)}
													aria-label="Delete project"
												>
													<Trash2 className="size-3.5" />
												</button>
											)}
										</div>
									);
								})}
							</div>
						</div>
					)}

					{!hasCurrentProject && previousProjects.length === 0 && <p className="px-2 py-4 text-center text-[13px] text-muted-foreground">No projects yet</p>}
				</nav>
			</aside>
		</>
	);
}
