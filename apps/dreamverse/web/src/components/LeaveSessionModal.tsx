"use client";

import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";

const STORAGE_KEY = "fastvideo-suppress-leave-warning";

interface LeaveSessionModalProps {
	open?: boolean;
	onClose?: () => void;
	onConfirmLeave?: () => void;
}

export default function LeaveSessionModal({
	open = false,
	onClose = () => {},
	onConfirmLeave = () => {},
}: LeaveSessionModalProps) {
	const [suppress, setSuppress] = useState(false);

	useEffect(() => {
		if (open) setSuppress(false);
	}, [open]);

	if (!open) return null;

	function handleConfirm() {
		if (suppress) {
			try {
				localStorage.setItem(STORAGE_KEY, "1");
			} catch {}
		}
		onConfirmLeave();
	}

	return (
		<div className="fixed inset-0 z-[70] flex items-center justify-center bg-black/45 px-4 backdrop-blur-[3px]">
			<div
				role="dialog"
				aria-modal="true"
				aria-labelledby="leave-session-title"
				className="w-full max-w-md rounded-3xl border border-border/70 bg-card/95 p-6 shadow-2xl"
			>
				<div className="space-y-3">
					<div className="space-y-1">
						<h2 id="leave-session-title" className="text-lg font-semibold text-foreground">
							Leave session?
						</h2>
						<p className="text-sm text-muted-foreground">
							This will end your current session. You will need to queue again for a GPU to start a new project.
						</p>
					</div>
					<div className="flex items-center gap-2">
						<Checkbox
							id="suppress-leave-warning"
							checked={suppress}
							onCheckedChange={(v) => setSuppress(v === true)}
						/>
						<Label htmlFor="suppress-leave-warning" className="text-sm text-muted-foreground cursor-pointer">
							Do not warn again
						</Label>
					</div>
				</div>
				<div className="mt-5 flex flex-col-reverse gap-2 sm:flex-row sm:justify-end">
					<Button variant="outline" onClick={onClose}>
						Cancel
					</Button>
					<Button variant="destructive" onClick={handleConfirm}>
						Leave session
					</Button>
				</div>
			</div>
		</div>
	);
}

export function shouldShowLeaveWarning(): boolean {
	try {
		return localStorage.getItem(STORAGE_KEY) !== "1";
	} catch {
		return true;
	}
}
