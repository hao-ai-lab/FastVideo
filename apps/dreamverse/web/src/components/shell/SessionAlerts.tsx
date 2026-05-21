"use client";

import React from "react";

import { Alert, AlertDescription } from "@/components/ui/alert";

interface Props {
	sessionNotice?: string;
}

export default function SessionAlerts({ sessionNotice = "" }: Props) {
	if (!sessionNotice) return null;

	return (
		<div className="flex h-full self-center">
			<Alert className="border-blue-700 bg-accent-blue/15 text-blue-700 dark:text-blue-50 mt-14 mb-4 w-fit h-fit">
				<AlertDescription className="font-medium">{sessionNotice}</AlertDescription>
			</Alert>
		</div>
	);
}
