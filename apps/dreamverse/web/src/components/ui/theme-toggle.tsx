"use client";

import * as React from "react";
import { Sun, Moon } from "@carbon/icons-react";

import { Button } from "@/components/ui/button";

function ThemeToggle({ className }: { className?: string }) {
	const [dark, setDark] = React.useState(false);

	React.useEffect(() => {
		const root = document.documentElement;
		let nextDark = root.classList.contains("dark");
		try {
			const storedTheme = localStorage.getItem("theme");
			if (storedTheme === "dark") {
				nextDark = true;
			} else if (storedTheme === "light") {
				nextDark = false;
			}
		} catch {
			// storage unavailable
		}
		root.classList.toggle("dark", nextDark);
		setDark(nextDark);
	}, []);

	const transitionTimer = React.useRef<ReturnType<typeof setTimeout>>(undefined);

	function toggle() {
		const next = !dark;
		setDark(next);

		clearTimeout(transitionTimer.current);
		document.documentElement.classList.add("theme-transition");
		document.documentElement.classList.toggle("dark", next);

		transitionTimer.current = setTimeout(() => {
			document.documentElement.classList.remove("theme-transition");
		}, 350);

		try {
			localStorage.setItem("theme", next ? "dark" : "light");
		} catch {
			// storage unavailable
		}
	}

	React.useEffect(() => {
		return () => clearTimeout(transitionTimer.current);
	}, []);

	return (
		<Button variant="outline" size="icon" onClick={toggle} aria-label={dark ? "Switch to light mode" : "Switch to dark mode"} className={className}>
			{dark ? <Sun size={18} /> : <Moon size={18} />}
		</Button>
	);
}

export { ThemeToggle };
