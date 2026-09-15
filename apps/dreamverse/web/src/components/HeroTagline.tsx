"use client";

import { Fragment, useEffect, useRef } from "react";

const HERO_WAVE_LIGHT = ["#2A4A98", "#4878E5", "#6FA0F2", "#B0BCC8", "#E8D99E", "#D8C844", "#C2A620"];
const HERO_WAVE_DARK = ["#143468", "#1E58B8", "#3892F0", "#80B8E8", "#B8D0EA", "#E2D498", "#DABB50"];
const HERO_TEXT = "Direct scenes in seconds";

export default function HeroTagline() {
	const ref = useRef<HTMLHeadingElement>(null);

	useEffect(() => {
		const el = ref.current;
		if (!el) return;

		let rafId = 0;

		function play() {
			const chars = el!.querySelectorAll<HTMLSpanElement>("[data-char]");
			if (!chars.length) return;
			cancelAnimationFrame(rafId);

			const isDark = document.documentElement.classList.contains("dark");
			const colors = isDark ? HERO_WAVE_DARK : HERO_WAVE_LIGHT;
			const waveLen = 10;
			const total = chars.length + waveLen;
			const duration = 1200;
			const maxBlur = 3.5;
			const start = performance.now();

			function tick() {
				const t = Math.min((performance.now() - start) / duration, 1);
				const pos = t * total;
				chars.forEach((ch, i) => {
					const rel = pos - i;
					if (rel >= 0 && rel < waveLen) {
						const norm = rel / waveLen;
						const ci = Math.floor(norm * colors.length);
						ch.style.color = colors[Math.min(colors.length - 1, ci)];

						let blur = 0;
						if (norm < 0.25) {
							blur = maxBlur * (1 - norm / 0.25);
						} else if (norm > 0.75) {
							blur = maxBlur * ((norm - 0.75) / 0.25);
						}
						ch.style.filter = blur > 0.1 ? `blur(${blur.toFixed(1)}px)` : "";
					} else {
						ch.style.color = "";
						ch.style.filter = "";
					}
				});
				if (t < 1) {
					rafId = requestAnimationFrame(tick);
				} else {
					chars.forEach((ch) => {
						ch.style.color = "";
						ch.style.filter = "";
					});
				}
			}

			rafId = requestAnimationFrame(tick);
		}

		const initialDelay = setTimeout(play, 400);
		const interval = setInterval(play, 5000);
		return () => {
			clearTimeout(initialDelay);
			clearInterval(interval);
			cancelAnimationFrame(rafId);
		};
	}, []);

	return (
		<h1 ref={ref} className="text-balance text-center text-3xl font-medium text-[#343537] dark:text-[#FAFAFB] sm:text-4xl">
			{HERO_TEXT.split(" ").map((word, wi) => (
				<Fragment key={wi}>
					{wi > 0 && (
						<span data-char className="transition-[color,filter] duration-150">
							{" "}
						</span>
					)}
					<span className="inline-flex">
						{word.split("").map((char, ci) => (
							<span key={ci} data-char className="inline-block transition-[color,filter] duration-150">
								{char}
							</span>
						))}
					</span>
				</Fragment>
			))}
		</h1>
	);
}
