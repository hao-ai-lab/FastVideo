import { execFileSync } from "node:child_process";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { test, expect } from "@playwright/test";

const imagePath = path.resolve("public/k2.png");
const framePrompt = "A paper fox walks through a sunlit forest, gentle birdsong.";

function makeAudio(sampleRate = 8000, seconds = 1): Buffer {
	const sampleCount = sampleRate * seconds;
	const bytes = Buffer.alloc(44 + sampleCount * 2);
	bytes.write("RIFF", 0); bytes.writeUInt32LE(bytes.length - 8, 4); bytes.write("WAVEfmt ", 8);
	bytes.writeUInt32LE(16, 16); bytes.writeUInt16LE(1, 20); bytes.writeUInt16LE(1, 22);
	bytes.writeUInt32LE(sampleRate, 24); bytes.writeUInt32LE(sampleRate * 2, 28);
	bytes.writeUInt16LE(2, 32); bytes.writeUInt16LE(16, 34); bytes.write("data", 36);
	bytes.writeUInt32LE(sampleCount * 2, 40);
	for (let i = 0; i < sampleCount; i++) bytes.writeInt16LE(Math.round(Math.sin(i * 440 * 2 * Math.PI / sampleRate) * 1000), 44 + i * 2);
	return bytes;
}

test.describe("generation modes through the mock runtime", () => {
	for (const mode of ["t2va", "fl2va", "ref2va"] as const) {
		test(`${mode} sends validated assets and plays a clearly labeled sample`, async ({ page, request }, testInfo) => {
			const response = await request.get("/generation-capabilities");
			const capabilities = response.ok() ? await response.json() : {};
			test.skip(capabilities.mock !== true, "This test uses the CPU mock runtime; it must not silently allocate a real GPU.");
			const sent: Record<string, any>[] = [];
			const received: Record<string, any>[] = [];
			page.on("websocket", (socket) => {
				socket.on("framesent", ({ payload }) => { if (typeof payload === "string") { try { sent.push(JSON.parse(payload)); } catch {} } });
				socket.on("framereceived", ({ payload }) => { if (typeof payload === "string") { try { received.push(JSON.parse(payload)); } catch {} } });
			});
			await page.goto("/");
			await expect(page.getByText(/Demo runtime · Sample playback only/)).toBeVisible();
			const modeSelect = page.getByRole("combobox", { name: "Generation mode" });
			await modeSelect.selectOption(mode);
			await page.getByLabel("Continuation prompt").fill(framePrompt);
			const uploadedIds: string[] = [];
			page.on("response", async (uploadResponse) => {
				if (uploadResponse.request().method() === "POST" && uploadResponse.url().endsWith("/assets") && uploadResponse.ok()) {
					const asset = await uploadResponse.json().catch(() => null);
					if (asset?.asset_id) uploadedIds.push(asset.asset_id);
				}
			});
			try {
				if (mode === "fl2va") {
					await expect(page.getByRole("button", { name: "Generate", exact: true })).toBeDisabled();
					await page.locator('input[type="file"]').setInputFiles([
						{ name: "first-frame.png", mimeType: "image/png", buffer: await readFile(imagePath) },
						{ name: "last-frame.png", mimeType: "image/png", buffer: await readFile(imagePath) },
					]);
					await expect(page.getByRole("option", { name: "first-frame.png", exact: true }).first()).toBeAttached();
					await page.getByRole("combobox", { name: "First frame", exact: true }).selectOption({ label: "first-frame.png" });
					await expect(page.getByRole("button", { name: "Generate", exact: true })).toBeEnabled();
					await page.getByRole("combobox", { name: "Last frame", exact: true }).selectOption({ label: "last-frame.png" });
				}
				if (mode === "ref2va") {
					const video = execFileSync(process.env.FASTVIDEO_FFMPEG_BIN || "ffmpeg", ["-v", "error", "-f", "lavfi", "-i", "color=c=royalblue:s=64x64:r=8", "-t", "1", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "frag_keyframe+empty_moov", "-f", "mp4", "pipe:1"]);
					await page.locator('input[type="file"]').setInputFiles([
						{ name: "subject.png", mimeType: "image/png", buffer: await readFile(imagePath) },
						{ name: "motion.mp4", mimeType: "video/mp4", buffer: video },
						{ name: "sound.wav", mimeType: "audio/wav", buffer: makeAudio() },
					]);
					await expect(page.getByRole("button", { name: "Add sound.wav as reference" })).toBeEnabled();
					await page.getByRole("button", { name: "Add sound.wav as reference" }).click();
					await expect(page.getByRole("button", { name: "Generate", exact: true })).toBeDisabled();
					await page.getByRole("button", { name: "Add subject.png as reference" }).click();
					await page.getByRole("button", { name: "Add motion.mp4 as reference" }).click();
					await page.getByRole("button", { name: "Move sound.wav down" }).click();
					const names = await page.getByRole("list", { name: "Ordered references" }).locator("li p.font-medium").allTextContents();
					expect(names).toEqual(["subject.png", "sound.wav", "motion.mp4"]);
				}
				await page.screenshot({ path: testInfo.outputPath(`${mode}-inputs.png`), fullPage: true });
				await page.getByRole("button", { name: "Generate", exact: true }).click();
				await expect.poll(() => sent.find((item) => item.type === "session_init_v2")?.generation_mode).toBe(mode);
				const init = sent.find((item) => item.type === "session_init_v2")!;
				expect(init.conditioning_assets.map((item: any) => item.role)).toEqual(mode === "t2va" ? [] : mode === "fl2va" ? ["first_frame", "last_frame"] : ["reference", "reference", "reference"]);
				if (mode === "ref2va") expect(init.conditioning_assets.map((item: any) => item.asset_id)).toEqual([uploadedIds[0], uploadedIds[2], uploadedIds[1]]);
				await expect.poll(() => received.find((item) => item.type === "gpu_assigned")?.generation_mode).toBe(mode);
				await expect.poll(() => received.some((item) => item.type === "media_segment_complete")).toBe(true);
				await expect(page.getByText(/Demo runtime · Sample playback only/)).toBeVisible();
				await expect(modeSelect).toHaveCount(0);
				await expect.poll(async () => page.locator("video:visible").first().evaluate((element: HTMLVideoElement) => element.readyState)).toBeGreaterThanOrEqual(2);
				await page.screenshot({ path: testInfo.outputPath(`${mode}-playback.png`), fullPage: true });
				if (mode === "ref2va") {
					await page.getByRole("button", { name: "Toggle sidebar" }).click();
					await page.getByRole("button", { name: "New project", exact: true }).click();
					await expect(modeSelect).toHaveValue("t2va");
					await modeSelect.selectOption("fl2va");
					await page.getByRole("combobox", { name: "First frame", exact: true }).selectOption({ label: "subject.png" });
					await expect(page.getByRole("combobox", { name: "Last frame", exact: true })).toHaveValue("");
					await page.getByLabel("Continuation prompt").fill("The paper fox explores a new scene.");
					await page.getByRole("button", { name: "Generate", exact: true }).click();
					await expect.poll(() => sent.find((item) => item.type === "project_init_v1")?.generation_mode).toBe("fl2va");
					const secondProject = sent.find((item) => item.type === "project_init_v1")!;
					expect(secondProject.conditioning_assets).toEqual([{ asset_id: uploadedIds[0], role: "first_frame" }]);
					expect(sent.filter((item) => item.type === "session_init_v2")).toHaveLength(1);
					await expect.poll(() => received.filter((item) => item.type === "media_segment_complete").length).toBeGreaterThan(1);
				}
			} finally {
				await page.close();
				for (const id of uploadedIds) await request.delete(`/assets/${id}`);
			}
		});
	}

	test("proxies a media upload larger than Next's default 10 MiB body limit", async ({ request }) => {
		const response = await request.get("/generation-capabilities");
		const capabilities = response.ok() ? await response.json() : {};
		test.skip(capabilities.mock !== true, "Requires the local mock runtime.");
		const audio = makeAudio(192000, 29);
		expect(audio.length).toBeGreaterThan(10 * 1024 * 1024);
		const upload = await request.post("/assets", {
			headers: { "Content-Type": "audio/wav", "X-Asset-Name": "large-proxy-check.wav" },
			data: audio,
		});
		expect(upload.status()).toBe(201);
		const asset = await upload.json();
		try { expect(asset.size).toBe(audio.length); } finally { await request.delete(`/assets/${asset.asset_id}`); }
	});
});
