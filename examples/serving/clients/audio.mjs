// Run in Node.js 22+ or a trusted backend, not in a browser with a private key.
import { writeFile } from "node:fs/promises";

const baseUrl = process.env.FASTVIDEO_BASE_URL || "http://127.0.0.1:8000/v1";
const headers = {
  Authorization: `Bearer ${process.env.FASTVIDEO_API_KEY || "local"}`,
  "Content-Type": "application/json",
};
const body = {
  model: process.env.FASTVIDEO_MODEL || "stable-audio",
  prompt: process.env.FASTVIDEO_PROMPT || "Rain on a metal roof, distant thunder.",
  seconds: Number(process.env.FASTVIDEO_SECONDS || 6),
};
if (process.env.FASTVIDEO_VIDEO_REFERENCE) {
  body.video_reference = { video_url: process.env.FASTVIDEO_VIDEO_REFERENCE };
}

let response = await fetch(`${baseUrl}/videos`, {
  method: "POST",
  headers,
  body: JSON.stringify(body),
});
if (!response.ok) throw new Error(`Submit failed (${response.status}): ${await response.text()}`);
let job = await response.json();
console.log(`Submitted ${job.id}`);

const deadline = performance.now() + 1_800_000;
while (job.status === "queued" || job.status === "in_progress") {
  const remaining = deadline - performance.now();
  if (remaining <= 0) throw new Error(`Polling timed out; job ${job.id} may still be running`);
  await new Promise((resolve) => setTimeout(resolve, Math.min(2000, remaining)));
  response = await fetch(`${baseUrl}/videos/${job.id}`, { headers });
  if (!response.ok) throw new Error(`Poll failed (${response.status}): ${await response.text()}`);
  job = await response.json();
}

if (job.status !== "completed") {
  throw new Error(`Audio job ${job.id} failed: ${job.error?.message || job.status}`);
}
if (job.media_type !== "audio/wav") {
  throw new Error(`Expected audio/wav, got ${job.media_type}`);
}

response = await fetch(`${baseUrl}/videos/${job.id}/content`, { headers });
if (!response.ok) throw new Error(`Download failed (${response.status}): ${await response.text()}`);
const output = `${job.id}.wav`;
await writeFile(output, Buffer.from(await response.arrayBuffer()));
console.log(`Saved ${output}`);
