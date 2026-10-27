// Run in Node.js 22+ or your webapp's backend, not in a browser with a private key.
import OpenAI from "openai";
import { writeFile } from "node:fs/promises";

const client = new OpenAI({
  baseURL: process.env.FASTVIDEO_BASE_URL || "http://127.0.0.1:8000/v1",
  apiKey: process.env.FASTVIDEO_API_KEY || "local",
  timeout: 60_000,
  maxRetries: 0,
});

let video = await client.videos.create({
  model: process.env.FASTVIDEO_MODEL || "fasth3",
  prompt: "A fox runs through fresh snow.",
});
console.log(`Submitted ${video.id}`);

// The deadline limits polling, not GPU execution. Keep the ID to resume later.
const deadline = performance.now() + 1_800_000;
while (video.status === "queued" || video.status === "in_progress") {
  let remaining = deadline - performance.now();
  if (remaining <= 0) throw new Error(`Polling timed out; job ${video.id} may still be running`);
  await new Promise((resolve) => setTimeout(resolve, Math.min(2000, remaining)));
  remaining = deadline - performance.now();
  if (remaining <= 0) throw new Error(`Polling timed out; job ${video.id} may still be running`);
  video = await client.videos.retrieve(video.id, { timeout: Math.min(60_000, remaining) });
}

if (video.status !== "completed") {
  throw new Error(`Video ${video.id} failed: ${video.error?.message || video.status}`);
}

const content = await client.videos.downloadContent(video.id);
const output = `${video.id}.mp4`;
await writeFile(output, Buffer.from(await content.arrayBuffer()));
console.log(`Saved ${output}`);
