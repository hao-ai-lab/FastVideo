/**
 * Read a job file (the `ref2va_*.json` shape) into the create-job form's
 * job shape. Keys match the API's CreateJobRequest; `prompt` may be a string
 * or a list of lines (joined with newlines), since job files keep it as lines
 * for readability.
 */
import type { JobLike } from "@/lib/jobToFields";

export type JobJsonResult =
	| { ok: true; job: JobLike; workloadType: "t2v" | "i2v" | "t2i" }
	| { ok: false; error: string };

const WORKLOADS = ["t2v", "i2v", "t2i"] as const;
const MEDIA_TYPES = ["image", "video", "audio"] as const;
const VIDEO_EXT = /\.(mp4|mov|mkv|webm|avi)$/i;
const AUDIO_EXT = /\.(wav|mp3|flac|m4a|ogg)$/i;

const STRING_KEYS = ["name", "image_path", "last_image_path", "negative_prompt"] as const;
const NUMBER_KEYS = [
	"num_inference_steps",
	"num_frames",
	"height",
	"width",
	"guidance_scale",
	"guidance_rescale",
	"fps",
	"seed",
	"num_gpus",
	"vsa_sparsity",
	"tp_size",
	"sp_size",
] as const;
const BOOLEAN_KEYS = [
	"dit_cpu_offload",
	"dit_layerwise_offload",
	"text_encoder_cpu_offload",
	"vae_cpu_offload",
	"image_encoder_cpu_offload",
	"use_fsdp_inference",
	"enable_torch_compile",
] as const;

function inferMediaType(source: string): (typeof MEDIA_TYPES)[number] {
	if (VIDEO_EXT.test(source)) return "video";
	if (AUDIO_EXT.test(source)) return "audio";
	return "image";
}

export function parseJobJson(text: string): JobJsonResult {
	let raw: unknown;
	try {
		raw = JSON.parse(text);
	} catch {
		return { ok: false, error: "That file isn't valid JSON." };
	}
	if (typeof raw !== "object" || raw === null || Array.isArray(raw)) {
		return { ok: false, error: "Expected a JSON object describing one job." };
	}
	const data = raw as Record<string, unknown>;

	if (typeof data.model_id !== "string" || !data.model_id.trim()) {
		return { ok: false, error: "The job file has no model_id." };
	}

	let prompt: string;
	if (typeof data.prompt === "string") prompt = data.prompt;
	else if (Array.isArray(data.prompt) && data.prompt.every((l) => typeof l === "string")) {
		prompt = (data.prompt as string[]).join("\n");
	} else {
		return { ok: false, error: "prompt must be text or a list of text lines." };
	}

	const references: { source: string; media_type: string }[] = [];
	if (data.references !== undefined && data.references !== null) {
		if (!Array.isArray(data.references)) return { ok: false, error: "references must be a list." };
		for (const [i, ref] of data.references.entries()) {
			const r = ref as Record<string, unknown> | null;
			if (!r || typeof r.source !== "string" || !r.source) {
				return { ok: false, error: `references[${i}] needs a source path.` };
			}
			const mediaType =
				typeof r.media_type === "string" && (MEDIA_TYPES as readonly string[]).includes(r.media_type)
					? r.media_type
					: inferMediaType(r.source);
			references.push({ source: r.source, media_type: mediaType });
		}
	}

	const job: JobLike = { id: `imported-${Date.now()}`, model_id: data.model_id.trim(), prompt, references };
	const target = job as unknown as Record<string, unknown>;
	for (const key of STRING_KEYS) if (typeof data[key] === "string") target[key] = data[key];
	for (const key of NUMBER_KEYS) if (typeof data[key] === "number" && Number.isFinite(data[key])) target[key] = data[key];
	for (const key of BOOLEAN_KEYS) if (typeof data[key] === "boolean") target[key] = data[key];

	const declared = (WORKLOADS as readonly string[]).includes(data.workload_type as string)
		? (data.workload_type as (typeof WORKLOADS)[number])
		: null;
	const workloadType = declared ?? (references.length > 0 || job.image_path ? "i2v" : "t2v");
	job.workload_type = workloadType;

	return { ok: true, job, workloadType };
}
