import { describe, expect, it } from "vitest";
import { parseJobJson } from "@/lib/jobJson";

const ok = (text: string) => {
	const r = parseJobJson(text);
	if (!r.ok) throw new Error(r.error);
	return r;
};

describe("parseJobJson", () => {
	it("joins a prompt given as lines and carries the settings over", () => {
		const r = ok(
			JSON.stringify({
				model_id: "MiniMaxAI/MiniMax-H3",
				name: "demo",
				workload_type: "i2v",
				prompt: ["summary:", "Hello.", "", "detailed_description:", "[Shot 1] A."],
				references: [{ source: "/data/a.png", media_type: "image" }],
				guidance_scale: 1,
				num_frames: 243,
				fps: 24,
				num_gpus: 4,
				use_fsdp_inference: true,
			}),
		);
		expect(r.workloadType).toBe("i2v");
		expect(r.job.prompt).toBe("summary:\nHello.\n\ndetailed_description:\n[Shot 1] A.");
		expect(r.job).toMatchObject({
			model_id: "MiniMaxAI/MiniMax-H3",
			name: "demo",
			guidance_scale: 1,
			num_frames: 243,
			num_gpus: 4,
			use_fsdp_inference: true,
			references: [{ source: "/data/a.png", media_type: "image" }],
		});
	});

	it("accepts a plain string prompt", () => {
		expect(ok(JSON.stringify({ model_id: "m", prompt: "one line" })).job.prompt).toBe("one line");
	});

	it("infers the workload when it isn't declared", () => {
		expect(ok(JSON.stringify({ model_id: "m", prompt: "p" })).workloadType).toBe("t2v");
		expect(
			ok(JSON.stringify({ model_id: "m", prompt: "p", references: [{ source: "/a/b.mp4" }] })).workloadType,
		).toBe("i2v");
		expect(ok(JSON.stringify({ model_id: "m", prompt: "p", image_path: "/a.png" })).workloadType).toBe("i2v");
	});

	it("infers a reference's media type from its extension", () => {
		const r = ok(
			JSON.stringify({
				model_id: "m",
				prompt: "p",
				references: [{ source: "/a/b.mp4" }, { source: "/a/c.wav" }, { source: "/a/d.png" }],
			}),
		);
		expect(r.job.references?.map((x) => x.media_type)).toEqual(["video", "audio", "image"]);
	});

	it("ignores settings of the wrong type instead of passing them through", () => {
		const r = ok(JSON.stringify({ model_id: "m", prompt: "p", num_frames: "many", use_fsdp_inference: "yes", extra: 1 }));
		expect(r.job.num_frames).toBeUndefined();
		expect(r.job.use_fsdp_inference).toBeUndefined();
		expect(r.job).not.toHaveProperty("extra");
	});

	it.each([
		["not json", "That file isn't valid JSON."],
		["[1,2]", "Expected a JSON object describing one job."],
		[JSON.stringify({ prompt: "p" }), "The job file has no model_id."],
		[JSON.stringify({ model_id: "m", prompt: 5 }), "prompt must be text or a list of text lines."],
		[JSON.stringify({ model_id: "m", prompt: "p", references: [{}] }), "references[0] needs a source path."],
	])("rejects %s", (text, error) => {
		expect(parseJobJson(text)).toEqual({ ok: false, error });
	});
});
