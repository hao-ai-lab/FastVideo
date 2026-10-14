import { describe, expect, it } from "vitest";
import { computeElapsed, formatDuration } from "@/lib/jobTiming";
import type { Job } from "@/lib/types";

function job(over: Partial<Job>): Job {
	return { id: "j", model_id: "m", prompt: "", status: "pending", created_at: 0, ...over } as Job;
}

describe("formatDuration", () => {
	it("formats seconds, minutes and hours", () => {
		expect(formatDuration(42)).toBe("42s");
		expect(formatDuration(342)).toBe("5m 42s");
		expect(formatDuration(300)).toBe("5m");
		expect(formatDuration(3900)).toBe("1h 5m");
	});
});

describe("computeElapsed", () => {
	it("is empty until the job leaves the queue", () => {
		expect(computeElapsed(job({ status: "pending" }), 5_000_000)).toBeNull();
		expect(computeElapsed(job({ status: "queued", started_at: null }), 5_000_000)).toBeNull();
	});

	it("counts up from started_at while running", () => {
		const startedAt = 1_790_000_000; // server time, in seconds
		expect(computeElapsed(job({ status: "running", started_at: startedAt }), (startedAt + 90) * 1000)).toBe("1m 30s");
	});

	it("freezes at finished_at once done, whatever the clock says", () => {
		const done = job({ status: "completed", started_at: 1000, finished_at: 1342 });
		expect(computeElapsed(done, 9_999_999_999)).toBe("5m 42s");
		expect(computeElapsed(done, 0)).toBe("5m 42s");
	});
});
