import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import AssetList from "./AssetList";
import type { GenerationAsset } from "@/lib/generationMode";

const frame: GenerationAsset = { asset_id: "first", kind: "image", name: "frame.png", mime_type: "image/png", size: 2000, url: "/assets/first" };
const video: GenerationAsset = { asset_id: "video", kind: "video", name: "motion.mp4", mime_type: "video/mp4", size: 2000, url: "/assets/video" };
function props() {
	return { assets: [frame, video], onUpload: vi.fn(), onAssign: vi.fn(), onRemove: vi.fn(), onUnselect: vi.fn(), onMove: vi.fn(), onMissing: vi.fn() };
}

describe("Asset List", () => {
	it("uploads to the library and assigns images to endpoint roles", async () => {
		const callbacks = props();
		const user = userEvent.setup();
		render(<AssetList {...callbacks} mode="fl2va" conditioning={[]} />);
		await user.selectOptions(screen.getByRole("combobox", { name: "First frame" }), "first");
		expect(callbacks.onAssign).toHaveBeenCalledWith("first", "first_frame");
		expect(screen.getByRole("combobox", { name: "Last frame" })).toHaveValue("");
		expect(screen.queryByRole("option", { name: "motion.mp4" })).not.toBeInTheDocument();
		const file = new File(["image"], "new.png", { type: "image/png" });
		await user.upload(screen.getByLabelText("Upload assets", { selector: "input" }), file);
		expect(callbacks.onUpload).toHaveBeenCalledWith([file]);
	});
	it("exposes accessible ordering and removal controls for multimodal references", async () => {
		const callbacks = props();
		const user = userEvent.setup();
		render(<AssetList {...callbacks} mode="ref2va" conditioning={[{ asset_id: "first", role: "reference" }, { asset_id: "video", role: "reference" }]} />);
		await user.click(screen.getByRole("button", { name: "Move motion.mp4 up" }));
		expect(callbacks.onMove).toHaveBeenCalledWith(1, 0);
		await user.click(screen.getByRole("button", { name: "Unselect frame.png" }));
		expect(callbacks.onUnselect).toHaveBeenCalledWith(0);
		expect(screen.getByRole("button", { name: "Move frame.png up" })).toBeDisabled();
	});
	it("locks uploads and assignments while starting generation", () => {
		render(<AssetList {...props()} mode="fl2va" conditioning={[]} locked />);
		expect(screen.getByRole("button", { name: "Upload assets" })).toBeDisabled();
		expect(screen.getByRole("combobox", { name: "First frame" })).toBeDisabled();
	});
});
