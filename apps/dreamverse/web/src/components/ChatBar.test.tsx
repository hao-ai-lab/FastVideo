import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import ChatBar from "./ChatBar";

describe("ChatBar generation mode selection", () => {
	it("places Mode and the prompt input inside the same composer", () => {
		render(<ChatBar />);

		const composer = screen.getByRole("group", { name: "Prompt composer" });
		expect(within(composer).getByText("Mode", { exact: true })).toBeVisible();
		expect(within(composer).getByRole("combobox", { name: "Generation mode" }))
			.toBeVisible();
		expect(within(composer).getByRole("textbox", { name: "Continuation prompt" }))
			.toBeVisible();
		expect(screen.queryByText("Generation mode", { exact: true }))
			.not.toBeInTheDocument();
	});

	it("shows only mode abbreviations and keeps explanations in the tooltip", () => {
		render(<ChatBar />);

		expect(screen.getAllByRole("option").map((option) => option.textContent))
			.toEqual(["T2VA", "FL2VA", "Ref2VA"]);
		expect(screen.getByRole("combobox", { name: "Generation mode" }))
			.toHaveAttribute("title", "Text to video + audio. Start with a text prompt; no reference asset is required.");
		expect(screen.queryByText("Start with a text prompt; no reference asset is required."))
			.not.toBeInTheDocument();
	});

	it("defaults to T2VA and reports a selected mode", async () => {
		const onGenerationModeChange = vi.fn();
		const user = userEvent.setup();

		render(
			<ChatBar
				canJoinSession
				continuationDraft="A lighthouse in a storm"
				onGenerationModeChange={onGenerationModeChange}
			/>,
		);

		const modeSelect = screen.getByRole("combobox", { name: "Generation mode" });
		expect(modeSelect).toHaveValue("t2va");

		await user.selectOptions(modeSelect, "ref2va");

		expect(onGenerationModeChange).toHaveBeenCalledWith("ref2va");
	});

	it("hides mode selection after generation starts", () => {
		render(<ChatBar sessionStarted />);

		expect(screen.queryByRole("combobox", { name: "Generation mode" }))
			.not.toBeInTheDocument();
		expect(within(screen.getByRole("group", { name: "Prompt composer" }))
			.getByRole("textbox", { name: "Continuation prompt" })).toBeVisible();
	});

	it("disables mode selection and prompt editing while generation is busy", async () => {
		const onGenerationModeChange = vi.fn();
		const user = userEvent.setup();
		render(<ChatBar isGenerating onGenerationModeChange={onGenerationModeChange} />);

		const composer = screen.getByRole("group", { name: "Prompt composer" });
		const modeSelect = within(composer).getByRole("combobox", { name: "Generation mode" });
		expect(modeSelect).toBeDisabled();
		expect(within(composer).getByRole("textbox", { name: "Continuation prompt" }))
			.toBeDisabled();
		await user.selectOptions(modeSelect, "fl2va");
		expect(onGenerationModeChange).not.toHaveBeenCalled();
	});

	it("still submits the prompt with Enter from the combined composer", async () => {
		const onGenerate = vi.fn();
		const user = userEvent.setup();
		render(<ChatBar canJoinSession continuationDraft="A lighthouse in a storm" onGenerate={onGenerate} />);

		const input = within(screen.getByRole("group", { name: "Prompt composer" }))
			.getByRole("textbox", { name: "Continuation prompt" });
		await user.click(input);
		await user.keyboard("{Enter}");
		expect(onGenerate).toHaveBeenCalledTimes(1);
	});

	it("disables unsupported modes and labels mock playback", () => {
		render(<ChatBar supportedGenerationModes={["t2va"]} mockRuntime />);
		expect(screen.getByRole("option", { name: "FL2VA" })).toBeDisabled();
		expect(screen.getByRole("option", { name: "Ref2VA" })).toBeDisabled();
		expect(screen.getByRole("option", { name: "Ref2VA" }))
			.toHaveAttribute("title", "References to video + audio (unavailable on this runtime)");
		expect(screen.getByText(/No AI model is generating/)).toBeInTheDocument();
	});
});
