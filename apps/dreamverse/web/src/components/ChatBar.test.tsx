import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterAll, beforeAll, describe, expect, it, vi } from "vitest";

import ChatBar from "./ChatBar";

// JSDOM does not implement the pointer/scroll APIs used by the Radix popup.
const domPolyfills = {
	hasPointerCapture: () => false,
	releasePointerCapture: () => {},
	scrollIntoView: () => {},
};
const originalDescriptors = new Map<string, PropertyDescriptor | undefined>();
beforeAll(() => {
	for (const [name, implementation] of Object.entries(domPolyfills)) {
		originalDescriptors.set(name, Object.getOwnPropertyDescriptor(HTMLElement.prototype, name));
		Object.defineProperty(HTMLElement.prototype, name, { configurable: true, value: implementation });
	}
	vi.stubGlobal("PointerEvent", MouseEvent);
});
afterAll(() => {
	for (const [name, descriptor] of originalDescriptors) {
		if (descriptor) Object.defineProperty(HTMLElement.prototype, name, descriptor);
		else Reflect.deleteProperty(HTMLElement.prototype, name);
	}
	vi.unstubAllGlobals();
});

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

	it("shows only mode abbreviations and keeps explanations in the tooltip", async () => {
		const user = userEvent.setup();
		render(<ChatBar />);

		expect(screen.getByRole("combobox", { name: "Generation mode" }))
			.toHaveAttribute("title", "Text to video + audio. Start with a text prompt; no reference asset is required.");
		expect(screen.queryByText("Start with a text prompt; no reference asset is required."))
			.not.toBeInTheDocument();
		expect(screen.queryByRole("listbox")).not.toBeInTheDocument();
		await user.click(screen.getByRole("combobox", { name: "Generation mode" }));
		const menu = await screen.findByRole("listbox");
		expect(within(menu).getAllByRole("option").map((option) => option.textContent))
			.toEqual(["T2VA", "FL2VA", "Ref2VA"]);
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
		expect(modeSelect).toHaveTextContent("T2VA");

		await user.click(modeSelect);
		await user.click(await screen.findByRole("option", { name: "Ref2VA" }));

		expect(onGenerationModeChange).toHaveBeenCalledWith("ref2va");
		expect(screen.queryByRole("listbox")).not.toBeInTheDocument();
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
		await user.click(modeSelect);
		expect(screen.queryByRole("listbox")).not.toBeInTheDocument();
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

	it("disables unsupported modes and labels mock playback", async () => {
		const user = userEvent.setup();
		render(<ChatBar supportedGenerationModes={["t2va"]} mockRuntime />);
		expect(screen.getByText(/No AI model is generating/)).toBeInTheDocument();
		await user.click(screen.getByRole("combobox", { name: "Generation mode" }));
		expect(await screen.findByRole("option", { name: "FL2VA" })).toHaveAttribute("aria-disabled", "true");
		expect(screen.getByRole("option", { name: "Ref2VA" })).toHaveAttribute("aria-disabled", "true");
		expect(screen.getByRole("option", { name: "Ref2VA" }))
			.toHaveAttribute("title", "References to video + audio (unavailable on this runtime)");
	});

	it("closes the menu with Escape and restores focus to Mode", async () => {
		const onGenerationModeChange = vi.fn();
		const user = userEvent.setup();
		render(<ChatBar onGenerationModeChange={onGenerationModeChange} />);
		const modeSelect = screen.getByRole("combobox", { name: "Generation mode" });
		await user.click(modeSelect);
		await screen.findByRole("listbox");
		await user.keyboard("{Escape}");
		expect(screen.queryByRole("listbox")).not.toBeInTheDocument();
		await waitFor(() => expect(modeSelect).toHaveFocus());
		expect(onGenerationModeChange).not.toHaveBeenCalled();
	});

	it("supports choosing a mode with the keyboard", async () => {
		const onGenerationModeChange = vi.fn();
		const user = userEvent.setup();
		render(<ChatBar onGenerationModeChange={onGenerationModeChange} />);
		await user.click(screen.getByRole("textbox", { name: "Continuation prompt" }));
		await user.tab();
		expect(screen.getByRole("combobox", { name: "Generation mode" })).toHaveFocus();
		await user.keyboard("{ArrowDown}");
		await waitFor(() => expect(screen.getByRole("option", { name: "T2VA" })).toHaveFocus());
		await user.keyboard("{ArrowDown}");
		await waitFor(() => expect(screen.getByRole("option", { name: "FL2VA" })).toHaveFocus());
		await user.keyboard("{Enter}");
		expect(onGenerationModeChange).toHaveBeenCalledWith("fl2va");
		expect(screen.queryByRole("listbox")).not.toBeInTheDocument();
	});
});
