import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import ChatBar from "./ChatBar";

describe("ChatBar generation mode selection", () => {
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
	});

	it("disables unsupported modes and labels mock playback", () => {
		render(<ChatBar supportedGenerationModes={["t2va"]} mockRuntime />);
		expect(screen.getByRole("option", { name: /FL2VA/ })).toBeDisabled();
		expect(screen.getByRole("option", { name: /Ref2VA/ })).toBeDisabled();
		expect(screen.getByText(/No AI model is generating/)).toBeInTheDocument();
	});
});
