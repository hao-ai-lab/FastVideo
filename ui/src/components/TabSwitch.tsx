"use client";

import tabSwitchStyles from "./styles/TabSwitch.module.css";

export interface TabSwitchOption {
	id: string;
	label: string;
}

interface TabSwitchProps {
	options: TabSwitchOption[];
	value: string;
	onChange: (id: string) => void;
	disabled?: boolean;
}

export default function TabSwitch({
	options,
	value,
	onChange,
	disabled = false,
}: TabSwitchProps) {
	return (
		<div className={tabSwitchStyles.tabs} role="tablist">
			{options.map((opt) => (
				<button
					key={opt.id}
					type="button"
					role="tab"
					aria-selected={value === opt.id}
					className={`${tabSwitchStyles.tab} ${
						value === opt.id ? tabSwitchStyles.tabActive : ""
					}`}
					onClick={() => onChange(opt.id)}
					disabled={disabled}
				>
					{opt.label}
				</button>
			))}
		</div>
	);
}
