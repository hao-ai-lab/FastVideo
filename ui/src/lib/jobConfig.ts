// SPDX-License-Identifier: Apache-2.0

import type { JobType } from "./types";

export interface WorkloadOption {
	type: string;
	label: string;
	desc: string;
}

export const WORKLOAD_OPTIONS: Record<JobType, WorkloadOption[]> = {
	inference: [
		{ type: "t2v", label: "T2V", desc: "Text to Video" },
		{ type: "i2v", label: "I2V", desc: "Image to Video" },
		{ type: "t2i", label: "T2I", desc: "Text to Image" },
	],
	finetuning: [
		{
			type: "full_t2v",
			label: "Full",
			desc: "Full finetune Text to Video",
		},
		{
			type: "vsa_t2v",
			label: "VSA",
			desc: "Video Sparse Attention",
		},
		{
			type: "ode_init",
			label: "ODE Init",
			desc: "ODE-init consistency finetune",
		},
		{
			type: "lora_t2v",
			label: "LoRA",
			desc: "Low Rank Adaptation",
		},
	],
	distillation: [
		{
			type: "dmd_t2v",
			label: "DMD",
			desc: "Distribution Matching Distillation",
		},
		{
			type: "self_forcing_t2v",
			label: "Self-forcing",
			desc: "Self-forcing distillation",
		},
	]
};
