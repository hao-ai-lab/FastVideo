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
			label: "Full T2V",
			desc: "Full finetune Text to Video",
		},
		{
			type: "full_i2v",
			label: "Full I2V",
			desc: "Full finetune Image to Video",
		},
		{
			type: "vsa_t2v",
			label: "VSA T2V",
			desc: "VSA finetune Text to Video",
		},
		{
			type: "vsa_i2v",
			label: "VSA I2V",
			desc: "VSA finetune Image to Video",
		},
		{
			type: "ode_init",
			label: "ODE Init",
			desc: "ODE-init consistency finetune",
		},
		{
			type: "matrixgame_i2v",
			label: "MatrixGame I2V",
			desc: "MatrixGame 2.0 I2V finetune",
		},
		{
			type: "ltx2_t2v",
			label: "LTX2 T2V",
			desc: "LTX-2 Text to Video finetune",
		},
		{
			type: "lora_t2v",
			label: "LoRA T2V",
			desc: "LoRA finetune Text to Video",
		},
		{
			type: "lora_i2v",
			label: "LoRA I2V",
			desc: "LoRA finetune Image to Video",
		},
	],
	distillation: [
		{
			type: "dmd_t2v",
			label: "DMD T2V",
			desc: "DMD distillation Text to Video",
		},
		{
			type: "dmd_i2v",
			label: "DMD I2V",
			desc: "DMD distillation Image to Video",
		},
		{
			type: "self_forcing_t2v",
			label: "Self-forcing T2V",
			desc: "Self-forcing distillation T2V",
		},
		{
			type: "self_forcing_i2v",
			label: "Self-forcing I2V",
			desc: "Self-forcing distillation I2V",
		},
	],
	lora: [
		{ type: "lora_t2v", label: "LoRA T2V", desc: "LoRA finetune Text to Video" },
		{ type: "lora_i2v", label: "LoRA I2V", desc: "LoRA finetune Image to Video" },
	],
};
