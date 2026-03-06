"use client";

import { useState } from "react";
import CreateJobModal from "./CreateJobModal";
import { useJobsRefresh } from "@/contexts/JobsRefreshContext";
import { WORKLOAD_OPTIONS } from "@/lib/jobConfig";
import type { JobType } from "@/lib/types";
import buttonStyles from "./styles/Button.module.css";
import dropdownStyles from "./styles/Dropdown.module.css";

interface CreateJobButtonProps {
	jobType: JobType;
	onJobCreated?: () => void;
}

export default function CreateJobButton({
	jobType,
	onJobCreated,
}: CreateJobButtonProps) {
	const options = WORKLOAD_OPTIONS[jobType];
	const [modalOpen, setModalOpen] = useState(false);
	const [workloadType, setWorkloadType] = useState(options[0]?.type ?? "t2v");
	const { triggerRefresh } = useJobsRefresh();

	const handleSuccess = () => {
		triggerRefresh();
		onJobCreated?.();
	};

	const openModal = (type: string) => {
		setWorkloadType(type);
		setModalOpen(true);
	};

	return (
		<>
			<div className={dropdownStyles.wrapper}>
				<button
					className={`${buttonStyles.btn} ${buttonStyles.btnPrimary} ${dropdownStyles.trigger}`}
				>
					Create Job
				</button>
				<div className={dropdownStyles.menu} role="menu">
					{options.map((opt) => (
						<button
							key={opt.type}
							className={dropdownStyles.menuItem}
							role="menuitem"
							onClick={() => openModal(opt.type)}
						>
							{opt.label}
							<div className={dropdownStyles.menuItemDesc}>
								{opt.desc}
							</div>
						</button>
					))}
				</div>
			</div>
			<CreateJobModal
				isOpen={modalOpen}
				onClose={() => setModalOpen(false)}
				onSuccess={handleSuccess}
				jobType={jobType}
				workloadType={workloadType}
			/>
		</>
	);
}
