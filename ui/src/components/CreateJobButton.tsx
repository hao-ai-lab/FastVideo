'use client';

import { useState } from "react";
import CreateJobModal from "./CreateJobModal";
import { useJobsRefresh } from "@/contexts/JobsRefreshContext";
import buttonStyles from "./styles/Button.module.css";
import dropdownStyles from "./styles/Dropdown.module.css";

export type WorkloadType = "t2v" | "i2v" | "t2i";

interface CreateJobButtonProps {
  onJobCreated?: () => void;
}

const WORKLOAD_OPTIONS: { type: WorkloadType; label: string; desc: string }[] = [
  { type: "t2v", label: "T2V", desc: "Text to Video" },
  { type: "i2v", label: "I2V", desc: "Image to Video" },
  { type: "t2i", label: "T2I", desc: "Text to Image" },
];

export default function CreateJobButton({ onJobCreated }: CreateJobButtonProps) {
  const [modalOpen, setModalOpen] = useState(false);
  const [workloadType, setWorkloadType] = useState<WorkloadType>("t2v");
  const { triggerRefresh } = useJobsRefresh();

  const handleSuccess = () => {
    triggerRefresh();
    onJobCreated?.();
  };

  const openModal = (type: WorkloadType) => {
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
          {WORKLOAD_OPTIONS.map((opt) => (
            <button
              key={opt.type}
              className={dropdownStyles.menuItem}
              role="menuitem"
              onClick={() => openModal(opt.type)}
            >
              {opt.label}
              <div className={dropdownStyles.menuItemDesc}>{opt.desc}</div>
            </button>
          ))}
        </div>
      </div>
      <CreateJobModal
        isOpen={modalOpen}
        onClose={() => setModalOpen(false)}
        onSuccess={handleSuccess}
        workloadType={workloadType}
      />
    </>
  );
}
