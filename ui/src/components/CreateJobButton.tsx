'use client';

import { useState } from "react";
import CreateJobModal from "./CreateJobModal";
import { useJobsRefresh } from "@/contexts/JobsRefreshContext";
import buttonStyles from "./styles/Button.module.css";

interface CreateJobButtonProps {
  onJobCreated?: () => void;
}

export default function CreateJobButton({ onJobCreated }: CreateJobButtonProps) {
  const [isModalOpen, setIsModalOpen] = useState(false);
  const { triggerRefresh } = useJobsRefresh();

  const handleSuccess = () => {
    triggerRefresh();
    onJobCreated?.();
  };

  return (
    <>
      <button
        className={`${buttonStyles.btn} ${buttonStyles.btnPrimary}`}
        onClick={() => setIsModalOpen(true)}
      >
        Create Job
      </button>
      <CreateJobModal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        onSuccess={handleSuccess}
      />
    </>
  );
}
