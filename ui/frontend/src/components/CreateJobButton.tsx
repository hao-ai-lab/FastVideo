'use client';

import { useState } from "react";
import CreateJobModal from "./CreateJobModal";
import buttonStyles from "./styles/Button.module.css";

interface CreateJobButtonProps {
  onJobCreated?: () => void;
}

export default function CreateJobButton({ onJobCreated }: CreateJobButtonProps) {
  const [isModalOpen, setIsModalOpen] = useState(false);

  const handleSuccess = () => {
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
