'use client';

import { useRouter } from "next/navigation";
import { useState } from "react";
import CreateJobModal from "./CreateJobModal";
import buttonStyles from "./styles/Button.module.css";

export default function CreateJobButton() {
  const router = useRouter();
  const [isModalOpen, setIsModalOpen] = useState(false);

  const handleSuccess = () => {
    router.refresh();
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
