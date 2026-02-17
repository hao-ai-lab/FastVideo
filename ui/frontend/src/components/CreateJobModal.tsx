'use client';

import { createJob } from "@/lib/api";
import { useEffect, useState } from "react";
import modalStyles from "./Modal.module.css";
import formStyles from "./Form.module.css";
import cardStyles from "./Card.module.css";
import buttonStyles from "./Button.module.css";

interface CreateJobModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSuccess: () => void;
}

export default function CreateJobModal({ isOpen, onClose, onSuccess }: CreateJobModalProps) {
  const [modelId, setModelId] = useState("");
  const [prompt, setPrompt] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);

  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isOpen && !isSubmitting) {
        onClose();
      }
    };
    if (isOpen) {
      document.addEventListener('keydown', handleEscape);
      return () => document.removeEventListener('keydown', handleEscape);
    }
  }, [isOpen, isSubmitting, onClose]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);
    try {
      await createJob({ model_id: modelId, prompt });
      setModelId("");
      setPrompt("");
      onSuccess();
      onClose();
    } catch (error) {
      console.error("Failed to create job:", error);
      // You could add error handling/toast here
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleClose = () => {
    if (!isSubmitting) {
      setModelId("");
      setPrompt("");
      onClose();
    }
  };

  if (!isOpen) return null;

  return (
    <div className={modalStyles.modal}>
      <div className={modalStyles.modalBackdrop} onClick={handleClose} />
      <div className={`${modalStyles.modalContent} ${modalStyles.modalForm}`}>
        <button
          className={modalStyles.modalClose}
          onClick={handleClose}
          disabled={isSubmitting}
          aria-label="Close"
        >
          ×
        </button>
        <div className={cardStyles.card} style={{ margin: 0, border: 'none' }}>
          <h2>New Job</h2>
          <form onSubmit={handleSubmit} autoComplete="off">
            <div className={formStyles.formRow}>
              <label htmlFor="modal-modelId">Model ID</label>
              <input
                type="text"
                name="modelId"
                id="modal-modelId"
                value={modelId}
                onChange={(e) => setModelId(e.target.value)}
                placeholder="e.g. Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
                required
                disabled={isSubmitting}
              />
            </div>
            <div className={formStyles.formRow}>
              <label htmlFor="modal-prompt">Prompt</label>
              <textarea
                name="prompt"
                id="modal-prompt"
                rows={3}
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                placeholder="A curious raccoon peers through a vibrant field of yellow sunflowers…"
                required
                disabled={isSubmitting}
              />
            </div>
            <button type="submit" className={`${buttonStyles.btn} ${buttonStyles.btnPrimary}`} disabled={isSubmitting}>
              {isSubmitting ? "Creating..." : "Create Job"}
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}
