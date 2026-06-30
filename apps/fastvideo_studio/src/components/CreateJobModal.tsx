'use client';

import type { JobType } from '@/lib/types';

// Stub — replaced by unit L2 with the full create-job modal.
// The prop signature is a hard contract: L1's CreateJobButton renders this,
// and L2's implementation must accept exactly these props.
export interface CreateJobModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSuccess: () => void;
  jobType: JobType;
  workloadType: string;
}

export default function CreateJobModal(_props: CreateJobModalProps) {
  return null;
}
