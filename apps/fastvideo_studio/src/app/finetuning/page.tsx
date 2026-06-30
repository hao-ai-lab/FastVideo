'use client';

import * as React from 'react';

import CreateJobButton from '@/components/CreateJobButton';
import { useHeaderActions } from '@/components/HeaderActionsContext';
import JobQueue from '@/components/JobQueue';
import type { JobType } from '@/lib/types';

// 'lora' is a backend job_type (LoRA finetunes) that isn't part of the
// JobType union; the finetuning queue lists both alongside full finetunes.
const FINETUNING_LIST = ['finetuning', 'lora'] as JobType[];

export default function FinetuningPage() {
  const { setActions } = useHeaderActions();

  React.useEffect(() => {
    setActions(<CreateJobButton jobType="finetuning" />);
    return () => setActions(null);
  }, [setActions]);

  return <JobQueue jobType="finetuning" jobTypesForList={FINETUNING_LIST} />;
}
