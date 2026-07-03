'use client';

import CreateJobButton from '@/components/CreateJobButton';
import { HeaderActions } from '@/components/HeaderActionsContext';
import JobQueue from '@/components/JobQueue';

export default function DistillationPage() {
  return (
    <>
      <HeaderActions>
        <CreateJobButton jobType="distillation" />
      </HeaderActions>
      <JobQueue jobType="distillation" />
    </>
  );
}
