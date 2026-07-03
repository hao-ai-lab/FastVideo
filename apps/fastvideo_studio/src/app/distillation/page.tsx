'use client';

import CreateJobButton from '@/components/jobs/CreateJobButton';
import { HeaderActions } from '@/components/shell/HeaderActionsContext';
import JobQueue from '@/components/jobs/JobQueue';

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
