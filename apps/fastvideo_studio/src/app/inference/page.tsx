'use client';

import CreateJobButton from '@/components/jobs/CreateJobButton';
import { HeaderActions } from '@/components/shell/HeaderActionsContext';
import JobQueue from '@/components/jobs/JobQueue';

export default function InferencePage() {
  return (
    <>
      <HeaderActions>
        <CreateJobButton jobType="inference" />
      </HeaderActions>
      <JobQueue jobType="inference" />
    </>
  );
}
