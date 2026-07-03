'use client';

import CreateJobButton from '@/components/CreateJobButton';
import { HeaderActions } from '@/components/HeaderActionsContext';
import JobQueue from '@/components/JobQueue';

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
