'use client';

import CreateJobButton from '@/components/jobs/CreateJobButton';
import EngineConsole from '@/components/jobs/EngineConsole';
import { HeaderActions } from '@/components/shell/HeaderActionsContext';
import JobQueue from '@/components/jobs/JobQueue';
import WarmModelsPanel from '@/components/jobs/WarmModelsPanel';

export default function InferencePage() {
  return (
    <>
      <HeaderActions>
        <CreateJobButton jobType="inference" />
      </HeaderActions>
      <WarmModelsPanel />
      <EngineConsole />
      <JobQueue jobType="inference" />
    </>
  );
}
