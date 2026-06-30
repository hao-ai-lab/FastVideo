'use client';

import * as React from 'react';

import CreateJobButton from '@/components/CreateJobButton';
import { useHeaderActions } from '@/components/HeaderActionsContext';
import JobQueue from '@/components/JobQueue';

export default function InferencePage() {
  const { setActions } = useHeaderActions();

  React.useEffect(() => {
    setActions(<CreateJobButton jobType="inference" />);
    return () => setActions(null);
  }, [setActions]);

  return <JobQueue jobType="inference" />;
}
