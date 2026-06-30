'use client';

import * as React from 'react';

import CreateJobButton from '@/components/CreateJobButton';
import { useHeaderActions } from '@/components/HeaderActionsContext';
import JobQueue from '@/components/JobQueue';

export default function DistillationPage() {
  const { setActions } = useHeaderActions();

  React.useEffect(() => {
    setActions(<CreateJobButton jobType="distillation" />);
    return () => setActions(null);
  }, [setActions]);

  return <JobQueue jobType="distillation" />;
}
