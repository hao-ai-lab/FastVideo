'use client';

import { getJobsList } from "@/lib/api";
import { useEffect, useState, useRef, useCallback } from "react";
import JobCard from "@/components/JobCard";
import CreateJobButton from "@/components/CreateJobButton";
import { useJobsRefresh } from "@/contexts/JobsRefreshContext";
import { useHeaderActions } from "@/contexts/ActiveTabContext";
import { useActiveJob } from "@/contexts/ActiveJobContext";
import type { JobType } from "@/lib/types";
import cardStyles from "@styles/Card.module.css";
import layoutStyles from "@/app/Layout.module.css";

interface JobQueuePageProps {
  jobType: JobType;
}

export default function JobQueuePage({ jobType }: JobQueuePageProps) {
  useHeaderActions([
    <CreateJobButton key="create-job" jobType={jobType} />,
  ]);
  const { activeJobId, setActiveJobId, setActiveJob } = useActiveJob();
  const [jobs, setJobs] = useState<Awaited<ReturnType<typeof getJobsList>>>([]);
  const activeJob = activeJobId
    ? jobs.find((j) => j.id === activeJobId) ?? null
    : null;

  // Sync activeJob to context for layout-level SecondarySidebar
  useEffect(() => {
    setActiveJob(activeJob);
  }, [activeJob, setActiveJob]);

  // Clear selection if the selected job was deleted
  useEffect(() => {
    if (activeJobId && !activeJob) {
      setActiveJobId(null);
    }
  }, [activeJobId, activeJob, setActiveJobId]);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  const { registerRefresh } = useJobsRefresh();

  const fetchJobs = useCallback(async () => {
    try {
      const jobsList = await getJobsList(jobType);
      setJobs(jobsList);
    } catch (error) {
      console.error("Failed to fetch jobs:", error);
    }
  }, [jobType]);

  useEffect(() => {
    getJobsList(jobType)
      .then(setJobs)
      .catch((error) => {
        console.error("Failed to fetch jobs:", error);
      });
  }, [jobType]);

  useEffect(() => {
    return registerRefresh(fetchJobs);
  }, [registerRefresh, fetchJobs]);

  useEffect(() => {
    const hasActiveJobs = jobs.some(
      (job) => job.status === "running" || job.status === "pending"
    );

    if (hasActiveJobs) {
      intervalRef.current = setInterval(() => {
        fetchJobs();
      }, 1000);
    } else if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, [jobs, fetchJobs]);

  return (
    <main className={layoutStyles.main}>
      <section className={cardStyles.card}>
        <div id="jobs-container">
          {jobs.length === 0 ? (
            <p className={layoutStyles.placeholder}>
              No {jobType} jobs yet. Create one above.
            </p>
          ) : (
            jobs.map((job) => (
              <JobCard key={job.id} job={job} onJobUpdated={fetchJobs} />
            ))
          )}
        </div>
      </section>
    </main>
  );
}
