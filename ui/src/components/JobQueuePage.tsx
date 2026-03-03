'use client';

import { getJobsList } from "@/lib/api";
import { useEffect, useState, useRef, useCallback } from "react";
import JobCard from "@/components/JobCard";
import { useJobsRefresh } from "@/contexts/JobsRefreshContext";
import cardStyles from "@styles/Card.module.css";
import layoutStyles from "@/app/Layout.module.css";

export default function JobQueuePage() {
  const [jobs, setJobs] = useState<Awaited<ReturnType<typeof getJobsList>>>([]);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  const { registerRefresh } = useJobsRefresh();

  const fetchJobs = useCallback(async () => {
    try {
      const jobsList = await getJobsList();
      setJobs(jobsList);
    } catch (error) {
      console.error("Failed to fetch jobs:", error);
    }
  }, []);

  useEffect(() => {
    getJobsList()
      .then(setJobs)
      .catch((error) => {
        console.error("Failed to fetch jobs:", error);
      });
  }, []);

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
              No jobs yet. Create one above.
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
