'use client';

import { getJobsList } from "@/lib/api";
import { useEffect, useState, useRef, useCallback } from "react";
import CreateJobButton from "@/components/CreateJobButton";
import JobCard from "@/components/JobCard";
import cardStyles from "@styles/Card.module.css";
import layoutStyles from "./Layout.module.css";

export default function Home() {
  const [jobs, setJobs] = useState<Awaited<ReturnType<typeof getJobsList>>>([]);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  const fetchJobs = useCallback(async () => {
    try {
      const jobsList = await getJobsList();
      setJobs(jobsList);
    } catch (error) {
      console.error("Failed to fetch jobs:", error);
    }
  }, []);

  // Initial fetch
  useEffect(() => {
    getJobsList()
      .then(setJobs)
      .catch((error) => {
        console.error("Failed to fetch jobs:", error);
      });
  }, []);

  // Poll every second if there are running jobs
  useEffect(() => {
    // Clear any existing interval
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }

    const hasRunningJobs = jobs.some(job => job.status === "running");
    
    if (hasRunningJobs) {
      intervalRef.current = setInterval(() => {
        fetchJobs();
      }, 1000);
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
        <div className={layoutStyles.sectionHeader}>
          <h2>Jobs</h2>
          <CreateJobButton onJobCreated={fetchJobs} />
        </div>
        <div id="jobs-container">
          {jobs.length === 0 ? (
            <p className={layoutStyles.placeholder}>No jobs yet. Create one above.</p>
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
