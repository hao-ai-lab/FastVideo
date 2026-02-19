'use client';

import { Job } from "@/lib/types";
import Link from "next/link";
import { startJob, stopJob, deleteJob, getJobLogs, downloadJobLog } from "@/lib/api";
import jobCardStyles from "@styles/JobCard.module.css";
import badgeStyles from "@styles/Badge.module.css";
import buttonStyles from "@styles/Button.module.css";
import { useState, useEffect, useRef } from "react";

interface JobCardProps {
  job: Job;
  onJobUpdated?: () => void;
}

export default function JobCard({ job, onJobUpdated }: JobCardProps) {
  const [isLoading, setIsLoading] = useState(false);
  const [showConsole, setShowConsole] = useState(false);
  const [logs, setLogs] = useState<string[]>([]);
  const [currentTime, setCurrentTime] = useState(Date.now());
  const logAfterRef = useRef(0);
  const isPollingRef = useRef(false);
  const consoleRef = useRef<HTMLPreElement>(null);
  const previousJobIdRef = useRef<string | null>(null);
  const previousStatusRef = useRef<string | null>(null);
  const badgeClass = `badge${job.status.charAt(0).toUpperCase() + job.status.slice(1)}`;
  
  // Format duration in a human-readable way
  const formatDuration = (seconds: number): string => {
    // Round to nearest second for display
    const roundedSeconds = Math.round(seconds);
    
    if (roundedSeconds < 60) {
      return `${roundedSeconds}s`;
    } else if (roundedSeconds < 3600) {
      const mins = Math.floor(roundedSeconds / 60);
      const secs = roundedSeconds % 60;
      return secs > 0 ? `${mins}m ${secs}s` : `${mins}m`;
    } else {
      const hours = Math.floor(roundedSeconds / 3600);
      const mins = Math.floor((roundedSeconds % 3600) / 60);
      return mins > 0 ? `${hours}h ${mins}m` : `${hours}h`;
    }
  };
  
  // Calculate elapsed time from API data (started_at and finished_at)
  const getElapsedTime = (): string | null => {
    // Only show timer if job has started
    if (!job.started_at) return null;
    
    let endTime: number;
    
    if (job.status === "running") {
      // For running jobs, use current time for live updates
      endTime = currentTime;
    } else {
      // For completed/failed/stopped jobs, require finished_at from API
      if (!job.finished_at) return null;
      endTime = job.finished_at;
    }
    
    // Convert Python timestamps (seconds) to JavaScript timestamps (milliseconds)
    const startedAtMs = job.started_at < 1e12 ? job.started_at * 1000 : job.started_at;
    const endTimeMs = endTime < 1e12 ? endTime * 1000 : endTime;
    
    const elapsedSeconds = (endTimeMs - startedAtMs) / 1000;
    
    // Don't show timer if elapsed time is negative or zero
    if (elapsedSeconds <= 0) return null;
    return formatDuration(elapsedSeconds);
  };
  
  // Update current time every second for running jobs to show live elapsed time
  useEffect(() => {
    if (job.status !== "running" || !job.started_at) {
      return;
    }
    
    const interval = setInterval(() => {
      setCurrentTime(Date.now());
    }, 1000);
    
    return () => clearInterval(interval);
  }, [job.status, job.started_at]);
  
  const handleStart = async (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (isLoading || job.status === "running" || job.status === "completed") return;
    
    setIsLoading(true);
    try {
      await startJob(job.id);
      onJobUpdated?.();
    } catch (error) {
      console.error("Failed to start job:", error);
      alert(error instanceof Error ? error.message : "Failed to start job");
    } finally {
      setIsLoading(false);
    }
  };

  const handleStop = async (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (isLoading || job.status !== "running") return;
    
    setIsLoading(true);
    try {
      await stopJob(job.id);
      onJobUpdated?.();
    } catch (error) {
      console.error("Failed to stop job:", error);
      alert(error instanceof Error ? error.message : "Failed to stop job");
    } finally {
      setIsLoading(false);
    }
  };

  const handleDelete = async (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (isLoading) return;
    
    if (!confirm("Delete this job?")) return;
    
    setIsLoading(true);
    try {
      await deleteJob(job.id);
      onJobUpdated?.();
    } catch (error) {
      console.error("Failed to delete job:", error);
      alert(error instanceof Error ? error.message : "Failed to delete job");
    } finally {
      setIsLoading(false);
    }
  };

  // Poll for logs when console is open
  useEffect(() => {
    if (!showConsole) {
      // Reset when console is closed
      logAfterRef.current = 0;
      return;
    }

    // Only poll for running or pending jobs
    const shouldPoll = job.status === "running" || job.status === "pending";
    
    let pollInterval: NodeJS.Timeout | null = null;
    let isMounted = true;

    const pollLogs = async () => {
      if (!isMounted || isPollingRef.current) return;
      isPollingRef.current = true;
      try {
        const logData = await getJobLogs(job.id, logAfterRef.current);
        if (isMounted && logData.lines.length > 0) {
          setLogs(prev => [...prev, ...logData.lines]);
          logAfterRef.current = logData.total;
          // Auto-scroll to bottom
          if (consoleRef.current) {
            consoleRef.current.scrollTop = consoleRef.current.scrollHeight;
          }
        }
      } catch (error) {
        // Silently ignore fetch errors for logs
        console.error("Failed to fetch logs:", error);
      } finally {
        isPollingRef.current = false;
      }
    };

    // Initial fetch (always fetch once to show existing logs)
    pollLogs();

    // Only set up polling interval for running/pending jobs
    if (shouldPoll) {
      // Poll every 500ms when job is running, every 2s when pending
      const pollIntervalMs = 2000;
      pollInterval = setInterval(pollLogs, pollIntervalMs);
    }

    return () => {
      isMounted = false;
      if (pollInterval) {
        clearInterval(pollInterval);
      }
    };
  }, [showConsole, job.id, job.status]);

  // Reset logs when console is closed, job changes, or job is restarted
  useEffect(() => {
    const previousJobId = previousJobIdRef.current;
    const previousStatus = previousStatusRef.current;
    const currentJobId = job.id;
    const currentStatus = job.status;
    
    // Detect job restart: same job ID, but status changed from terminal state to pending/running
    const wasTerminal = previousStatus === "failed" || previousStatus === "stopped" || previousStatus === "completed";
    const isRestarting = previousJobId === currentJobId && wasTerminal && (currentStatus === "pending" || currentStatus === "running");
    
    // Reset logs when:
    // 1. Console is closed
    // 2. Switching to a different job
    // 3. Job is restarted (same job, but transitioned from terminal to pending/running)
    if (!showConsole) {
      setLogs([]);
      logAfterRef.current = 0;
    } else if (showConsole && (previousJobId !== currentJobId || isRestarting)) {
      setLogs([]);
      logAfterRef.current = 0;
    }
    
    // Update refs for next comparison
    previousJobIdRef.current = currentJobId;
    previousStatusRef.current = currentStatus;
  }, [showConsole, job.id, job.status]);

  const toggleConsole = () => {
    setShowConsole(!showConsole);
  };

  const handleDownloadLog = async (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (isLoading) return;
    
    setIsLoading(true);
    try {
      const blob = await downloadJobLog(job.id);
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `job_${job.id}.log`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (error) {
      console.error("Failed to download log:", error);
      alert(error instanceof Error ? error.message : "Failed to download log");
    } finally {
      setIsLoading(false);
    }
  };

  const getActionButton = () => {
    if (job.status === "running") {
      return (
        <button
          className={`${buttonStyles.btn} ${buttonStyles.btnStop} ${buttonStyles.btnSmall}`}
          onClick={handleStop}
          disabled={isLoading}
        >
          Stop
        </button>
      );
    } else if (job.status === "failed") {
      return (
        <button
          className={`${buttonStyles.btn} ${buttonStyles.btnStart} ${buttonStyles.btnSmall}`}
          onClick={handleStart}
          disabled={isLoading}
        >
          Restart
        </button>
      );
    } else if (job.status === "pending" || job.status === "stopped") {
      return (
        <button
          className={`${buttonStyles.btn} ${buttonStyles.btnStart} ${buttonStyles.btnSmall}`}
          onClick={handleStart}
          disabled={isLoading}
        >
          Start
        </button>
      );
    }
    return null;
  };

  return (
    <div className={jobCardStyles.jobCard}>
      <Link 
        href={`/jobs/${job.id}`} 
        style={{ textDecoration: "none", color: "inherit" }}
      >
        <div className={jobCardStyles.jobHeader}>
          <span className={jobCardStyles.jobModel}>{job.model_id}</span>
          <span className={`${badgeStyles.badge} ${badgeStyles[badgeClass as keyof typeof badgeStyles] || badgeStyles.badgePending}`}>
            {job.status}
          </span>
        </div>
        <p className={jobCardStyles.jobPrompt}>{job.prompt}</p>
        <div className={jobCardStyles.jobMeta}>
          <span>{job.num_frames} frames</span>
          <span>{job.height}×{job.width}</span>
          {getElapsedTime() && (
            <span className={jobCardStyles.jobDuration}>
              {"⏱ " + getElapsedTime()}
            </span>
          )}
        </div>
      </Link>
      <div className={jobCardStyles.jobActions}>
        {getActionButton()}
        <button
          className={`${buttonStyles.btn} ${buttonStyles.btnDelete} ${buttonStyles.btnSmall}`}
          onClick={handleDelete}
          disabled={isLoading}
        >
          Delete
        </button>
        <button
          className={`${jobCardStyles.toggleButton} ${showConsole ? jobCardStyles.toggleButtonOpen : ''}`}
          onClick={toggleConsole}
          disabled={isLoading}
          title={showConsole ? "Hide logs" : "Show logs"}
        >
          <svg
            width="16"
            height="16"
            viewBox="0 0 16 16"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
          >
            <path
              d="M4 6L8 10L12 6"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </button>
      </div>
      {showConsole && (
        <div className={jobCardStyles.consoleContainer}>
          <div className={jobCardStyles.consoleHeader}>
            <span className={jobCardStyles.consoleTitle}>Console Output</span>
            <div className={jobCardStyles.consoleHeaderRight}>
              {job.status === "running" && (
                <span className={jobCardStyles.consoleStatus}>● Live</span>
              )}
              <button
                className={`${buttonStyles.btn} ${buttonStyles.btnSmall}`}
                onClick={handleDownloadLog}
                disabled={isLoading || !job.log_file_path}
                title="Download log file"
              >
                Download Log
              </button>
            </div>
          </div>
          <pre ref={consoleRef} className={jobCardStyles.consoleOutput}>
            {logs.length === 0 ? (
              <span className={jobCardStyles.consoleEmpty}>
                {job.status === "running" ? "Waiting for logs..." : "No logs available"}
              </span>
            ) : (
              logs.join("\n")
            )}
          </pre>
        </div>
      )}
    </div>
  );
}
