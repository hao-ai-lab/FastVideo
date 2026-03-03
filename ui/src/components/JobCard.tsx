'use client';

import { Job } from "@/lib/types";
import {
  startJob,
  stopJob,
  deleteJob,
  downloadJobVideo,
} from "@/lib/api";
import { useActiveJob } from "@/contexts/ActiveJobContext";
import jobCardStyles from "@styles/JobCard.module.css";
import badgeStyles from "@styles/Badge.module.css";
import buttonStyles from "@styles/Button.module.css";
import { useState, useEffect, useRef } from "react";

interface JobCardProps {
  job: Job;
  onJobUpdated?: () => void;
}

export default function JobCard({ job, onJobUpdated }: JobCardProps) {
  const { activeJobId, setActiveJobId } = useActiveJob();
  const [isLoading, setIsLoading] = useState(false);
  const [currentTime, setCurrentTime] = useState(Date.now());
  const badgeClass = `badge${job.status.charAt(0).toUpperCase() + job.status.slice(1)}`;
  const isSelected = activeJobId === job.id;
  
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

  const handleSelectJob = (e: React.MouseEvent) => {
    if ((e.target as HTMLElement).closest("button")) return;
    setActiveJobId(isSelected ? null : job.id);
  };

  const handleDownloadVideo = async (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (isLoading || !job.output_path) return;

    setIsLoading(true);
    try {
      const blob = await downloadJobVideo(job.id);
      const ext = job.output_path.endsWith(".png") ? "png" : "mp4";
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `job_${job.id}.${ext}`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (error) {
      console.error("Failed to download video:", error);
      alert(error instanceof Error ? error.message : "Failed to download video");
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
    <div
      className={`${jobCardStyles.jobCard} ${isSelected ? jobCardStyles.jobCardSelected : ""}`}
      onClick={handleSelectJob}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          handleSelectJob(e as unknown as React.MouseEvent);
        }
      }}
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
      <div className={jobCardStyles.jobActions}>
        {getActionButton()}
        {job.status === "completed" && job.output_path && (
          <button
            className={`${buttonStyles.btn} ${buttonStyles.btnSmall}`}
            onClick={handleDownloadVideo}
            disabled={isLoading}
            title="Download video"
          >
            Download Video
          </button>
        )}
        <button
          className={`${buttonStyles.btn} ${buttonStyles.btnDelete} ${buttonStyles.btnSmall}`}
          onClick={handleDelete}
          disabled={isLoading}
        >
          Delete
        </button>
      </div>
    </div>
  );
}
