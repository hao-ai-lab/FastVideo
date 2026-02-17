'use client';

import { getJobDetails } from "@/lib/api";
import { Job } from "@/lib/types";
import Link from "next/link";
import { useParams } from "next/navigation";
import { useEffect, useState } from "react";

function getBadgeClass(status: string) {
  switch (status) {
    case "pending":
      return "badge-pending";
    case "running":
      return "badge-running";
    case "completed":
      return "badge-completed";
    case "failed":
      return "badge-failed";
    case "stopped":
      return "badge-stopped";
    default:
      return "badge-pending";
  }
}

export default function JobDetailsPage() {
  const { id } = useParams<{ id: string }>();
  const [job, setJob] = useState<Job | null>(null);

  useEffect(() => {
    if (id) {
      getJobDetails(id as string).then(setJob);
    }
  }, [id]);

  if (!job) {
    return (
      <main>
        <section className="card">
          <p className="placeholder">Loading…</p>
        </section>
      </main>
    );
  }

  const progressPercent = typeof job.progress === "number" ? job.progress : 0;
  const isCompleted = job.status === "completed";

  return (
    <main>
      <section className="card">
        <div className="section-header">
          <h2>Job Details</h2>
          <Link href="/" className="btn btn-small">
            ← Back
          </Link>
        </div>

        <div className="job-card">
          <div className="job-header">
            <span className="job-model">{job.model_id}</span>
            <span className={`badge badge-${job.status}`}>
              {job.status}
            </span>
          </div>
          <p className="job-prompt" style={{ whiteSpace: "normal" }}>{job.prompt}</p>
          <div className="job-meta">
            <span>{job.num_frames} frames</span>
            <span>{job.height}×{job.width}</span>
          </div>

          {(job.status === "running" || job.status === "completed") && (
            <div className="progress-container">
              <div className="progress-bar-bg">
                <div
                  className={`progress-bar-fill ${isCompleted ? "completed" : ""}`}
                  style={{ width: `${progressPercent}%` }}
                />
              </div>
              <span className="progress-label">
                {job.progress_msg || `${Math.round(progressPercent)}%`}
              </span>
            </div>
          )}

          {job.error && (
            <div className="job-error">{job.error}</div>
          )}
        </div>
      </section>
    </main>
  );
}
