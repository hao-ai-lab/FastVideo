'use client';

import { getJobDetails } from "@/lib/api";
import { Job } from "@/lib/types";
import Link from "next/link";
import { useParams } from "next/navigation";
import { useEffect, useState } from "react";
import cardStyles from "@styles/Card.module.css";
import jobCardStyles from "@styles/JobCard.module.css";
import badgeStyles from "@styles/Badge.module.css";
import progressBarStyles from "@styles/ProgressBar.module.css";
import buttonStyles from "@styles/Button.module.css";
import layoutStyles from "../../Layout.module.css";

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
      <main className={layoutStyles.main}>
        <section className={cardStyles.card}>
          <p className={layoutStyles.placeholder}>Loading…</p>
        </section>
      </main>
    );
  }

  const progressPercent = typeof job.progress === "number" ? job.progress : 0;
  const isCompleted = job.status === "completed";
  const badgeClass = `badge${job.status.charAt(0).toUpperCase() + job.status.slice(1)}`;

  return (
    <main className={layoutStyles.main}>
      <section className={cardStyles.card}>
        <div className={layoutStyles.sectionHeader}>
          <h2>Job Details</h2>
          <Link href="/" className={`${buttonStyles.btn} ${buttonStyles.btnSmall}`}>
            ← Back
          </Link>
        </div>

        <div className={jobCardStyles.jobCard}>
          <div className={jobCardStyles.jobHeader}>
            <span className={jobCardStyles.jobModel}>{job.model_id}</span>
            <span className={`${badgeStyles.badge} ${badgeStyles[badgeClass as keyof typeof badgeStyles] || badgeStyles.badgePending}`}>
              {job.status}
            </span>
          </div>
          <p className={jobCardStyles.jobPrompt} style={{ whiteSpace: "normal" }}>{job.prompt}</p>
          <div className={jobCardStyles.jobMeta}>
            <span>{job.num_frames} frames</span>
            <span>{job.height}×{job.width}</span>
          </div>

          {(job.status === "running" || job.status === "completed") && (
            <div className={progressBarStyles.progressContainer}>
              <div className={progressBarStyles.progressBarBg}>
                <div
                  className={`${progressBarStyles.progressBarFill} ${isCompleted ? progressBarStyles.progressBarFillCompleted : ""}`}
                  style={{ width: `${progressPercent}%` }}
                />
              </div>
              <span className={progressBarStyles.progressLabel}>
                {job.progress_msg || `${Math.round(progressPercent)}%`}
              </span>
            </div>
          )}

          {job.error && (
            <div className={jobCardStyles.jobError}>{job.error}</div>
          )}
        </div>
      </section>
    </main>
  );
}
