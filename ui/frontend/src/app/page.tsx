import { getJobsList } from "@/lib/api";
import { Job } from "@/lib/types";
import Link from "next/link";
import CreateJobButton from "@/components/CreateJobButton";
import cardStyles from "@/components/Card.module.css";
import jobCardStyles from "@/components/JobCard.module.css";
import badgeStyles from "@/components/Badge.module.css";
import layoutStyles from "./Layout.module.css";

function getBadgeClass(status: string) {
  switch (status) {
    case "pending":
      return badgeStyles.badgePending;
    case "running":
      return badgeStyles.badgeRunning;
    case "completed":
      return badgeStyles.badgeCompleted;
    case "failed":
      return badgeStyles.badgeFailed;
    case "stopped":
      return badgeStyles.badgeStopped;
    default:
      return badgeStyles.badgePending;
  }
}

export default async function Home() {
  const jobs = await getJobsList();

  return (
    <main className={layoutStyles.main}>
      <section className={cardStyles.card}>
        <div className={layoutStyles.sectionHeader}>
          <h2>Jobs</h2>
          <CreateJobButton />
        </div>
        <div id="jobs-container">
          {jobs.length === 0 ? (
            <p className={layoutStyles.placeholder}>No jobs yet. Create one above.</p>
          ) : (
            jobs.map((job) => (
              <Link href={`/jobs/${job.id}`} key={job.id} className={jobCardStyles.jobCard} style={{ textDecoration: "none", color: "inherit" }}>
                <div className={jobCardStyles.jobHeader}>
                  <span className={jobCardStyles.jobModel}>{job.model_id}</span>
                  <span className={`${badgeStyles.badge} ${getBadgeClass(job.status)}`}>
                    {job.status}
                  </span>
                </div>
                <p className={jobCardStyles.jobPrompt}>{job.prompt}</p>
                <div className={jobCardStyles.jobMeta}>
                  <span>{job.num_frames} frames</span>
                  <span>{job.height}×{job.width}</span>
                </div>
              </Link>
            ))
          )}
        </div>
      </section>
    </main>
  );
}
