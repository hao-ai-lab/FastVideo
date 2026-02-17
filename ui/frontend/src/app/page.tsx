import { getJobsList } from "@/lib/api";
import { Job } from "@/lib/types";
import Link from "next/link";

export default async function Home() {
  const jobs = await getJobsList();

  return (
    <main>
      <section className="card">
        <div className="section-header">
          <h2>Jobs</h2>
          <Link href="/jobs/create" className="btn btn-primary">
            Create Job
          </Link>
        </div>
        <div id="jobs-container">
          {jobs.length === 0 ? (
            <p className="placeholder">No jobs yet. Create one above.</p>
          ) : (
            jobs.map((job) => (
              <Link href={`/jobs/${job.id}`} key={job.id} className="job-card" style={{ textDecoration: "none", color: "inherit" }}>
                <div className="job-header">
                  <span className="job-model">{job.model_id}</span>
                  <span className={`badge badge-${job.status}`}>
                    {job.status}
                  </span>
                </div>
                <p className="job-prompt">{job.prompt}</p>
                <div className="job-meta">
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
