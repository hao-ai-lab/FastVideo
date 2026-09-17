import { getApiBaseUrl } from '@/lib/api';
import type { Job } from '@/lib/types';

/** Keep each results view to at most 50 lightweight poster images. */
export const MAX_RESULT_PREVIEWS = 50;

export function hasJobResult(job: Job): boolean {
  return (
    job.status === 'completed' &&
    Boolean(job.output_path) &&
    (job.job_type === 'inference' || !job.job_type)
  );
}

export function isJobImage(job: Job): boolean {
  return /\.(png|jpe?g|webp)$/i.test(job.output_path ?? '');
}

export function getJobThumbnailUrl(jobId: string): string {
  return `${getApiBaseUrl()}/jobs/${encodeURIComponent(jobId)}/thumbnail`;
}

export function filterJobs(jobs: Job[], model: string, prompt: string): Job[] {
  const modelQuery = model.trim().toLowerCase();
  const promptQuery = prompt.trim().toLowerCase();
  return jobs.filter(
    (job) =>
      job.model_id.toLowerCase().includes(modelQuery) &&
      (job.prompt ?? '').toLowerCase().includes(promptQuery),
  );
}
