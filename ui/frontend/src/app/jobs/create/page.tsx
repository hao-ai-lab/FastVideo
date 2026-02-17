'use client';

import { createJob } from "@/lib/api";
import { useRouter } from "next/navigation";
import { useState } from "react";
import cardStyles from "@/components/Card.module.css";
import formStyles from "@/components/Form.module.css";
import buttonStyles from "@/components/Button.module.css";
import layoutStyles from "../../Layout.module.css";

export default function CreateJobPage() {
  const router = useRouter();
  const [modelId, setModelId] = useState("");
  const [prompt, setPrompt] = useState("");

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    await createJob({ model_id: modelId, prompt });
    router.push("/");
  };

  return (
    <main className={layoutStyles.main}>
      <section className={cardStyles.card}>
        <h2>New Job</h2>
        <form onSubmit={handleSubmit} autoComplete="off">
          <div className={formStyles.formRow}>
            <label htmlFor="modelId">Model ID</label>
            <input
              type="text"
              name="modelId"
              id="modelId"
              value={modelId}
              onChange={(e) => setModelId(e.target.value)}
              placeholder="e.g. Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
              required
            />
          </div>
          <div className={formStyles.formRow}>
            <label htmlFor="prompt">Prompt</label>
            <textarea
              name="prompt"
              id="prompt"
              rows={3}
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="A curious raccoon peers through a vibrant field of yellow sunflowers…"
              required
            />
          </div>
          <button type="submit" className={`${buttonStyles.btn} ${buttonStyles.btnPrimary}`}>
            Create Job
          </button>
        </form>
      </section>
    </main>
  );
}
