'use client';

import { useState, useEffect } from "react";
import PrimarySidebar from "@/components/PrimarySidebar";
import SecondarySidebar from "@/components/SecondarySidebar";
import DatasetSidebar from "@/components/DatasetSidebar";
import JobQueuePage from "@/components/JobQueuePage";
import DatasetsPage from "@/components/DatasetsPage";
import SettingsPage from "@/components/SettingsPage";
import {
  useActiveTab,
  type ActiveTab,
} from "@/contexts/ActiveTabContext";
import { useActiveJob } from "@/contexts/ActiveJobContext";
import { useActiveDataset } from "@/contexts/ActiveDatasetContext";
import primarySidebarStyles from "@/components/styles/PrimarySidebar.module.css";

export default function Home() {
  const { activeTab, setActiveTab } = useActiveTab();
  const { activeJob, setActiveJobId } = useActiveJob();
  const { activeDataset, setActiveDatasetId } = useActiveDataset();

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape" && !document.querySelector("[data-modal]")) {
        if (activeJob) setActiveJobId(null);
        if (activeDataset) setActiveDatasetId(null);
      }
    };
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [activeJob, activeDataset, setActiveJobId, setActiveDatasetId]);

  const [primaryWidth, setPrimaryWidth] = useState(220);
  const [secondaryWidth, setSecondaryWidth] = useState(0);

  const jobTabs: ActiveTab[] = [
    "inference",
    "finetuning",
    "distillation",
    "lora",
  ];
  const jobSidebarOpen = jobTabs.includes(activeTab) && !!activeJob;
  const datasetSidebarOpen = activeTab === "datasets" && !!activeDataset;

  const secondaryOpen = jobSidebarOpen || datasetSidebarOpen;

  return (
    <div className={primarySidebarStyles.layout}>
      <PrimarySidebar
        activeTab={activeTab}
        onTabChange={setActiveTab}
        onWidthChange={setPrimaryWidth}
      />
      <div
        className={primarySidebarStyles.content}
        style={{
          marginLeft: primaryWidth,
          marginRight: secondaryOpen ? secondaryWidth : 0,
        }}
      >
        {activeTab === "inference" && <JobQueuePage jobType="inference" />}
        {activeTab === "finetuning" && <JobQueuePage jobType="finetuning" />}
        {activeTab === "distillation" && <JobQueuePage jobType="distillation" />}
        {activeTab === "lora" && <JobQueuePage jobType="lora" />}
        {activeTab === "datasets" && <DatasetsPage />}
        {activeTab === "settings" && <SettingsPage />}
      </div>
      {jobSidebarOpen && activeJob && (
        <SecondarySidebar
          job={activeJob}
          onClose={() => setActiveJobId(null)}
          onWidthChange={setSecondaryWidth}
        />
      )}
      {datasetSidebarOpen && activeDataset && (
        <DatasetSidebar
          dataset={activeDataset}
          onClose={() => setActiveDatasetId(null)}
          onWidthChange={setSecondaryWidth}
          onUpdated={() => {}}
        />
      )}
    </div>
  );
}
