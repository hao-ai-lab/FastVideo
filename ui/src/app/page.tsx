'use client';

import { useState, useEffect } from "react";
import PrimarySidebar, { type SidebarTab } from "@/components/PrimarySidebar";
import SecondarySidebar from "@/components/SecondarySidebar";
import JobQueuePage from "@/components/JobQueuePage";
import SettingsPage from "@/components/SettingsPage";
import {
  useActiveTab,
  type ActiveTab,
} from "@/contexts/ActiveTabContext";
import { useActiveJob } from "@/contexts/ActiveJobContext";
import primarySidebarStyles from "@/components/styles/PrimarySidebar.module.css";

export default function Home() {
  const { activeTab, setActiveTab } = useActiveTab();
  const { activeJob, setActiveJobId } = useActiveJob();

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape" && activeJob && !document.querySelector("[data-modal]")) {
        setActiveJobId(null);
      }
    };
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [activeJob, setActiveJobId]);
  const [primaryWidth, setPrimaryWidth] = useState(220);
  const [secondaryWidth, setSecondaryWidth] = useState(0);

  const jobTabs: ActiveTab[] = [
    "inference",
    "finetuning",
    "distillation",
    "lora",
  ];
  const secondaryOpen =
    jobTabs.includes(activeTab) && !!activeJob;

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
        {activeTab === "settings" && <SettingsPage />}
      </div>
      {secondaryOpen && activeJob && (
        <SecondarySidebar
          job={activeJob}
          onClose={() => setActiveJobId(null)}
          onWidthChange={setSecondaryWidth}
        />
      )}
    </div>
  );
}
