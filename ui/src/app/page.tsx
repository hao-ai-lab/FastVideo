'use client';

import { useState } from "react";
import PrimarySidebar, { type SidebarTab } from "@/components/PrimarySidebar";
import SecondarySidebar from "@/components/SecondarySidebar";
import JobQueuePage from "@/components/JobQueuePage";
import SettingsPage from "@/components/SettingsPage";
import { useActiveTab } from "@/contexts/ActiveTabContext";
import { useActiveJob } from "@/contexts/ActiveJobContext";
import primarySidebarStyles from "@/components/styles/PrimarySidebar.module.css";

export default function Home() {
  const { activeTab, setActiveTab } = useActiveTab();
  const { activeJob, setActiveJobId } = useActiveJob();
  const [primaryWidth, setPrimaryWidth] = useState(220);
  const [secondaryWidth, setSecondaryWidth] = useState(0);

  const secondaryOpen = activeTab === "job-queue" && !!activeJob;

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
        {activeTab === "job-queue" && <JobQueuePage />}
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
