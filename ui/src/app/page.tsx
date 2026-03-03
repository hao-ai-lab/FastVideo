'use client';

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

  return (
    <div className={primarySidebarStyles.layout}>
      <PrimarySidebar activeTab={activeTab} onTabChange={setActiveTab} />
      <div className={primarySidebarStyles.content}>
        {activeTab === "job-queue" && <JobQueuePage />}
        {activeTab === "settings" && <SettingsPage />}
      </div>
      {activeTab === "job-queue" && activeJob && (
        <SecondarySidebar
          job={activeJob}
          onClose={() => setActiveJobId(null)}
        />
      )}
    </div>
  );
}
