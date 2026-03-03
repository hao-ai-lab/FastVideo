'use client';

import Sidebar, { type SidebarTab } from "@/components/Sidebar";
import SecondarySidebar from "@/components/SecondarySidebar";
import JobQueuePage from "@/components/JobQueuePage";
import SettingsPage from "@/components/SettingsPage";
import { useActiveTab } from "@/contexts/ActiveTabContext";
import { useActiveJob } from "@/contexts/ActiveJobContext";
import sidebarStyles from "@/components/styles/Sidebar.module.css";

export default function Home() {
  const { activeTab, setActiveTab } = useActiveTab();
  const { activeJob, setActiveJobId } = useActiveJob();

  return (
    <div className={sidebarStyles.layout}>
      <Sidebar activeTab={activeTab} onTabChange={setActiveTab} />
      <div className={sidebarStyles.content}>
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
