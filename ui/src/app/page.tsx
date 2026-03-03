'use client';

import Sidebar, { type SidebarTab } from "@/components/Sidebar";
import JobQueuePage from "@/components/JobQueuePage";
import SettingsPage from "@/components/SettingsPage";
import { useActiveTab } from "@/contexts/ActiveTabContext";
import sidebarStyles from "@/components/styles/Sidebar.module.css";

export default function Home() {
  const { activeTab, setActiveTab } = useActiveTab();

  return (
    <div className={sidebarStyles.layout}>
      <Sidebar activeTab={activeTab} onTabChange={setActiveTab} />
      <div className={sidebarStyles.content}>
        {activeTab === "job-queue" && <JobQueuePage />}
        {activeTab === "settings" && <SettingsPage />}
      </div>
    </div>
  );
}
