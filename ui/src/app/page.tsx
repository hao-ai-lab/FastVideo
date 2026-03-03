'use client';

import { useState } from "react";
import Sidebar, { type SidebarTab } from "@/components/Sidebar";
import JobQueuePage from "@/components/JobQueuePage";
import SettingsPage from "@/components/SettingsPage";
import sidebarStyles from "@/components/styles/Sidebar.module.css";

export default function Home() {
  const [activeTab, setActiveTab] = useState<SidebarTab>("job-queue");

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
