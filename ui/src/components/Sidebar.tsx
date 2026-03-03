'use client';

import { useState } from "react";
import sidebarStyles from "./styles/Sidebar.module.css";

export type SidebarTab = "job-queue" | "settings";

interface SidebarProps {
  activeTab: SidebarTab;
  onTabChange: (tab: SidebarTab) => void;
}

export default function Sidebar({ activeTab, onTabChange }: SidebarProps) {
  return (
    <aside className={sidebarStyles.sidebar}>
      <nav className={sidebarStyles.tabs}>
        <button
          type="button"
          className={`${sidebarStyles.tab} ${
            activeTab === "job-queue" ? sidebarStyles.tabActive : ""
          }`}
          onClick={() => onTabChange("job-queue")}
        >
          Job Queue
        </button>
        <button
          type="button"
          className={`${sidebarStyles.tab} ${
            activeTab === "settings" ? sidebarStyles.tabActive : ""
          }`}
          onClick={() => onTabChange("settings")}
        >
          Settings
        </button>
      </nav>
    </aside>
  );
}
