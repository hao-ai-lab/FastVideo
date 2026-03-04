'use client';

import { useCallback, useEffect, useRef, useState } from "react";
import primarySidebarStyles from "./styles/PrimarySidebar.module.css";

const SIDEBAR_MIN_WIDTH = 100;
const SIDEBAR_MAX_WIDTH = 300;
const SIDEBAR_COLLAPSED_WIDTH = 0;
const SIDEBAR_COLLAPSED_VISIBLE_WIDTH = 60; /* collapse button width */

export type SidebarTab =
  | "inference"
  | "finetuning"
  | "distillation"
  | "lora"
  | "datasets"
  | "settings";

interface PrimarySidebarProps {
  activeTab: SidebarTab;
  onTabChange: (tab: SidebarTab) => void;
  onWidthChange?: (width: number) => void;
}

function CollapseIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M15 18l-6-6 6-6" />
    </svg>
  );
}

function ExpandIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M9 18l6-6-6-6" />
    </svg>
  );
}

export default function PrimarySidebar({
  activeTab,
  onTabChange,
  onWidthChange,
}: PrimarySidebarProps) {
  const [width, setWidth] = useState(220);
  const [isCollapsed, setIsCollapsed] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const dragStartRef = useRef({ x: 0, width: 0 });

  const effectiveWidth = isCollapsed ? SIDEBAR_COLLAPSED_WIDTH : width;
  const layoutWidth = isCollapsed ? SIDEBAR_COLLAPSED_VISIBLE_WIDTH : width;

  useEffect(() => {
    onWidthChange?.(layoutWidth);
  }, [layoutWidth, onWidthChange]);

  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      e.preventDefault();
      dragStartRef.current = { x: e.clientX, width };
      setIsDragging(true);
    },
    [width]
  );

  useEffect(() => {
    if (!isDragging) return;

    const handleMouseMove = (e: MouseEvent) => {
      const { x, width: startWidth } = dragStartRef.current;
      const delta = e.clientX - x;
      const newWidth = Math.min(
        SIDEBAR_MAX_WIDTH,
        Math.max(SIDEBAR_MIN_WIDTH, startWidth + delta)
      );
      setWidth(newWidth);
    };

    const handleMouseUp = () => {
      setIsDragging(false);
    };

    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseup", handleMouseUp);

    return () => {
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
    };
  }, [isDragging]);

  const toggleCollapse = useCallback(() => {
    setIsCollapsed((prev) => !prev);
  }, []);

  return (
    <aside
      className={`${primarySidebarStyles.sidebar} ${
        isCollapsed ? primarySidebarStyles.collapsed : ""
      }`}
      style={{ width: effectiveWidth }}
    >
      <nav className={primarySidebarStyles.tabs}>
        <button
          type="button"
          className={`${primarySidebarStyles.tab} ${
            activeTab === "inference" ? primarySidebarStyles.tabActive : ""
          }`}
          onClick={() => onTabChange("inference")}
        >
          Inference
        </button>
        <button
          type="button"
          className={`${primarySidebarStyles.tab} ${
            activeTab === "finetuning" ? primarySidebarStyles.tabActive : ""
          }`}
          onClick={() => onTabChange("finetuning")}
        >
          Finetuning
        </button>
        <button
          type="button"
          className={`${primarySidebarStyles.tab} ${
            activeTab === "distillation" ? primarySidebarStyles.tabActive : ""
          }`}
          onClick={() => onTabChange("distillation")}
        >
          Distillation
        </button>
        <button
          type="button"
          className={`${primarySidebarStyles.tab} ${
            activeTab === "lora" ? primarySidebarStyles.tabActive : ""
          }`}
          onClick={() => onTabChange("lora")}
        >
          LoRA
        </button>
        <button
          type="button"
          className={`${primarySidebarStyles.tab} ${
            activeTab === "datasets" ? primarySidebarStyles.tabActive : ""
          }`}
          onClick={() => onTabChange("datasets")}
        >
          Datasets
        </button>
        <button
          type="button"
          className={`${primarySidebarStyles.tab} ${
            activeTab === "settings" ? primarySidebarStyles.tabActive : ""
          }`}
          onClick={() => onTabChange("settings")}
        >
          Settings
        </button>
      </nav>
      <div className={primarySidebarStyles.collapseFooter}>
        <button
          type="button"
          className={primarySidebarStyles.collapseBtn}
          onClick={toggleCollapse}
          title={isCollapsed ? "Expand sidebar" : "Collapse sidebar"}
        >
          {isCollapsed ? <ExpandIcon /> : <CollapseIcon />}
        </button>
      </div>
      {!isCollapsed && (
        <div
          className={`${primarySidebarStyles.resizeHandle} ${
            isDragging ? primarySidebarStyles.resizeHandleActive : ""
          }`}
          onMouseDown={handleMouseDown}
        />
      )}
    </aside>
  );
}
