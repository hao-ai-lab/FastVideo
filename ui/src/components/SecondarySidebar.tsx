'use client';

import { Job } from "@/lib/types";
import { getJobLogs, downloadJobLog } from "@/lib/api";
import { useCallback, useEffect, useRef, useState } from "react";
import secondaryStyles from "./styles/SecondarySidebar.module.css";
import buttonStyles from "@styles/Button.module.css";

const SIDEBAR_MIN_WIDTH = 280;
const SIDEBAR_MAX_WIDTH = 750;

interface SecondarySidebarProps {
  job: Job;
  onClose: () => void;
  onWidthChange?: (width: number) => void;
}

function CloseIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M18 6L6 18M6 6l12 12" />
    </svg>
  );
}

export default function SecondarySidebar({
  job,
  onClose,
  onWidthChange,
}: SecondarySidebarProps) {
  const [width, setWidth] = useState(360);

  useEffect(() => {
    onWidthChange?.(width);
  }, [width, onWidthChange]);
  const [isDragging, setIsDragging] = useState(false);
  const dragStartRef = useRef({ x: 0, width: 0 });
  const [isLoading, setIsLoading] = useState(false);
  const [logs, setLogs] = useState<string[]>([]);
  const logAfterRef = useRef(0);
  const isPollingRef = useRef(false);
  const consoleRef = useRef<HTMLPreElement>(null);
  const previousJobIdRef = useRef<string | null>(null);
  const previousStatusRef = useRef<string | null>(null);

  // Poll for logs when sidebar is open
  useEffect(() => {
    const shouldPoll = job.status === "running" || job.status === "pending";
    let pollInterval: NodeJS.Timeout | null = null;
    let isMounted = true;

    const pollLogs = async () => {
      if (!isMounted || isPollingRef.current) return;
      isPollingRef.current = true;
      try {
        const logData = await getJobLogs(job.id, logAfterRef.current);
        if (isMounted && logData.lines.length > 0) {
          setLogs((prev) => [...prev, ...logData.lines]);
          logAfterRef.current = logData.total;
          if (consoleRef.current) {
            consoleRef.current.scrollTop = consoleRef.current.scrollHeight;
          }
        }
      } catch (error) {
        console.error("Failed to fetch logs:", error);
      } finally {
        isPollingRef.current = false;
      }
    };

    pollLogs();

    if (shouldPoll) {
      pollInterval = setInterval(pollLogs, 2000);
    }

    return () => {
      isMounted = false;
      isPollingRef.current = false;  // Allow re-run after Strict Mode cleanup
      if (pollInterval) clearInterval(pollInterval);
    };
  }, [job.id, job.status]);

  // Reset logs when job changes or job is restarted
  useEffect(() => {
    const previousJobId = previousJobIdRef.current;
    const previousStatus = previousStatusRef.current;
    const currentJobId = job.id;
    const currentStatus = job.status;

    const wasTerminal =
      previousStatus === "failed" ||
      previousStatus === "stopped" ||
      previousStatus === "completed";
    const isRestarting =
      previousJobId === currentJobId &&
      wasTerminal &&
      (currentStatus === "pending" || currentStatus === "running");

    if (previousJobId !== currentJobId || isRestarting) {
      setLogs([]);
      logAfterRef.current = 0;
    }

    previousJobIdRef.current = currentJobId;
    previousStatusRef.current = currentStatus;
  }, [job.id, job.status]);

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
      // Right sidebar: drag left (negative delta) = wider
      const newWidth = Math.min(
        SIDEBAR_MAX_WIDTH,
        Math.max(SIDEBAR_MIN_WIDTH, startWidth - delta)
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

  const handleDownloadLog = async (e: React.MouseEvent) => {
    e.preventDefault();
    if (isLoading) return;

    setIsLoading(true);
    try {
      const blob = await downloadJobLog(job.id);
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `job_${job.id}.log`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (error) {
      console.error("Failed to download log:", error);
      alert(error instanceof Error ? error.message : "Failed to download log");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <aside
      className={secondaryStyles.sidebar}
      style={{ width, maxWidth: SIDEBAR_MAX_WIDTH }}
    >
      <div className={secondaryStyles.header}>
        <h2 className={secondaryStyles.title}>Job Details</h2>
        <div className={secondaryStyles.headerActions}>
          <button
            className={`${buttonStyles.btn} ${buttonStyles.btnSmall}`}
            onClick={handleDownloadLog}
            disabled={isLoading || !job.log_file_path}
            title="Download log file"
          >
            Download Log
          </button>
          <button
            type="button"
            className={secondaryStyles.closeBtn}
            onClick={onClose}
            title="Close"
          >
            <CloseIcon />
          </button>
        </div>
      </div>
      <div className={secondaryStyles.consoleSection}>
        <div className={secondaryStyles.consoleHeader}>
          <span className={secondaryStyles.consoleTitle}>Console Output</span>
          {job.status === "running" && (
            <span className={secondaryStyles.consoleStatus}>● Live</span>
          )}
        </div>
        <pre ref={consoleRef} className={secondaryStyles.consoleOutput}>
          {logs.length === 0 ? (
            <span className={secondaryStyles.consoleEmpty}>
              {job.status === "running"
                ? "Waiting for logs..."
                : "No logs available"}
            </span>
          ) : (
            logs.join("\n")
          )}
        </pre>
      </div>
      <div
        className={`${secondaryStyles.resizeHandle} ${
          isDragging ? secondaryStyles.resizeHandleActive : ""
        }`}
        onMouseDown={handleMouseDown}
      />
    </aside>
  );
}
