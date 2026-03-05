"use client";

import React, { useState, useEffect } from "react";

interface LiveTimerProps {
  startMs: number;
  endMs?: number;
  className?: string;
}

/**
 * Displays an auto-updating elapsed duration.
 * When endMs is provided, shows the final duration without updating.
 */
export function LiveTimer({ startMs, endMs, className }: LiveTimerProps) {
  const [now, setNow] = useState(() => Date.now());

  useEffect(() => {
    if (endMs) return;
    const interval = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(interval);
  }, [endMs]);

  const duration = endMs ? endMs - startMs : now - startMs;
  const seconds = duration / 1000;

  if (seconds < 1) return null;

  const formatted =
    seconds < 60
      ? `${Math.floor(seconds)}s`
      : `${Math.floor(seconds / 60)}:${Math.floor(seconds % 60)
          .toString()
          .padStart(2, "0")}`;

  return (
    <span
      className={
        className ||
        "tabular-nums text-xs font-medium text-muted-foreground/50"
      }
    >
      {formatted}
    </span>
  );
}

/**
 * Format a duration in milliseconds to a human-readable string.
 */
export function formatDuration(ms: number): string {
  const seconds = ms / 1000;
  if (seconds < 1) return "<1s";
  if (seconds < 60) return `${Math.floor(seconds)}s`;
  const minutes = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${minutes}:${secs.toString().padStart(2, "0")}`;
}
