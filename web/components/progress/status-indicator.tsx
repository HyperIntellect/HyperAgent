"use client";

import React from "react";
import { cn } from "@/lib/utils";

type Status = "pending" | "running" | "completed" | "failed";

interface StatusIndicatorProps {
  status: Status;
  size?: "sm" | "md";
}

/**
 * Compact status indicator — spinning ring for running, colored dot for others.
 */
export function StatusIndicator({ status, size = "sm" }: StatusIndicatorProps) {
  const outerSize = size === "sm" ? "w-2.5 h-2.5" : "w-3.5 h-3.5";
  const innerSize = size === "sm" ? "w-1.5 h-1.5" : "w-2 h-2";
  const spinnerBorder = size === "sm" ? "border-[1.5px]" : "border-2";

  if (status === "running") {
    return (
      <span className={cn("flex items-center justify-center", outerSize)}>
        <span
          className={cn(
            "rounded-full border-primary border-t-transparent animate-spin-slow",
            size === "sm" ? "w-2 h-2" : "w-2.5 h-2.5",
            spinnerBorder
          )}
        />
      </span>
    );
  }

  const dotColor = {
    completed: "bg-primary/40",
    failed: "bg-destructive/50",
    pending: "bg-muted-foreground/20",
  }[status];

  return (
    <span className={cn("flex items-center justify-center", outerSize)}>
      <span className={cn("rounded-full", innerSize, dotColor)} />
    </span>
  );
}
