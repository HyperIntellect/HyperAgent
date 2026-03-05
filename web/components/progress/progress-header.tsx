"use client";

import React, { useMemo } from "react";
import { useTranslations } from "next-intl";
import { cn } from "@/lib/utils";
import type { ExecutionProgress } from "@/lib/stores/execution-progress-store";
import { LiveTimer, formatDuration } from "./live-timer";

interface ProgressHeaderProps {
  progress: ExecutionProgress;
  compact?: boolean;
  className?: string;
}

// Complexity weight multipliers for ETA calculation
const COMPLEXITY_WEIGHT: Record<string, number> = {
  low: 0.5,
  medium: 1.0,
  high: 2.0,
};

/**
 * Progress summary bar showing step count, current step title, elapsed time, and ETA.
 * - Planned: "Step 3/7 — Installing dependencies — 2:34 elapsed — ~5m remaining"
 * - Unplanned: "Processing... — 3 stages — 45s"
 */
export function ProgressHeader({
  progress,
  compact = false,
  className,
}: ProgressHeaderProps) {
  const t = useTranslations("progress");

  const { plan, activeStepIndex, completedSteps, totalSteps, isStreaming, startedAt } = progress;

  // Calculate ETA based on average step duration weighted by complexity
  const eta = useMemo(() => {
    if (!plan || completedSteps === 0 || completedSteps >= totalSteps) return null;

    const completedMs = plan
      .filter((s) => s.status === "completed" && s.durationMs)
      .reduce((sum, s) => {
        const weight = COMPLEXITY_WEIGHT[s.estimatedComplexity || "medium"] || 1;
        return sum + (s.durationMs || 0) / weight;
      }, 0);

    const completedWeightedCount = plan
      .filter((s) => s.status === "completed")
      .reduce((sum, s) => sum + (COMPLEXITY_WEIGHT[s.estimatedComplexity || "medium"] || 1), 0);

    if (completedWeightedCount === 0) return null;

    const avgMsPerWeight = completedMs / completedWeightedCount;

    const remainingWeight = plan
      .filter((s) => s.status !== "completed" && s.status !== "failed")
      .reduce((sum, s) => sum + (COMPLEXITY_WEIGHT[s.estimatedComplexity || "medium"] || 1), 0);

    return Math.round(avgMsPerWeight * remainingWeight);
  }, [plan, completedSteps, totalSteps]);

  // Current step title
  const currentStepTitle = useMemo(() => {
    if (!plan || activeStepIndex === null || activeStepIndex >= plan.length) return null;
    return plan[activeStepIndex].title;
  }, [plan, activeStepIndex]);

  // Progress percentage
  const progressPct = totalSteps > 0 ? (completedSteps / totalSteps) * 100 : 0;

  if (!plan) {
    // Unplanned execution — minimal status
    if (!isStreaming && !startedAt) return null;
    return (
      <div className={cn("flex items-center gap-2", className)}>
        {isStreaming && (
          <span className="w-1.5 h-1.5 rounded-full bg-info animate-pulse shrink-0" />
        )}
        <span className="text-xs text-muted-foreground/60 truncate">
          {t("processing")}
        </span>
        {startedAt && (
          <LiveTimer
            startMs={startedAt}
            endMs={!isStreaming ? progress.completedAt ?? undefined : undefined}
          />
        )}
      </div>
    );
  }

  // Planned execution
  return (
    <div className={cn("space-y-1.5", className)}>
      {/* Progress bar */}
      <div className="flex items-center gap-2">
        <div className="flex-1 h-1 bg-muted/50 rounded-full overflow-hidden">
          <div
            className={cn(
              "h-full rounded-full transition-all duration-500 ease-out",
              isStreaming ? "bg-primary/60" : "bg-primary/40"
            )}
            style={{ width: `${Math.max(progressPct, 2)}%` }}
          />
        </div>
        <span className="text-[11px] tabular-nums text-muted-foreground/50 shrink-0">
          {completedSteps}/{totalSteps}
        </span>
      </div>

      {/* Status line */}
      {!compact && (
        <div className="flex items-center gap-2 text-xs text-muted-foreground/60 min-w-0">
          {currentStepTitle && isStreaming && (
            <>
              <span className="truncate max-w-[60%]">
                {t("stepOf", {
                  current: (activeStepIndex ?? 0) + 1,
                  total: totalSteps,
                })}
                {" — "}
                {currentStepTitle}
              </span>
              <span className="text-muted-foreground/30">·</span>
            </>
          )}

          {startedAt && (
            <LiveTimer
              startMs={startedAt}
              endMs={!isStreaming ? progress.completedAt ?? undefined : undefined}
            />
          )}

          {eta && isStreaming && (
            <>
              <span className="text-muted-foreground/30">·</span>
              <span className="shrink-0">
                ~{formatDuration(eta)} {t("remaining")}
              </span>
            </>
          )}
        </div>
      )}
    </div>
  );
}
