"use client";

import React, { useCallback } from "react";
import { cn } from "@/lib/utils";
import type { PlanStep } from "@/lib/stores/execution-progress-store";
import { StatusIndicator } from "./status-indicator";
import { LiveTimer, formatDuration } from "./live-timer";
import { StepToolList } from "./step-tool-list";

interface PlanStepRowProps {
  step: PlanStep;
  index: number;
  isLast: boolean;
  isExpanded: boolean;
  onToggle: () => void;
}

/**
 * Individual plan step row with progressive disclosure.
 * Expansion state is managed by the parent PlanStepList for clean state management.
 */
export function PlanStepRow({ step, index, isLast, isExpanded, onToggle }: PlanStepRowProps) {
  const isRunning = step.status === "running";
  const isCompleted = step.status === "completed";

  const handleClick = useCallback(() => onToggle(), [onToggle]);

  const hasTools = step.toolCalls.length > 0;

  return (
    <div className="relative">
      {/* Vertical connector line */}
      {!isLast && (
        <div className="absolute left-[11px] top-[18px] bottom-0 w-px bg-border/40" />
      )}

      {/* Step header */}
      <button
        onClick={handleClick}
        className={cn(
          "flex items-center gap-2.5 w-full text-left py-1.5",
          "hover:opacity-80 transition-opacity",
          "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary rounded"
        )}
      >
        {/* Step number circle */}
        <div className="flex-shrink-0 relative z-10">
          {isRunning ? (
            <StatusIndicator status="running" size="md" />
          ) : (
            <div
              className={cn(
                "w-[22px] h-[22px] rounded-full flex items-center justify-center",
                "text-[10px] font-semibold",
                isCompleted && "border-2 border-primary/30 bg-primary/5 text-primary/60",
                step.status === "failed" && "border-2 border-destructive/30 bg-destructive/5 text-destructive/60",
                step.status === "pending" && "border-2 border-border bg-card text-muted-foreground/40"
              )}
            >
              {index + 1}
            </div>
          )}
        </div>

        {/* Title */}
        <span
          className={cn(
            "flex-1 text-sm leading-snug min-w-0 truncate",
            isRunning && "text-foreground font-medium",
            isCompleted && "text-muted-foreground/60",
            step.status === "failed" && "text-destructive/80",
            step.status === "pending" && "text-muted-foreground/30"
          )}
        >
          {step.title}
        </span>

        {/* Duration */}
        {step.durationMs ? (
          <span className="flex-shrink-0 tabular-nums text-xs font-medium text-muted-foreground/40">
            {formatDuration(step.durationMs)}
          </span>
        ) : step.startedAt && isRunning ? (
          <div className="flex-shrink-0">
            <LiveTimer startMs={step.startedAt} />
          </div>
        ) : null}
      </button>

      {/* Expanded content: tool calls */}
      {isExpanded && hasTools && (
        <div className="pl-[30px] pb-1">
          <StepToolList tools={step.toolCalls} />
        </div>
      )}

      {/* Result summary for completed steps */}
      {isExpanded && isCompleted && step.resultSummary && (
        <div className="pl-[30px] pb-1">
          <p className="text-xs text-muted-foreground/40 italic leading-relaxed truncate">
            {step.resultSummary}
          </p>
        </div>
      )}
    </div>
  );
}
