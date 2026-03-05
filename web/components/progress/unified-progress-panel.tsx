"use client";

import React, { useState, useMemo } from "react";
import { ChevronDown } from "lucide-react";
import { useTranslations } from "next-intl";
import { cn } from "@/lib/utils";
import {
  useExecutionProgressStore,
  type PlanStep,
  type ExecutionProgress,
  type ToolCallRecord,
} from "@/lib/stores/execution-progress-store";
import type { TimestampedEvent } from "@/lib/stores/agent-progress-store";
import type { Source, AgentEvent } from "@/lib/types";
import { ProgressHeader } from "./progress-header";
import { PlanStepList } from "./plan-step-list";
import { SimpleTimeline } from "./simple-timeline";

type ProgressEvent = TimestampedEvent | AgentEvent;

interface UnifiedProgressPanelProps {
  isStreaming: boolean;
  events: ProgressEvent[];
  sources?: Source[];
  agentType?: string;
  className?: string;
}

/**
 * Reconstruct plan state from historical events.
 * Used when rendering saved messages that contain plan_overview + step_activity events.
 */
function reconstructPlanFromEvents(events: ProgressEvent[]): ExecutionProgress | null {
  let plan: PlanStep[] | null = null;
  let activeStepIndex: number | null = null;

  for (const event of events) {
    if (event.type === "plan_overview") {
      const planEvent = event as unknown as {
        steps?: Array<{
          id: number;
          title: string;
          description: string;
          status: string;
          estimated_complexity?: string;
          depends_on?: number[];
        }>;
        total_steps?: number;
      };

      if (!planEvent.steps || !planEvent.total_steps) continue;

      plan = planEvent.steps.map((s) => ({
        id: s.id,
        title: s.title,
        description: s.description || "",
        status: "completed" as const,
        estimatedComplexity: (s.estimated_complexity as PlanStep["estimatedComplexity"]) || undefined,
        dependsOn: s.depends_on || [],
        toolCalls: [],
        stages: [],
      }));
    }

    if (event.type === "step_activity" && plan) {
      const stepIdx = event.step_index;
      if (stepIdx === undefined || stepIdx < 0 || stepIdx >= plan.length) continue;

      const status = event.status as string;
      if (status === "running") {
        activeStepIndex = stepIdx;
        plan[stepIdx] = { ...plan[stepIdx], status: "completed" };
      } else if (status === "completed" || status === "failed") {
        plan[stepIdx] = {
          ...plan[stepIdx],
          status: "completed",
          resultSummary: event.result_summary || plan[stepIdx].resultSummary,
          durationMs: event.duration_ms || plan[stepIdx].durationMs,
        };
      }
    }

    // Attach tool calls to the currently active step
    if (event.type === "tool_call" && plan && activeStepIndex !== null) {
      if (activeStepIndex >= 0 && activeStepIndex < plan.length) {
        const tc: ToolCallRecord = {
          id: event.id || `tc-hist-${plan[activeStepIndex].toolCalls.length}`,
          tool: event.tool || event.name || "unknown",
          args: event.args,
          status: "completed",
          startedAt: (event as TimestampedEvent).timestamp || Date.now(),
          completedAt: (event as TimestampedEvent).endTimestamp,
        };
        plan[activeStepIndex] = {
          ...plan[activeStepIndex],
          toolCalls: [...plan[activeStepIndex].toolCalls, tc],
        };
      }
    }
  }

  if (!plan || plan.length === 0) return null;

  const completedCount = plan.filter((s) => s.status === "completed").length;

  return {
    plan,
    activeStepIndex: null,
    completedSteps: completedCount,
    totalSteps: plan.length,
    isStreaming: false,
    startedAt: null,
    completedAt: null,
  };
}

/**
 * Unified progress panel that shows either:
 * - Plan-centric view with steps and nested tools (when plan exists)
 * - Simple timeline view (when no plan — same as current TaskProgressPanel)
 */
export function UnifiedProgressPanel({
  isStreaming,
  events,
  sources = [],
  agentType,
  className,
}: UnifiedProgressPanelProps) {
  const liveProgress = useExecutionProgressStore((state) => state.progress);

  // For historical (non-streaming), reconstruct from events
  const historicalProgress = useMemo(() => {
    if (isStreaming) return null;
    return reconstructPlanFromEvents(events);
  }, [isStreaming, events]);

  const progress = isStreaming ? liveProgress : historicalProgress;
  const hasPlan = progress?.plan !== null && (progress?.plan?.length ?? 0) > 0;

  // For planned execution, show plan-centric view
  if (hasPlan && progress?.plan) {
    if (isStreaming) {
      // Live streaming — inline
      return (
        <div className={cn("mt-2 mb-4", className)}>
          <ProgressHeader progress={progress} />
          <div className="mt-3">
            <PlanStepList steps={progress.plan} />
          </div>
        </div>
      );
    }

    // Historical — collapsible
    return (
      <HistoricalPlanView
        progress={progress}
        className={className}
      />
    );
  }

  // No plan — fallback to simple timeline
  return (
    <SimpleTimeline
      events={events}
      sources={sources}
      isStreaming={isStreaming}
      agentType={agentType}
      className={className}
    />
  );
}

/**
 * Collapsible historical plan view for completed executions.
 */
function HistoricalPlanView({
  progress,
  className,
}: {
  progress: ExecutionProgress;
  className?: string;
}) {
  const [isExpanded, setIsExpanded] = useState(false);
  const t = useTranslations("progress");

  const summaryText = useMemo(() => {
    const parts: string[] = [];
    if (progress.completedSteps > 0) {
      parts.push(
        t("completedStepsOf", {
          completed: progress.completedSteps,
          total: progress.totalSteps,
        })
      );
    }
    const totalTools = (progress.plan || []).reduce(
      (sum, step) => sum + step.toolCalls.length,
      0
    );
    if (totalTools > 0) {
      parts.push(t("toolsUsed", { count: totalTools }));
    }
    return parts.join(" · ") || t("completed");
  }, [progress, t]);

  return (
    <div className={cn("mt-3", className)}>
      <button
        className="flex items-center gap-2 w-full text-left group hover:opacity-80 transition-opacity py-1.5"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <ChevronDown
          className={cn(
            "w-3.5 h-3.5 text-muted-foreground/40 flex-shrink-0 transition-transform duration-200",
            !isExpanded && "-rotate-90"
          )}
        />
        <span className="text-xs text-muted-foreground/60 font-medium">
          {summaryText}
        </span>
      </button>

      <div className={cn("accordion-grid", isExpanded && "accordion-open")}>
        <div className="accordion-inner">
          <div className="pl-1 pt-1 pb-2">
            <ProgressHeader progress={progress} />
            {progress.plan && (
              <div className="mt-2">
                <PlanStepList steps={progress.plan} />
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
