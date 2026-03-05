"use client";

import React from "react";
import type { TimestampedEvent } from "@/lib/stores/agent-progress-store";
import type { Source, AgentEvent } from "@/lib/types";
import { TaskProgressPanel } from "@/components/ui/task-progress-panel";

// Re-export the existing TaskProgressPanel as SimpleTimeline for backward compat.
// In the unified progress panel, this is used for unplanned execution (no plan_overview).

type ProgressEvent = TimestampedEvent | AgentEvent;

interface SimpleTimelineProps {
  events: ProgressEvent[];
  sources?: Source[];
  isStreaming?: boolean;
  agentType?: string;
  className?: string;
}

/**
 * Simple timeline for unplanned execution — wraps the existing TaskProgressPanel.
 * Used when no plan_overview event was received (simple tasks).
 */
export function SimpleTimeline({
  events,
  sources = [],
  isStreaming = false,
  agentType,
  className,
}: SimpleTimelineProps) {
  return (
    <TaskProgressPanel
      events={events}
      sources={sources}
      isStreaming={isStreaming}
      agentType={agentType}
      className={className}
    />
  );
}
