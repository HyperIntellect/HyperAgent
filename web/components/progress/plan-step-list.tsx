"use client";

import React, { useState, useCallback, useEffect } from "react";
import type { PlanStep } from "@/lib/stores/execution-progress-store";
import { PlanStepRow } from "./plan-step-row";

interface PlanStepListProps {
  steps: PlanStep[];
}

/**
 * Vertical list of plan steps with progressive disclosure.
 * Manages expanded state: running steps auto-expand, completed auto-collapse.
 */
export function PlanStepList({ steps }: PlanStepListProps) {
  // Track which steps are expanded by their id
  const [expandedSteps, setExpandedSteps] = useState<Set<number>>(() => {
    // Initialize: expand running steps
    const initial = new Set<number>();
    for (const step of steps) {
      if (step.status === "running") initial.add(step.id);
    }
    return initial;
  });

  // When steps change, auto-expand running steps and auto-collapse completed ones
  useEffect(() => {
    const timer = setTimeout(() => {
      setExpandedSteps((prev) => {
        const next = new Set(prev);
        for (const step of steps) {
          if (step.status === "running") {
            next.add(step.id);
          }
        }
        return next;
      });
    }, 0);
    return () => clearTimeout(timer);
  }, [steps]);

  // Auto-collapse completed steps after a delay
  useEffect(() => {
    const completedIds = steps
      .filter((s) => s.status === "completed")
      .map((s) => s.id);

    if (completedIds.length === 0) return;

    const timer = setTimeout(() => {
      setExpandedSteps((prev) => {
        const next = new Set(prev);
        for (const id of completedIds) {
          next.delete(id);
        }
        return next;
      });
    }, 2000);
    return () => clearTimeout(timer);
  }, [steps]);

  const toggleStep = useCallback((stepId: number) => {
    setExpandedSteps((prev) => {
      const next = new Set(prev);
      if (next.has(stepId)) {
        next.delete(stepId);
      } else {
        next.add(stepId);
      }
      return next;
    });
  }, []);

  if (steps.length === 0) return null;

  return (
    <div className="space-y-0">
      {steps.map((step, index) => (
        <PlanStepRow
          key={step.id}
          step={step}
          index={index}
          isLast={index === steps.length - 1}
          isExpanded={expandedSteps.has(step.id)}
          onToggle={() => toggleStep(step.id)}
        />
      ))}
    </div>
  );
}
