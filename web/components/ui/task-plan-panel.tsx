"use client";

import React, { useState } from "react";
import {
  ChevronRight,
  Check,
  Circle,
  AlertTriangle,
  ListChecks,
  Target,
  HelpCircle,
  Lightbulb,
  ArrowRight,
  Workflow,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { useTranslations } from "next-intl";

/**
 * Plan step from task_planning skill output
 */
export interface PlanStep {
  step_number: number;
  action: string;
  tool_or_skill?: string | null;
  depends_on: number[];
  estimated_complexity: "low" | "medium" | "high";
}

/**
 * Full task plan output from task_planning skill
 */
export interface TaskPlan {
  task_summary: string;
  complexity_assessment: "simple" | "moderate" | "complex";
  steps: PlanStep[];
  success_criteria: string[];
  potential_challenges: string[];
  clarifying_questions: string[];
}

interface TaskPlanPanelProps {
  plan: TaskPlan;
  className?: string;
  defaultExpanded?: boolean;
}

// Complexity badge colors — labels come from i18n via `taskPlan.complexity.*`
const COMPLEXITY_CONFIG = {
  simple: {
    labelKey: "complexity.simple" as const,
    color: "bg-emerald-500/15 text-emerald-600 dark:text-emerald-400",
    dot: "bg-emerald-500",
  },
  moderate: {
    labelKey: "complexity.moderate" as const,
    color: "bg-amber-500/15 text-amber-600 dark:text-amber-400",
    dot: "bg-amber-500",
  },
  complex: {
    labelKey: "complexity.complex" as const,
    color: "bg-rose-500/15 text-rose-600 dark:text-rose-400",
    dot: "bg-rose-500",
  },
};

const STEP_COMPLEXITY_CONFIG = {
  low: { dot: "bg-emerald-500" },
  medium: { dot: "bg-amber-500" },
  high: { dot: "bg-rose-500" },
};

// Tool/skill badge
function ToolBadge({ name }: { name: string }) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 px-1.5 py-0.5",
        "text-[10px] font-medium",
        "bg-primary/10 text-primary/80",
        "rounded-md"
      )}
    >
      <Workflow className="w-2.5 h-2.5" />
      {name}
    </span>
  );
}

// Dependency indicator
function DependencyBadge({ stepNumbers }: { stepNumbers: number[] }) {
  const t = useTranslations("taskPlan");
  if (stepNumbers.length === 0) return null;

  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 px-1.5 py-0.5",
        "text-[10px] font-medium",
        "bg-muted text-muted-foreground/70",
        "rounded-md"
      )}
    >
      <ArrowRight className="w-2.5 h-2.5" />
      {t("after", { steps: stepNumbers.join(", ") })}
    </span>
  );
}

/**
 * Task Plan Panel - displays a structured execution plan
 * Shows steps, dependencies, tools, success criteria, and challenges
 */
export function TaskPlanPanel({
  plan,
  className,
  defaultExpanded = true,
}: TaskPlanPanelProps) {
  const t = useTranslations("taskPlan");
  const [isExpanded, setIsExpanded] = useState(defaultExpanded);
  const [expandedSections, setExpandedSections] = useState<Set<string>>(
    new Set(["steps"])
  );

  const toggleSection = (section: string) => {
    const newSections = new Set(expandedSections);
    if (newSections.has(section)) {
      newSections.delete(section);
    } else {
      newSections.add(section);
    }
    setExpandedSections(newSections);
  };

  const complexityConfig = COMPLEXITY_CONFIG[plan.complexity_assessment];

  return (
    <div
      className={cn(
        "rounded-xl border border-border/80 bg-card/50 overflow-hidden",
        "shadow-sm",
        className
      )}
    >
      {/* Header */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className={cn(
          "flex items-center gap-3 w-full px-4 py-3.5",
          "text-left hover:bg-muted/30 transition-colors",
          "border-b border-border/40"
        )}
      >
        {/* Plan icon */}
        <div
          className={cn(
            "w-8 h-8 rounded-lg flex items-center justify-center",
            "bg-primary/10"
          )}
        >
          <ListChecks className="w-4 h-4 text-primary" />
        </div>

        {/* Title and summary */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className="text-sm font-semibold text-foreground">
              {t("executionPlan")}
            </span>
            <span
              className={cn(
                "inline-flex items-center gap-1 px-2 py-0.5",
                "text-[10px] font-medium rounded-full",
                complexityConfig.color
              )}
            >
              <span
                className={cn("w-1.5 h-1.5 rounded-full", complexityConfig.dot)}
              />
              {t(complexityConfig.labelKey)}
            </span>
          </div>
          <p className="text-xs text-muted-foreground mt-0.5 truncate">
            {plan.task_summary}
          </p>
        </div>

        {/* Step count badge */}
        <div
          className={cn(
            "flex items-center gap-1.5 px-2.5 py-1",
            "text-xs font-medium text-muted-foreground",
            "bg-muted/50 rounded-full"
          )}
        >
          <span>{plan.steps.length}</span>
          <span className="text-muted-foreground/60">{t("stepsCount")}</span>
        </div>

        {/* Expand chevron */}
        <ChevronRight
          className={cn(
            "w-4 h-4 text-muted-foreground/50 transition-transform duration-200",
            isExpanded && "rotate-90"
          )}
        />
      </button>

      {/* Expanded content */}
      {isExpanded && (
        <div className="p-4 space-y-4">
          {/* Steps Section */}
          <div>
            <button
              onClick={() => toggleSection("steps")}
              className="flex items-center gap-2 mb-3 text-xs font-semibold text-muted-foreground uppercase tracking-wider hover:text-foreground transition-colors"
            >
              <ChevronRight
                className={cn(
                  "w-3 h-3 transition-transform",
                  expandedSections.has("steps") && "rotate-90"
                )}
              />
              {t("steps")}
            </button>

            {expandedSections.has("steps") && (
              <div className="space-y-0 ml-1">
                {plan.steps.map((step, index) => {
                  const isLast = index === plan.steps.length - 1;
                  const complexityDot =
                    STEP_COMPLEXITY_CONFIG[step.estimated_complexity].dot;

                  return (
                    <div key={step.step_number} className="relative">
                      {/* Vertical connector line */}
                      {!isLast && (
                        <div className="absolute left-[11px] top-7 bottom-0 w-px bg-border/50" />
                      )}

                      <div className="flex items-start gap-3 py-2">
                        {/* Step number circle */}
                        <div
                          className={cn(
                            "flex-shrink-0 w-6 h-6 rounded-full",
                            "flex items-center justify-center",
                            "border-2 border-border bg-card",
                            "text-[11px] font-semibold text-muted-foreground"
                          )}
                        >
                          {step.step_number}
                        </div>

                        {/* Step content */}
                        <div className="flex-1 min-w-0 pt-0.5">
                          <p className="text-sm text-foreground/90 leading-relaxed">
                            {step.action}
                          </p>

                          {/* Tool/skill and dependencies */}
                          <div className="flex flex-wrap items-center gap-2 mt-1.5">
                            {step.tool_or_skill && (
                              <ToolBadge name={step.tool_or_skill} />
                            )}
                            <DependencyBadge stepNumbers={step.depends_on} />
                            {/* Complexity indicator */}
                            <span
                              className={cn(
                                "w-1.5 h-1.5 rounded-full",
                                complexityDot
                              )}
                              title={t("stepComplexity", { complexity: t(`complexity.${step.estimated_complexity}`) })}
                            />
                          </div>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>

          {/* Success Criteria Section */}
          {plan.success_criteria.length > 0 && (
            <div>
              <button
                onClick={() => toggleSection("criteria")}
                className="flex items-center gap-2 mb-2 text-xs font-semibold text-muted-foreground uppercase tracking-wider hover:text-foreground transition-colors"
              >
                <ChevronRight
                  className={cn(
                    "w-3 h-3 transition-transform",
                    expandedSections.has("criteria") && "rotate-90"
                  )}
                />
                <Target className="w-3 h-3" />
                {t("successCriteria")}
              </button>

              {expandedSections.has("criteria") && (
                <div className="space-y-1.5 ml-5">
                  {plan.success_criteria.map((criterion, index) => (
                    <div
                      key={index}
                      className="flex items-start gap-2 text-sm text-foreground/80"
                    >
                      <Check
                        className="w-3.5 h-3.5 text-emerald-500 mt-0.5 flex-shrink-0"
                        strokeWidth={2.5}
                      />
                      <span>{criterion}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Potential Challenges Section */}
          {plan.potential_challenges.length > 0 && (
            <div>
              <button
                onClick={() => toggleSection("challenges")}
                className="flex items-center gap-2 mb-2 text-xs font-semibold text-muted-foreground uppercase tracking-wider hover:text-foreground transition-colors"
              >
                <ChevronRight
                  className={cn(
                    "w-3 h-3 transition-transform",
                    expandedSections.has("challenges") && "rotate-90"
                  )}
                />
                <AlertTriangle className="w-3 h-3" />
                {t("potentialChallenges")}
              </button>

              {expandedSections.has("challenges") && (
                <div className="space-y-1.5 ml-5">
                  {plan.potential_challenges.map((challenge, index) => (
                    <div
                      key={index}
                      className="flex items-start gap-2 text-sm text-foreground/80"
                    >
                      <Circle
                        className="w-1.5 h-1.5 fill-amber-500 text-amber-500 mt-2 flex-shrink-0"
                      />
                      <span>{challenge}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Clarifying Questions Section */}
          {plan.clarifying_questions.length > 0 && (
            <div
              className={cn(
                "p-3 rounded-lg",
                "bg-accent-cyan/5 border border-accent-cyan/20"
              )}
            >
              <div className="flex items-center gap-2 mb-2">
                <HelpCircle className="w-4 h-4 text-accent-cyan" />
                <span className="text-xs font-semibold text-accent-cyan">
                  {t("questionsToClarify")}
                </span>
              </div>
              <div className="space-y-1.5 ml-6">
                {plan.clarifying_questions.map((question, index) => (
                  <div
                    key={index}
                    className="flex items-start gap-2 text-sm text-foreground/80"
                  >
                    <Lightbulb
                      className="w-3.5 h-3.5 text-accent-cyan/70 mt-0.5 flex-shrink-0"
                    />
                    <span>{question}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
