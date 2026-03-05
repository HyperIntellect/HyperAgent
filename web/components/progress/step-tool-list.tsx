"use client";

import React, { useState } from "react";
import {
  Search, Terminal, FileText, Sparkles, Globe, ImageIcon, Wrench,
} from "lucide-react";
import { useTranslations } from "next-intl";
import type { ToolCallRecord } from "@/lib/stores/execution-progress-store";
import { StatusIndicator } from "./status-indicator";

// Tool icon mapping
function ToolIcon({ name, className }: { name: string; className?: string }) {
  const cls = className || "w-3 h-3 flex-shrink-0 text-muted-foreground/50";
  const n = name.toLowerCase();
  if (n.includes("search")) return <Search className={cls} />;
  if (n.includes("execute") || n.includes("command")) return <Terminal className={cls} />;
  if (n.includes("file")) return <FileText className={cls} />;
  if (n.startsWith("invoke_skill")) return <Sparkles className={cls} />;
  if (n.startsWith("browser")) return <Globe className={cls} />;
  if (n.includes("image")) return <ImageIcon className={cls} />;
  return <Wrench className={cls} />;
}

function getToolDisplayName(tool: string, tTools: ReturnType<typeof useTranslations>): string {
  const key = tool.replace(/^invoke_skill:/, "skill_");
  try {
    if (typeof tTools.has === "function" && tTools.has(key as Parameters<typeof tTools.has>[0])) {
      return tTools(key as Parameters<typeof tTools>[0]);
    }
  } catch { /* fall through */ }

  return tool
    .replace(/_/g, " ")
    .replace(/([a-z])([A-Z])/g, "$1 $2")
    .toLowerCase()
    .replace(/^\w/, (c) => c.toUpperCase());
}

interface GroupedTool {
  name: string;
  displayName: string;
  count: number;
  completedCount: number;
}

function groupToolCalls(tools: ToolCallRecord[], tTools: ReturnType<typeof useTranslations>): GroupedTool[] {
  const map = new Map<string, GroupedTool>();
  for (const tc of tools) {
    const key = tc.tool;
    const existing = map.get(key);
    if (existing) {
      existing.count += 1;
      if (tc.status === "completed") existing.completedCount += 1;
    } else {
      map.set(key, {
        name: key,
        displayName: getToolDisplayName(key, tTools),
        count: 1,
        completedCount: tc.status === "completed" ? 1 : 0,
      });
    }
  }
  return Array.from(map.values());
}

interface StepToolListProps {
  tools: ToolCallRecord[];
  maxVisible?: number;
}

/**
 * Renders tool calls nested within a plan step.
 * Groups by tool name when there are many tools, with "show more" toggle.
 */
export function StepToolList({ tools, maxVisible = 6 }: StepToolListProps) {
  const tTools = useTranslations("chat.agent.tools");
  const [showAll, setShowAll] = useState(false);

  if (tools.length === 0) return null;

  // If few tools, show individually
  if (tools.length <= maxVisible) {
    return (
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 mt-1">
        {tools.map((tc) => (
          <span
            key={tc.id}
            className="inline-flex items-center gap-1 text-xs text-muted-foreground/60"
          >
            <ToolIcon name={tc.tool} className="w-3 h-3 flex-shrink-0 text-muted-foreground/40" />
            <span className="truncate max-w-[180px]">
              {getToolDisplayName(tc.tool, tTools)}
            </span>
            <StatusIndicator status={tc.status} size="sm" />
          </span>
        ))}
      </div>
    );
  }

  // Group by name when > maxVisible
  const grouped = groupToolCalls(tools, tTools);
  const visible = showAll ? grouped : grouped.slice(0, maxVisible);
  const hiddenCount = grouped.length - maxVisible;

  return (
    <div className="mt-1 space-y-0.5">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
        {visible.map((g) => (
          <span
            key={g.name}
            className="inline-flex items-center gap-1 text-xs text-muted-foreground/60"
          >
            <ToolIcon name={g.name} className="w-3 h-3 flex-shrink-0 text-muted-foreground/40" />
            <span className="truncate max-w-[180px]">{g.displayName}</span>
            {g.count > 1 && (
              <span className="tabular-nums text-[10px] text-muted-foreground/40">
                {g.completedCount}/{g.count}
              </span>
            )}
          </span>
        ))}
      </div>
      {!showAll && hiddenCount > 0 && (
        <button
          onClick={() => setShowAll(true)}
          className="text-[11px] text-muted-foreground/40 hover:text-muted-foreground/60 transition-colors"
        >
          +{hiddenCount} more
        </button>
      )}
    </div>
  );
}
