"use client";

import React, { useState, useEffect, useMemo } from "react";
import { useTranslations } from "next-intl";
import {
    ChevronDown, Globe, ExternalLink,
    Search, Terminal, FileText, Sparkles, Wrench, ImageIcon,
} from "lucide-react";
import { cn } from "@/lib/utils";
import type { TimestampedEvent } from "@/lib/stores/agent-progress-store";
import type { Source, AgentEvent } from "@/lib/types";

// Union type for events - can be either timestamped (live) or plain (historical)
type ProgressEvent = TimestampedEvent | AgentEvent;

// Stage group containing a stage and its child tools
interface StageGroup {
    stage: ProgressEvent;
    stageIndex: number;
    tools: ProgressEvent[];
    reasoningEvents: ProgressEvent[]; // Reasoning transparency events
    startTime?: number; // Optional for historical events
    endTime?: number;
}

// Internal stages to hide from the UI
const HIDDEN_STAGES = new Set(["thinking", "routing"]);

// Known stage names for translation
const KNOWN_STAGES = [
    "handoff", "task", "analyze", "search", "tool", "write", "synthesize",
    "research", "plan", "generate", "execute", "summarize", "finalize",
    "outline", "data", "source", "code_result", "config", "search_tools",
    "collect", "report", "thinking", "routing", "refine", "present",
    "analyze_image", "generate_image",
    "browser_launch", "browser_navigate", "browser_click", "browser_type",
    "browser_screenshot", "browser_scroll", "browser_key", "browser_computer",
    "computer", "context", "processing",
    // App builder stages
    "scaffold", "server",
    // Deep research skill LangGraph nodes
    "init_config", "react_loop", "execute_tools",
    // Agentic search skill stages + LangGraph node names
    "classifying", "searching", "planning", "evaluating", "synthesizing",
    "classify", "quick_search", "plan_search", "execute_search", "evaluate",
];

/**
 * Helper to get timestamp from event (works for both TimestampedEvent and AgentEvent)
 */
function getEventTimestamp(event: ProgressEvent): number | undefined {
    return (event as TimestampedEvent).timestamp ?? event.timestamp;
}

function getEventEndTimestamp(event: ProgressEvent): number | undefined {
    return (event as TimestampedEvent).endTimestamp;
}

function groupEventsByStage(
    events: ProgressEvent[],
    processingLabel: string,
    isHistorical: boolean = false
): StageGroup[] {
    const groups: StageGroup[] = [];
    let currentGroup: StageGroup | null = null;
    const stageGroupsByName: Record<string, StageGroup> = {};
    const seenToolIds = new Set<string>();
    const completedToolIds = new Set<string>();

    for (const event of events) {
        if (event.type === "tool_result" && event.id) {
            completedToolIds.add(event.id);
        }
    }

    for (let i = 0; i < events.length; i++) {
        const event = events[i];
        const eventTimestamp = getEventTimestamp(event);
        const eventEndTimestamp = getEventEndTimestamp(event);

        if ((event as unknown as { type: string }).type === "browser_action") {
            const browserEvent = event as unknown as Record<string, unknown>;
            const action = browserEvent.action as string;
            const description = browserEvent.description as string;
            const target = browserEvent.target as string | undefined;
            const rawStatus = (browserEvent.status as string) || "running";
            const effectiveStatus: AgentEvent["status"] = rawStatus === "failed"
                ? "failed"
                : (isHistorical ? "completed" : (rawStatus as AgentEvent["status"]));

            const stageName = `browser_${action}`;
            const stageEvent: ProgressEvent = {
                type: "stage",
                name: stageName,
                description: target ? `${description}: ${target}` : description,
                status: effectiveStatus,
                timestamp: eventTimestamp,
            };

            if (stageEvent.name && HIDDEN_STAGES.has(stageEvent.name)) continue;

            const name = stageEvent.name;
            if (name && stageGroupsByName[name]) {
                // Update existing group
                stageGroupsByName[name].stage = {
                    ...stageGroupsByName[name].stage,
                    status: stageEvent.status,
                };
                if (eventEndTimestamp || eventTimestamp) {
                    stageGroupsByName[name].endTime = eventEndTimestamp || eventTimestamp;
                }
            } else {
                currentGroup = {
                    stage: stageEvent,
                    stageIndex: i,
                    tools: [],
                    reasoningEvents: [],
                    startTime: eventTimestamp,
                    endTime: eventEndTimestamp,
                };
                groups.push(currentGroup);
                if (name) stageGroupsByName[name] = currentGroup;
            }
            continue;
        }

        // Handle reasoning events - attach to current group
        if (event.type === "reasoning") {
            if (currentGroup) {
                currentGroup.reasoningEvents.push(event);
            } else {
                // Create an implicit group for orphaned reasoning events
                currentGroup = {
                    stage: {
                        type: "stage",
                        name: "processing",
                        description: processingLabel,
                        status: isHistorical ? "completed" : "running",
                        timestamp: eventTimestamp,
                    },
                    stageIndex: -1,
                    tools: [],
                    reasoningEvents: [event],
                    startTime: eventTimestamp,
                };
                groups.push(currentGroup);
            }
            continue;
        }

        if (event.type === "stage") {
            if (event.name && HIDDEN_STAGES.has(event.name)) continue;

            // For historical events, treat everything as completed
            const effectiveStatus = event.status === "failed"
                ? "failed"
                : (isHistorical ? "completed" : event.status);

            if (event.status === "running" && !isHistorical) {
                currentGroup = {
                    stage: { ...event, status: effectiveStatus },
                    stageIndex: i,
                    tools: [],
                    reasoningEvents: [],
                    startTime: eventTimestamp,
                    endTime: eventEndTimestamp,
                };
                groups.push(currentGroup);
                if (event.name) stageGroupsByName[event.name] = currentGroup;
            } else if (event.status === "completed" || event.status === "failed" || isHistorical) {
                const stageName = event.name;
                if (stageName && stageGroupsByName[stageName]) {
                    if (eventEndTimestamp || eventTimestamp) {
                        stageGroupsByName[stageName].endTime = eventEndTimestamp || eventTimestamp;
                    }
                    stageGroupsByName[stageName].stage = {
                        ...stageGroupsByName[stageName].stage,
                        status: effectiveStatus,
                    };
                } else {
                    currentGroup = {
                        stage: { ...event, status: effectiveStatus },
                        stageIndex: i,
                        tools: [],
                        reasoningEvents: [],
                        startTime: eventTimestamp,
                        endTime: eventEndTimestamp || eventTimestamp,
                    };
                    groups.push(currentGroup);
                    if (stageName) stageGroupsByName[stageName] = currentGroup;
                }
            }
        } else if (event.type === "tool_call") {
            const toolKey = event.id || `${event.tool || event.name}-${i}`;
            if (seenToolIds.has(toolKey)) continue;
            seenToolIds.add(toolKey);

            const toolEvent: ProgressEvent = {
                ...event,
                // For historical events, mark tools as completed
                status: isHistorical ? "completed" : (
                    event.id && completedToolIds.has(event.id) ? "completed" : event.status
                ),
            };

            if (currentGroup) {
                currentGroup.tools.push(toolEvent);
            } else {
                const implicitGroup: StageGroup = {
                    stage: {
                        type: "stage",
                        name: "processing",
                        description: processingLabel,
                        status: isHistorical ? "completed" : "running",
                        timestamp: eventTimestamp,
                    },
                    stageIndex: -1,
                    tools: [toolEvent],
                    reasoningEvents: [],
                    startTime: eventTimestamp,
                };
                groups.push(implicitGroup);
                currentGroup = implicitGroup;
            }
        }
    }

    return groups;
}

function getStageDescription(
    stage: ProgressEvent,
    tStages: ReturnType<typeof useTranslations>,
    agentType?: string,
    t?: ReturnType<typeof useTranslations>,
    tTools?: ReturnType<typeof useTranslations>
): string {
    const stageName = stage.name || "processing";
    const status = stage.status || "running";

    if (agentType === "image") {
        if (stageName === "analyze" || stageName === "generate") {
            const translated = tryTranslate(tStages, `${stageName}_image.${status}`, `${stageName}_image`);
            if (translated) return translated;
        }
    }

    if (KNOWN_STAGES.includes(stageName)) {
        const translated = tryTranslate(tStages, `${stageName}.${status}`, "chat.agent.stages");
        if (translated && translated.trim()) return translated;
    }

    // Handle skill_*:node_name format (e.g. "skill_app_builder:scaffold")
    // Extract the node name suffix and try known stage translation
    if (stageName.includes(":")) {
        const nodeName = stageName.split(":").pop() || "";
        if (KNOWN_STAGES.includes(nodeName)) {
            const translated = tryTranslate(tStages, `${nodeName}.${status}`, "chat.agent.stages");
            if (translated && translated.trim()) return translated;
        }
    }

    // Check if description matches "Running {node}" or "Executing {tool}" and translate
    if (stage.description) {
        // Handle "Running {node_name}" from skill executor
        const runningMatch = stage.description.match(/^Running (.+)$/i);
        if (runningMatch) {
            const nodeName = runningMatch[1];
            if (KNOWN_STAGES.includes(nodeName)) {
                const translated = tryTranslate(tStages, `${nodeName}.${status}`, "chat.agent.stages");
                if (translated && translated.trim()) return translated;
            }
        }

        const executingMatch = stage.description.match(/^Executing (.+)$/i);
        if (executingMatch && t && tTools) {
            const toolNameRaw = executingMatch[1];
            const toolKey = toolNameRaw.toLowerCase().replace(/\s+/g, "_");
            const skillKey = `skill_${toolKey}`;
            let translatedToolName = toolNameRaw; // Fallback to original

            // Try to translate as skill first, then as regular tool
            const skillTranslated = tryTranslate(tTools, skillKey, "chat.agent.tools");
            if (skillTranslated) {
                translatedToolName = skillTranslated;
            } else {
                const toolTranslated = getToolDisplayName(toolKey, tTools);
                if (toolTranslated && toolTranslated !== toolKey) {
                    translatedToolName = toolTranslated;
                }
            }

            // Use the translation system for "Executing {tool}"
            const hasExecutingKey = t && typeof t.has === "function" && t.has("executing" as Parameters<typeof t.has>[0]);
            if (hasExecutingKey) {
                try {
                    return t("executing", { tool: translatedToolName });
                } catch {
                    // Fall through
                }
            }
        }
        return stage.description;
    }
    return stageName.charAt(0).toUpperCase() + stageName.slice(1).replace(/_/g, " ");
}

/**
 * Safely try to translate a key using next-intl, avoiding MISSING_MESSAGE console errors.
 * Returns the translated string if the key exists, otherwise undefined.
 */
function tryTranslate(
    t: ReturnType<typeof useTranslations>,
    key: string,
    namespace?: string
): string | undefined {
    try {
        // Use .has() to check existence before translating (avoids console error)
        if (typeof t.has === "function" && !t.has(key as Parameters<typeof t.has>[0])) {
            return undefined;
        }
        const translated = t(key as Parameters<typeof t>[0]);
        // Double-check the result doesn't contain the namespace path (fallback indicator)
        if (translated && (!namespace || !translated.includes(namespace))) {
            return translated;
        }
    } catch {
        // Fall through
    }
    return undefined;
}

function getToolDisplayName(
    toolName: string,
    tTools?: ReturnType<typeof useTranslations>,
    skillId?: string
): string {
    // Special case: for invoke_skill, show the skill name instead
    if (toolName === "invoke_skill" && skillId) {
        // Try to get translation for the skill
        if (tTools) {
            const translated = tryTranslate(tTools, `skill_${skillId}`, "chat.agent.tools");
            if (translated) return translated;
        }
        // Fallback: format the skill_id nicely
        return skillId
            .replace(/_/g, " ")
            .replace(/([a-z])([A-Z])/g, "$1 $2")
            .toLowerCase()
            .replace(/^\w/, (c) => c.toUpperCase());
    }

    if (tTools) {
        const translated = tryTranslate(tTools, toolName, "chat.agent.tools");
        if (translated) return translated;
    }

    return toolName
        .replace(/_/g, " ")
        .replace(/([a-z])([A-Z])/g, "$1 $2")
        .toLowerCase()
        .replace(/^\w/, (c) => c.toUpperCase());
}

// Live duration with clean formatting
function LiveDuration({ startMs, endMs }: { startMs: number; endMs?: number }) {
    const [now, setNow] = useState(() => Date.now());

    useEffect(() => {
        if (endMs) return;
        const interval = setInterval(() => setNow(Date.now()), 1000);
        return () => clearInterval(interval);
    }, [endMs]);

    const duration = endMs ? endMs - startMs : now - startMs;
    const seconds = duration / 1000;

    if (seconds < 1) return null;
    if (seconds < 60) return <span className="tabular-nums text-xs font-medium text-muted-foreground/50">{Math.floor(seconds)}s</span>;
    const minutes = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return <span className="tabular-nums text-xs font-medium text-muted-foreground/50">{minutes}:{secs.toString().padStart(2, '0')}</span>;
}

// Group tools by name and count them
interface GroupedTool {
    name: string;
    displayName: string;
    count: number;
    completedCount: number;
}

function groupTools(
    tools: ProgressEvent[],
    tTools?: ReturnType<typeof useTranslations>,
    isHistorical: boolean = false
): GroupedTool[] {
    const toolMap = new Map<string, GroupedTool>();

    for (const tool of tools) {
        const toolName = tool.tool || tool.name || "unknown";
        // For invoke_skill, extract the skill_id from args and use it as part of the key
        const skillId = toolName === "invoke_skill"
            ? (tool.args?.skill_id as string | undefined)
            : undefined;
        // Use skill_id in the key to group by specific skill
        const groupKey = skillId ? `invoke_skill:${skillId}` : toolName;

        const existing = toolMap.get(groupKey);
        const endTimestamp = getEventEndTimestamp(tool);
        const isCompleted = isHistorical || tool.status === "completed" || endTimestamp !== undefined;

        if (existing) {
            existing.count += 1;
            if (isCompleted) existing.completedCount += 1;
        } else {
            toolMap.set(groupKey, {
                name: groupKey,
                displayName: getToolDisplayName(toolName, tTools, skillId),
                count: 1,
                completedCount: isCompleted ? 1 : 0,
            });
        }
    }

    return Array.from(toolMap.values());
}

// Tool icon component — maps tool names to Lucide icons
function ToolIcon({ name: toolName, className }: { name: string; className?: string }) {
    const cls = className || "w-3 h-3 flex-shrink-0 text-muted-foreground/50";
    const n = toolName.toLowerCase();
    if (n === "web_search" || n === "google_search" || n.includes("search")) return <Search className={cls} />;
    if (n === "execute_code" || n === "app_run_command" || n.includes("execute")) return <Terminal className={cls} />;
    if (n === "app_write_file" || n === "app_read_file" || n.includes("sandbox_file") || n.includes("file")) return <FileText className={cls} />;
    if (n === "invoke_skill" || n.startsWith("invoke_skill:")) return <Sparkles className={cls} />;
    if (n.startsWith("browser_") || n === "browser") return <Globe className={cls} />;
    if (n === "generate_image" || n === "analyze_image" || n.includes("image")) return <ImageIcon className={cls} />;
    return <Wrench className={cls} />;
}

// Compute stage status from group data
function computeStageStatus(
    group: StageGroup,
    isHistorical: boolean,
    isStreaming: boolean
): "pending" | "running" | "completed" | "failed" {
    if (group.stage.status === "failed") return "failed";
    if (isHistorical) return "completed";

    const hasTools = group.tools.length > 0;
    const allToolsCompleted = hasTools && group.tools.every(t => {
        const endTs = getEventEndTimestamp(t);
        return endTs !== undefined || t.status === "completed";
    });
    const stageEndTs = getEventEndTimestamp(group.stage);

    if (group.stage.status === "completed" || stageEndTs !== undefined || allToolsCompleted || !isStreaming) {
        return "completed";
    }
    if (group.stage.status === "running") return "running";
    return "pending";
}

/**
 * Auto-complete stages that are still "running" but have later stages
 * already started — the backend may not always send explicit completion events.
 */
function autoCompleteStaleStages(groups: StageGroup[]): StageGroup[] {
    if (groups.length <= 1) return groups;

    const result = [...groups];
    for (let i = 0; i < result.length - 1; i++) {
        const current = result[i];
        if (current.stage.status !== "running") continue;

        // Check if any later stage has started
        const hasLaterActivity = result.slice(i + 1).some(
            (g) => g.startTime !== undefined || g.stage.status === "running" || g.stage.status === "completed"
        );

        if (hasLaterActivity) {
            const nextStart = result[i + 1]?.startTime;
            result[i] = {
                ...current,
                stage: { ...current.stage, status: "completed" },
                endTime: current.endTime ?? nextStart,
            };
        }
    }
    return result;
}

// --- New timeline components ---

// Tiny dot indicator for stage status
function StageDot({ status }: { status: "pending" | "running" | "completed" | "failed" }) {
    if (status === "running") {
        return (
            <span className="flex items-center justify-center w-2.5 h-2.5">
                <span className="w-2 h-2 rounded-full border-[1.5px] border-primary border-t-transparent animate-spin-slow" />
            </span>
        );
    }
    const dotClass = {
        completed: "bg-primary/40",
        failed: "bg-destructive/50",
        pending: "bg-muted-foreground/20",
    }[status];
    return (
        <span className="flex items-center justify-center w-2.5 h-2.5">
            <span className={cn("w-1.5 h-1.5 rounded-full", dotClass)} />
        </span>
    );
}

// Inline tool list — plain text, no borders
function InlineToolList({ tools, tTools, isHistorical, isActive }: {
    tools: ProgressEvent[];
    tTools?: ReturnType<typeof useTranslations>;
    isHistorical: boolean;
    isActive: boolean;
}) {
    if (tools.length === 0) return null;

    const textClass = isActive ? "text-muted-foreground/70" : "text-muted-foreground/50";

    if (tools.length <= 6) {
        return (
            <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
                {tools.map((tool, idx) => {
                    const toolName = tool.tool || tool.name || "unknown";
                    const skillId = toolName === "invoke_skill"
                        ? (tool.args?.skill_id as string | undefined)
                        : undefined;
                    const display = getToolDisplayName(toolName, tTools, skillId);
                    return (
                        <span key={`tool-${idx}`} className={cn("inline-flex items-center gap-1 text-xs", textClass)}>
                            <ToolIcon name={skillId ? `invoke_skill:${skillId}` : toolName} className={cn("w-3 h-3 flex-shrink-0", textClass)} />
                            <span className="truncate max-w-[200px]">{display}</span>
                        </span>
                    );
                })}
            </div>
        );
    }

    // Group by name when > 6 tools
    const grouped = groupTools(tools, tTools, isHistorical);
    return (
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
            {grouped.map((g) => (
                <span key={g.name} className={cn("inline-flex items-center gap-1 text-xs", textClass)}>
                    <ToolIcon name={g.name} className={cn("w-3 h-3 flex-shrink-0", textClass)} />
                    <span className="truncate max-w-[200px]">{g.displayName}</span>
                    {g.count > 1 && (
                        <span className="tabular-nums text-[10px] text-muted-foreground/40">
                            {g.completedCount}/{g.count}
                        </span>
                    )}
                </span>
            ))}
        </div>
    );
}

// Timeline row: dot + vertical connector + label + inline tools
function TimelineStage({
    label,
    status,
    duration,
    tools,
    reasoningEvents,
    tTools,
    isHistorical,
    isLast,
    showTools,
}: {
    label: string;
    status: "pending" | "running" | "completed" | "failed";
    duration?: { start?: number; end?: number };
    tools?: ProgressEvent[];
    reasoningEvents?: ProgressEvent[];
    tTools?: ReturnType<typeof useTranslations>;
    isHistorical: boolean;
    isLast: boolean;
    showTools: boolean;
}) {
    const showDuration = !isHistorical && duration?.start !== undefined;
    const hasTools = showTools && tools && tools.length > 0;
    const hasReasoning = reasoningEvents && reasoningEvents.length > 0;
    const hasContent = hasTools || hasReasoning;

    return (
        <div className="relative">
            {/* Vertical connector line */}
            {!isLast && (
                <div className="absolute left-[5px] top-[14px] bottom-0 w-px bg-border/40" />
            )}

            {/* Stage header row */}
            <div className="flex items-center gap-2.5 py-1.5">
                <div className="flex-shrink-0 relative z-10">
                    <StageDot status={status} />
                </div>

                <span className={cn(
                    "flex-1 text-sm font-medium leading-snug min-w-0 truncate",
                    status === "running" && "text-foreground",
                    status === "completed" && "text-muted-foreground/60",
                    status === "failed" && "text-destructive/80",
                    status === "pending" && "text-muted-foreground/30"
                )}>
                    {label}
                </span>

                {showDuration && duration?.start !== undefined && (
                    <div className="flex-shrink-0">
                        <LiveDuration startMs={duration.start} endMs={duration.end} />
                    </div>
                )}
            </div>

            {/* Tool list and reasoning — indented under the connector */}
            {hasContent && (
                <div className="pl-[22px] pb-1">
                    {hasTools && (
                        <InlineToolList
                            tools={tools!}
                            tTools={tTools}
                            isHistorical={isHistorical}
                            isActive={status === "running"}
                        />
                    )}
                    {hasReasoning && (
                        <div className="mt-1 space-y-0.5">
                            {reasoningEvents!.map((re, idx) => (
                                <p
                                    key={`reasoning-${idx}`}
                                    className="text-xs text-muted-foreground/40 italic leading-relaxed"
                                >
                                    {re.thinking}
                                </p>
                            ))}
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}

// Sources section — streamlined for borderless layout
function SourcesSection({ sources }: { sources: Source[] }) {
    const [isExpanded, setIsExpanded] = useState(false);
    const tProgress = useTranslations("sidebar.progress");

    if (sources.length === 0) return null;

    return (
        <div className="mt-3 pt-2">
            <button
                className="flex items-center gap-2 w-full text-left group hover:opacity-80 transition-opacity"
                onClick={() => setIsExpanded(!isExpanded)}
            >
                <Globe className="w-3 h-3 text-muted-foreground/50" />
                <span className="flex-1 text-xs font-medium text-muted-foreground/50 uppercase tracking-wider">
                    {tProgress("sourcesCount", { count: sources.length })}
                </span>
                <ChevronDown className={cn(
                    "w-3 h-3 text-muted-foreground/40 transition-transform duration-200",
                    !isExpanded && "-rotate-90"
                )} />
            </button>

            <div className={cn("accordion-grid", isExpanded && "accordion-open")}>
                <div className="accordion-inner">
                    <div className="mt-2 space-y-0.5">
                        {sources.slice(0, 5).map((source) => (
                            <a
                                key={source.id}
                                href={source.url}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="flex items-center gap-2 py-1.5 px-2 -mx-2 rounded-md text-xs text-muted-foreground/50 hover:text-foreground hover:bg-muted/60 transition-colors group"
                            >
                                <ExternalLink className="w-3 h-3 flex-shrink-0 opacity-60 group-hover:opacity-100 transition-opacity" />
                                <span className="truncate leading-relaxed">{source.title}</span>
                            </a>
                        ))}
                        {sources.length > 5 && (
                            <div className="pt-1.5 pl-5">
                                <span className="text-[11px] text-muted-foreground/40 font-medium">
                                    {tProgress("moreSources", { count: sources.length - 5 })}
                                </span>
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
}

interface TaskProgressPanelProps {
    events: ProgressEvent[];
    sources?: Source[];
    isStreaming?: boolean; // Optional - when false or undefined, treats as historical
    agentType?: string;
    className?: string;
}

/**
 * Task progress panel — clean borderless timeline
 * Handles both live streaming events and historical (saved) events
 */
export function TaskProgressPanel({
    events,
    sources = [],
    isStreaming = false,
    agentType,
    className,
}: TaskProgressPanelProps) {
    const isHistorical = !isStreaming;
    const [isExpanded, setIsExpanded] = useState(false);
    const tProgress = useTranslations("sidebar.progress");
    const t = useTranslations("chat.agent");
    const tStages = useTranslations("chat.agent.stages");
    const tTools = useTranslations("chat.agent.tools");

    const processingLabel = tProgress("processing");
    const stageGroups = useMemo(() => {
        const groups = groupEventsByStage(events, processingLabel, isHistorical);
        // Auto-complete stale running stages when later stages have started
        return isHistorical ? groups : autoCompleteStaleStages(groups);
    }, [events, processingLabel, isHistorical]);

    const progressSummary = useMemo(() => {
        let completed = 0;
        let totalTools = 0;
        let hasError = false;

        for (const group of stageGroups) {
            if (group.stage.status === "failed") hasError = true;
            totalTools += group.tools.length;

            if (isHistorical) {
                if (group.stage.status !== "failed") completed++;
                continue;
            }

            const status = computeStageStatus(group, isHistorical, isStreaming);
            if (status === "completed") completed++;
        }

        return { completed, total: stageGroups.length, totalTools, hasError };
    }, [stageGroups, isStreaming, isHistorical]);

    const summaryText = useMemo(() => {
        if (!isHistorical) return null;
        const parts: string[] = [];
        if (progressSummary.completed > 0) {
            parts.push(t("completedStages", { count: progressSummary.completed }));
        }
        if (progressSummary.totalTools > 0) {
            parts.push(t("toolsUsed", { count: progressSummary.totalTools }));
        }
        return parts.join(" · ") || t("completedStages", { count: 0 });
    }, [isHistorical, progressSummary, t]);

    if (stageGroups.length === 0) return null;

    // Shared timeline renderer
    const renderTimeline = () => (
        <>
            {stageGroups.map((group, index) => {
                const status = computeStageStatus(group, isHistorical, isStreaming);
                const label = getStageDescription(group.stage, tStages, agentType, t, tTools);
                const isLast = index === stageGroups.length - 1;
                // Live mode: show tools on running stages. Historical: show tools on all stages.
                const showTools = isHistorical || status === "running";

                return (
                    <TimelineStage
                        key={`stage-${index}`}
                        label={label}
                        status={status}
                        duration={{ start: group.startTime, end: group.endTime }}
                        tools={group.tools}
                        reasoningEvents={group.reasoningEvents}
                        tTools={tTools}
                        isHistorical={isHistorical}
                        isLast={isLast}
                        showTools={showTools}
                    />
                );
            })}
            <SourcesSection sources={sources} />
        </>
    );

    // Live streaming mode — borderless inline timeline
    if (isStreaming) {
        return (
            <div className={cn("mt-2 mb-4", className)}>
                {renderTimeline()}
            </div>
        );
    }

    // Historical mode — compact summary toggle
    return (
        <div className={cn("mt-3", className)}>
            <button
                className="flex items-center gap-2 w-full text-left group hover:opacity-80 transition-opacity py-1.5"
                onClick={() => setIsExpanded(!isExpanded)}
            >
                <ChevronDown className={cn(
                    "w-3.5 h-3.5 text-muted-foreground/40 flex-shrink-0 transition-transform duration-200",
                    !isExpanded && "-rotate-90"
                )} />
                <span className="text-xs text-muted-foreground/60 font-medium">
                    {summaryText}
                </span>
            </button>

            <div className={cn("accordion-grid", isExpanded && "accordion-open")}>
                <div className="accordion-inner">
                    <div className="pl-1 pt-1 pb-2">
                        {renderTimeline()}
                    </div>
                </div>
            </div>
        </div>
    );
}
