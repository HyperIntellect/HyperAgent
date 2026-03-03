"use client";

import React, { useState, useEffect, useMemo } from "react";
import { useTranslations } from "next-intl";
import {
    Check, AlertCircle, ChevronDown, Globe, ExternalLink,
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

// Animated pulsing dot for running state
function PulsingDot({ size = "md" }: { size?: "sm" | "md" }) {
    const outer = size === "sm" ? "w-4 h-4" : "w-5 h-5";
    const inner = size === "sm" ? "h-2 w-2" : "h-2.5 w-2.5";
    return (
        <span className={cn(outer, "flex items-center justify-center")}>
            <span className={cn("relative flex", inner)}>
                <span className={cn("animate-ping absolute inline-flex h-full w-full rounded-full bg-primary opacity-40")} />
                <span className={cn("relative inline-flex rounded-full bg-primary", inner)} />
            </span>
        </span>
    );
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
    if (seconds < 60) return <span className="tabular-nums text-xs font-medium text-muted-foreground/70">{Math.floor(seconds)}s</span>;
    const minutes = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return <span className="tabular-nums text-xs font-medium text-muted-foreground/70">{minutes}:{secs.toString().padStart(2, '0')}</span>;
}

// Stage status icon — simplified, no ring
function StageStatusIcon({ status, muted = false }: { status: "pending" | "running" | "completed" | "failed"; muted?: boolean }) {
    if (status === "running") {
        return <PulsingDot size="sm" />;
    }

    if (status === "completed") {
        return (
            <div className={cn(
                "w-4 h-4 rounded-full flex items-center justify-center",
                muted ? "bg-muted" : "bg-success/15"
            )}>
                <Check className={cn("w-2.5 h-2.5", muted ? "text-muted-foreground" : "text-success")} strokeWidth={2.5} />
            </div>
        );
    }

    if (status === "failed") {
        return (
            <div className="w-4 h-4 rounded-full bg-destructive/15 flex items-center justify-center">
                <AlertCircle className="w-2.5 h-2.5 text-destructive" strokeWidth={2.5} />
            </div>
        );
    }

    return <div className="w-2 h-2 rounded-full bg-muted-foreground/25 ml-1" />;
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
    const cls = className || "w-3 h-3 flex-shrink-0 text-muted-foreground/70";
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

// Rounded pill badge for a single tool
function ToolChip({ name, displayName, active }: { name: string; displayName: string; active?: boolean }) {
    return (
        <span className={cn(
            "inline-flex items-center gap-1.5 px-2.5 py-1 text-xs rounded-lg border border-border/50 max-w-[260px]",
            active ? "bg-secondary" : "bg-secondary/50"
        )}>
            <ToolIcon name={name} />
            <span className="truncate text-muted-foreground/80">{displayName}</span>
        </span>
    );
}

// Render tools as pill chips — group when > 6
function ToolChipList({ tools, tTools, isHistorical }: {
    tools: ProgressEvent[];
    tTools?: ReturnType<typeof useTranslations>;
    isHistorical: boolean;
}) {
    if (tools.length === 0) return null;

    if (tools.length <= 6) {
        // Render individual pills
        return (
            <div className="flex flex-wrap gap-1.5">
                {tools.map((tool, idx) => {
                    const toolName = tool.tool || tool.name || "unknown";
                    const skillId = toolName === "invoke_skill"
                        ? (tool.args?.skill_id as string | undefined)
                        : undefined;
                    const display = getToolDisplayName(toolName, tTools, skillId);
                    const endTs = getEventEndTimestamp(tool);
                    const isActive = !isHistorical && tool.status !== "completed" && endTs === undefined;
                    return (
                        <ToolChip
                            key={`tool-${idx}`}
                            name={skillId ? `invoke_skill:${skillId}` : toolName}
                            displayName={display}
                            active={isActive}
                        />
                    );
                })}
            </div>
        );
    }

    // Group by name when > 6 tools
    const grouped = groupTools(tools, tTools, isHistorical);
    return (
        <div className="flex flex-wrap gap-1.5">
            {grouped.map((g) => (
                <span
                    key={g.name}
                    className="inline-flex items-center gap-1.5 px-2.5 py-1 text-xs rounded-lg border border-border/50 bg-secondary/50 max-w-[260px]"
                >
                    <ToolIcon name={g.name} />
                    <span className="truncate text-muted-foreground/80">{g.displayName}</span>
                    {g.count > 1 && (
                        <span className="tabular-nums text-[10px] font-semibold text-muted-foreground/60 bg-muted/80 px-1.5 py-0.5 rounded-md leading-none">
                            {g.completedCount}/{g.count}
                        </span>
                    )}
                </span>
            ))}
        </div>
    );
}

// Collapsible stage section (accordion)
function StageSection({
    label,
    status,
    duration,
    tools,
    reasoningEvents,
    tTools,
    isHistorical = false,
    defaultExpanded = false,
}: {
    label: string;
    status: "pending" | "running" | "completed" | "failed";
    duration?: { start?: number; end?: number };
    tools?: ProgressEvent[];
    reasoningEvents?: ProgressEvent[];
    tTools?: ReturnType<typeof useTranslations>;
    isHistorical?: boolean;
    defaultExpanded?: boolean;
}) {
    const [isOpen, setIsOpen] = useState(() => defaultExpanded || status === "running");

    const hasContent = (tools && tools.length > 0) || (reasoningEvents && reasoningEvents.length > 0);
    const showDuration = !isHistorical && duration?.start !== undefined;

    return (
        <div className="py-1.5">
            {/* Stage header */}
            <button
                className="flex items-center gap-2.5 w-full text-left group hover:opacity-80"
                onClick={() => hasContent && setIsOpen(!isOpen)}
            >
                <div className="flex-shrink-0">
                    <StageStatusIcon status={status} muted={isHistorical && status === "completed"} />
                </div>

                <span className={cn(
                    "flex-1 text-base leading-snug font-semibold tracking-tight min-w-0 truncate",
                    status === "running" && "text-foreground",
                    status === "completed" && (isHistorical ? "text-muted-foreground/70" : "text-muted-foreground/80"),
                    status === "failed" && "text-destructive",
                    status === "pending" && "text-muted-foreground/40"
                )}>
                    {label}
                </span>

                {showDuration && duration?.start !== undefined && (
                    <div className="flex-shrink-0">
                        <LiveDuration startMs={duration.start} endMs={duration.end} />
                    </div>
                )}

                {hasContent && (
                    <ChevronDown className={cn(
                        "w-3.5 h-3.5 text-muted-foreground/50 flex-shrink-0 transition-transform duration-200",
                        !isOpen && "-rotate-90"
                    )} />
                )}
            </button>

            {/* Accordion content */}
            {hasContent && (
                <div className={cn("accordion-grid", isOpen && "accordion-open")}>
                    <div className="accordion-inner">
                        <div className="pl-[26px] pt-2 space-y-2.5">
                            {tools && tools.length > 0 && (
                                <ToolChipList tools={tools} tTools={tTools} isHistorical={isHistorical} />
                            )}

                            {reasoningEvents && reasoningEvents.length > 0 && (
                                <div className="space-y-1">
                                    {reasoningEvents.map((re, idx) => (
                                        <p
                                            key={`reasoning-${idx}`}
                                            className="text-xs text-muted-foreground/60 italic leading-relaxed"
                                        >
                                            {re.thinking}
                                        </p>
                                    ))}
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}

// Sources section with accordion
function SourcesSection({ sources }: { sources: Source[] }) {
    const [isExpanded, setIsExpanded] = useState(false);
    const tProgress = useTranslations("sidebar.progress");

    if (sources.length === 0) return null;

    return (
        <div className="mt-4 pt-4 border-t border-border/50">
            <button
                className="flex items-center gap-2.5 w-full text-left group hover:opacity-80 transition-opacity"
                onClick={() => setIsExpanded(!isExpanded)}
            >
                <Globe className="w-4 h-4 text-muted-foreground/70" />
                <span className="flex-1 text-xs font-semibold text-muted-foreground/90 uppercase tracking-wider">
                    {tProgress("sourcesCount", { count: sources.length })}
                </span>
                <ChevronDown className={cn(
                    "w-3.5 h-3.5 text-muted-foreground/50 transition-transform duration-200",
                    !isExpanded && "-rotate-90"
                )} />
            </button>

            <div className={cn("accordion-grid", isExpanded && "accordion-open")}>
                <div className="accordion-inner">
                    <div className="mt-3 space-y-0.5">
                        {sources.slice(0, 5).map((source) => (
                            <a
                                key={source.id}
                                href={source.url}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="flex items-center gap-2.5 py-2 px-2.5 -mx-2.5 rounded-md text-xs text-muted-foreground hover:text-foreground hover:bg-muted/60 transition-colors group"
                            >
                                <ExternalLink className="w-3.5 h-3.5 flex-shrink-0 opacity-60 group-hover:opacity-100 transition-opacity" />
                                <span className="truncate leading-relaxed">{source.title}</span>
                            </a>
                        ))}
                        {sources.length > 5 && (
                            <div className="pt-2 pl-6">
                                <span className="text-[11px] text-muted-foreground/60 font-medium">
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
 * Task progress panel — Manus-inspired collapsible stages with pill badges
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
    const [isExpanded, setIsExpanded] = useState(!isHistorical);
    const tProgress = useTranslations("sidebar.progress");
    const t = useTranslations("chat.agent");
    const tStages = useTranslations("chat.agent.stages");
    const tTools = useTranslations("chat.agent.tools");

    const processingLabel = tProgress("processing");
    const stageGroups = useMemo(() => {
        return groupEventsByStage(events, processingLabel, isHistorical);
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

    return (
        <div className={cn(
            "rounded-xl border border-border/50 bg-card overflow-hidden max-w-full",
            isHistorical ? "mt-4" : "mt-4 mb-6",
            className
        )}>
            {/* Header */}
            <button
                className="flex items-center gap-3.5 w-full px-5 py-3.5 text-left hover:bg-secondary/50 transition-colors"
                onClick={() => setIsExpanded(!isExpanded)}
            >
                {/* Status icon */}
                {isStreaming ? (
                    <PulsingDot />
                ) : progressSummary.hasError ? (
                    <div className="w-5 h-5 rounded-full bg-destructive/15 flex items-center justify-center">
                        <AlertCircle className="w-3 h-3 text-destructive" strokeWidth={2.5} />
                    </div>
                ) : isHistorical ? (
                    <div className="w-5 h-5 rounded-full bg-muted flex items-center justify-center">
                        <Check className="w-3 h-3 text-muted-foreground" strokeWidth={2.5} />
                    </div>
                ) : (
                    <div className="w-5 h-5 rounded-full bg-success/15 flex items-center justify-center">
                        <Check className="w-3 h-3 text-success" strokeWidth={2.5} />
                    </div>
                )}

                {/* Title / Summary */}
                <span className={cn(
                    "flex-1 text-sm font-semibold tracking-tight",
                    isHistorical ? "text-muted-foreground/90" : "text-foreground"
                )}>
                    {isHistorical
                        ? summaryText
                        : (isStreaming ? tProgress("processing") : tProgress("completed"))
                    }
                </span>

                {/* Progress badge */}
                {!isHistorical && (
                    <span className="text-xs tabular-nums font-semibold text-muted-foreground/80 px-2 py-1 rounded-md bg-muted/60">
                        {progressSummary.completed}/{progressSummary.total}
                    </span>
                )}

                {/* Expand chevron */}
                <ChevronDown className={cn(
                    "w-4 h-4 text-muted-foreground/50 transition-transform duration-200",
                    !isExpanded && "-rotate-90"
                )} />
            </button>

            {/* Content — accordion animated */}
            <div className={cn("accordion-grid", isExpanded && "accordion-open")}>
                <div className="accordion-inner">
                    <div className="px-5 pb-5 pt-1">
                        <div className="space-y-0">
                            {stageGroups.map((group, index) => {
                                const status = computeStageStatus(group, isHistorical, isStreaming);
                                const label = getStageDescription(group.stage, tStages, agentType, t, tTools);
                                const isLast = index === stageGroups.length - 1;

                                return (
                                    <StageSection
                                        key={`stage-${index}`}
                                        label={label}
                                        status={status}
                                        duration={{ start: group.startTime, end: group.endTime }}
                                        tools={group.tools}
                                        reasoningEvents={group.reasoningEvents}
                                        tTools={tTools}
                                        isHistorical={isHistorical}
                                        defaultExpanded={isLast || status === "running"}
                                    />
                                );
                            })}
                        </div>

                        <SourcesSection sources={sources} />

                        {/* Bottom footer — live mode only */}
                        {!isHistorical && (
                            <div className="mt-4 pt-3 border-t border-border/40 flex justify-end">
                                <span className="text-xs tabular-nums font-medium text-muted-foreground/60">
                                    {progressSummary.completed}/{progressSummary.total}
                                </span>
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
}
