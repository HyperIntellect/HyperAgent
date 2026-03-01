"use client";

import React, { useState, memo, useMemo } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import { useTranslations } from "next-intl";
import Image from "next/image";
import { Copy, Check, RotateCcw, ImageIcon } from "lucide-react";
import { cn } from "@/lib/utils";
import { usePreviewStore } from "@/lib/stores/preview-store";
import { TaskProgressPanel } from "@/components/ui/task-progress-panel";
import { TaskPlanPanel, type TaskPlan } from "@/components/ui/task-plan-panel";
import { InlineAppPreview } from "@/components/ui/app-preview-panel";
import { SlideOutputPanel, type SlideOutput } from "./slide-output-panel";
import { TypingIndicator } from "./typing-indicator";
import { StreamingCursor } from "./streaming-cursor";
import { MessageAttachments } from "./message-attachments";
import { markdownComponents } from "./markdown-components";
import { CostIndicator } from "@/components/ui/cost-indicator";
import type {
    Message,
    FileAttachment,
    GeneratedImage,
    AgentEvent,
    Source,
} from "@/lib/types";
import { generatedImageToFileAttachment } from "@/lib/utils/streaming-helpers";
import type { TimestampedEvent } from "@/lib/stores/agent-progress-store";

// Metadata that can come from message (parsed or raw)
interface ParsedMetadata {
    model?: string;
    tokens?: number;
    images?: GeneratedImage[];
    agentEvents?: AgentEvent[];
}

// Memoized markdown plugin arrays at module level
const REMARK_PLUGINS = [remarkGfm, remarkMath];
const REHYPE_PLUGINS = [rehypeKatex];

// Stages that should not trigger progress panel rendering
const HIDDEN_STAGES = new Set(["thinking", "routing"]);

/**
 * Check if streaming events will produce visible stage groups in the progress panel
 */
function willEventsRenderProgressPanel(events: TimestampedEvent[] | undefined): boolean {
    if (!events || events.length === 0) return false;

    for (const event of events) {
        if (event.type === "stage") {
            const stageName = event.name;
            if (stageName && !HIDDEN_STAGES.has(stageName)) {
                return true;
            }
        } else if (event.type === "tool_call" || (event as AgentEvent).type === "browser_action") {
            return true;
        }
    }

    return false;
}

/**
 * Parse metadata from message, handling both object and JSON string formats
 */
function parseMetadata(
    metadata: Message["metadata"] | string | undefined
): ParsedMetadata | undefined {
    if (!metadata || typeof metadata !== "string") {
        return metadata as ParsedMetadata | undefined;
    }
    try {
        return JSON.parse(metadata) as ParsedMetadata;
    } catch {
        return undefined;
    }
}

/**
 * Extract task plans from skill_output events
 */
function extractTaskPlans(events: (TimestampedEvent | AgentEvent)[]): TaskPlan[] {
    const plans: TaskPlan[] = [];

    for (const event of events) {
        if (event.type !== "skill_output") continue;

        const skillEvent = event as AgentEvent;
        if (skillEvent.skill_id !== "task_planning" || !skillEvent.output) continue;

        const output = skillEvent.output as Record<string, unknown>;
        if (
            output.task_summary &&
            output.complexity_assessment &&
            Array.isArray(output.steps) &&
            Array.isArray(output.success_criteria)
        ) {
            plans.push(output as unknown as TaskPlan);
        }
    }

    return plans;
}

interface AppPreview {
    url: string;
    template?: string;
}

/**
 * Validate that a URL uses a safe protocol (http/https only).
 * Rejects javascript:, data:, blob:, and other dangerous protocols to prevent stored XSS.
 */
function isSafePreviewUrl(url: string): boolean {
    try {
        const parsed = new URL(url);
        return parsed.protocol === "http:" || parsed.protocol === "https:";
    } catch {
        return false;
    }
}

/**
 * Extract app preview URLs from tool results (app_start_server or app_builder skill)
 */
function extractAppPreviews(events: (TimestampedEvent | AgentEvent)[]): AppPreview[] {
    const previews: AppPreview[] = [];

    for (const event of events) {
        if (event.type === "tool_result") {
            const agentEvent = event as AgentEvent;
            const toolName = agentEvent.tool || agentEvent.name;
            const content = agentEvent.content;

            if (!content || typeof content !== "string") continue;

            try {
                const parsed = JSON.parse(content);

                // Direct app-related tool results
                if (toolName === "app_start_server" || toolName === "app_get_preview_url") {
                    if (parsed.preview_url && typeof parsed.preview_url === "string" && isSafePreviewUrl(parsed.preview_url)) {
                        previews.push({
                            url: parsed.preview_url,
                            template: parsed.template as string | undefined,
                        });
                    }
                }

                // invoke_skill tool result for app_builder skill
                if (toolName === "invoke_skill" && parsed.skill_id === "app_builder" && parsed.output) {
                    const output = parsed.output as Record<string, unknown>;
                    if (output.preview_url && typeof output.preview_url === "string" && isSafePreviewUrl(output.preview_url)) {
                        previews.push({
                            url: output.preview_url,
                            template: output.template as string | undefined,
                        });
                    }
                }
            } catch {
                // Not valid JSON, skip
            }
        }

        // Direct skill_output for app_builder skill
        if (event.type === "skill_output") {
            const skillEvent = event as AgentEvent;
            if (skillEvent.skill_id === "app_builder" && skillEvent.output) {
                const output = skillEvent.output as Record<string, unknown>;
                if (output.preview_url && typeof output.preview_url === "string" && isSafePreviewUrl(output.preview_url)) {
                    previews.push({
                        url: output.preview_url,
                        template: output.template as string | undefined,
                    });
                }
            }
        }

        // browser_stream events from app sandbox (persisted after streaming)
        if (event.type === "browser_stream") {
            const streamUrl = (event as unknown as Record<string, unknown>).stream_url as string | undefined;
            if (streamUrl && typeof streamUrl === "string" && isSafePreviewUrl(streamUrl)) {
                previews.push({ url: streamUrl });
            }
        }
    }

    // Deduplicate by URL
    const seen = new Set<string>();
    return previews.filter((p) => {
        if (seen.has(p.url)) return false;
        seen.add(p.url);
        return true;
    });
}

/**
 * Extract slide generation outputs from skill_output events
 */
function extractSlideOutputs(events: (TimestampedEvent | AgentEvent)[]): SlideOutput[] {
    return events
        .filter(
            (e) =>
                e.type === "skill_output" &&
                (e as AgentEvent).skill_id === "slide_generation"
        )
        .map((e) => (e as AgentEvent).output as SlideOutput)
        .filter((o) => o && o.download_url);
}

interface MessageBubbleProps {
    message: Message;
    onRegenerate?: () => void;
    isStreaming?: boolean;
    streamingEvents?: TimestampedEvent[];
    streamingSources?: Source[];
    agentType?: string;
}

export const MessageBubble = memo(function MessageBubble({
    message,
    onRegenerate,
    isStreaming = false,
    streamingEvents,
    streamingSources,
    agentType,
}: MessageBubbleProps): JSX.Element {
    const isUser = message.role === "user";
    const [copied, setCopied] = useState(false);
    const openPreview = usePreviewStore((state) => state.openPreview);
    const t = useTranslations("chat");

    // Parse metadata and extract data
    const parsedMetadata = useMemo(
        () => parseMetadata(message.metadata as Message["metadata"] | string | undefined),
        [message.metadata]
    );
    const agentEvents = parsedMetadata?.agentEvents;

    // Check if progress panel will actually render
    const willProgressPanelRender = useMemo(
        () => willEventsRenderProgressPanel(streamingEvents),
        [streamingEvents]
    );

    // Strip image placeholders from content - we render images separately
    const normalizedContent = message.content
        .replace(/!\[generated-image:\d+\]\(placeholder\)/g, "")
        .replace(/!\[generated-image:\d+\]/g, "");
    const hasVisibleContent = normalizedContent.trim().length > 0;

    // Convert generated images from metadata to previewable attachments
    const imageAttachments = useMemo(() => {
        const imgs = parsedMetadata?.images;
        if (!imgs || imgs.length === 0) return [];
        return imgs
            .map((img) => generatedImageToFileAttachment(img))
            .filter((f): f is FileAttachment => f !== null);
    }, [parsedMetadata?.images]);

    // Extract task plans from events
    const taskPlans = useMemo(
        () => extractTaskPlans(streamingEvents || agentEvents || []),
        [streamingEvents, agentEvents]
    );

    // Extract app preview URLs from events
    const appPreviews = useMemo(() => {
        const events = streamingEvents || agentEvents || [];
        if (events.length > 0) {
            return extractAppPreviews(events);
        }
        return [];
    }, [streamingEvents, agentEvents]);

    // Extract slide generation outputs from events
    const slideOutputs = useMemo(
        () => extractSlideOutputs(streamingEvents || agentEvents || []),
        [streamingEvents, agentEvents]
    );

    async function handleCopyMessage(): Promise<void> {
        try {
            await navigator.clipboard.writeText(message.content);
            setCopied(true);
            setTimeout(() => setCopied(false), 2000);
        } catch (err) {
            console.warn("Failed to copy message to clipboard:", err);
        }
    }

    function handleAttachmentClick(attachment: FileAttachment): void {
        openPreview(attachment);
    }

    if (isUser) {
        return (
            <div className="group py-4 flex justify-end">
                <div className="relative max-w-[90%] md:max-w-[75%] break-words">
                    <div
                        className={cn(
                            "relative px-4 py-3",
                            "bg-primary/10 text-foreground",
                            "rounded-2xl rounded-br-md"
                        )}
                    >
                        {message.content && (
                            <p className="text-sm leading-relaxed whitespace-pre-wrap">
                                {message.content}
                            </p>
                        )}
                        {message.attachments && message.attachments.length > 0 && (
                            <div className={cn(message.content && "mt-3")}>
                                <MessageAttachments
                                    attachments={message.attachments}
                                    onAttachmentClick={handleAttachmentClick}
                                />
                            </div>
                        )}
                    </div>
                </div>
            </div>
        );
    }

    return (
        <div className="group py-4 flex justify-start">
            <div className="w-full max-w-[90%] md:max-w-[75%] break-words">
                {/* Assistant header with icon and name */}
                <AssistantHeader />

                {/* Show typing indicator at the very beginning when streaming starts */}
                {isStreaming && !hasVisibleContent && !willProgressPanelRender && (
                    <TypingIndicator />
                )}

                {/* Show live agent progress inline during streaming - replaces typing indicator */}
                {isStreaming && willProgressPanelRender && streamingEvents && (
                    <TaskProgressPanel
                        events={streamingEvents}
                        sources={streamingSources}
                        isStreaming={isStreaming}
                        agentType={agentType}
                    />
                )}

                <div
                    className={cn(
                        "prose prose-neutral dark:prose-invert max-w-none",
                        "text-foreground/95",
                        "[&>*:first-child]:mt-0 [&>*:last-child]:mb-0"
                    )}
                >
                    {normalizedContent && (
                        <ReactMarkdown
                            remarkPlugins={REMARK_PLUGINS}
                            rehypePlugins={REHYPE_PLUGINS}
                            components={markdownComponents}
                        >
                            {normalizedContent}
                        </ReactMarkdown>
                    )}
                    {/* Show streaming cursor at the end of content */}
                    {isStreaming && message.content && <StreamingCursor />}
                </div>

                {/* Clickable image thumbnails — open in preview sidebar */}
                {imageAttachments.length > 0 && (
                    <div className="mt-4 flex flex-wrap gap-2">
                        {imageAttachments.map((file) => (
                            <button
                                key={file.id}
                                onClick={() => openPreview(file)}
                                className={cn(
                                    "group/img relative rounded-xl overflow-hidden",
                                    "border border-border/60 hover:border-primary/40",
                                    "transition-all duration-150",
                                    "w-48 h-48 bg-secondary/30"
                                )}
                            >
                                {file.previewUrl ? (
                                    <img
                                        src={file.previewUrl}
                                        alt={file.filename}
                                        className="w-full h-full object-cover"
                                    />
                                ) : (
                                    <span className="flex items-center justify-center w-full h-full">
                                        <ImageIcon className="w-8 h-8 text-muted-foreground/50" />
                                    </span>
                                )}
                                {/* Hover overlay */}
                                <span className={cn(
                                    "absolute inset-0 flex items-center justify-center",
                                    "bg-foreground/0 group-hover/img:bg-foreground/30",
                                    "transition-colors duration-150"
                                )}>
                                    <span className={cn(
                                        "text-xs font-medium text-white px-3 py-1.5 rounded-lg bg-foreground/60",
                                        "opacity-0 group-hover/img:opacity-100",
                                        "transition-opacity duration-150"
                                    )}>
                                        {t("viewImage")}
                                    </span>
                                </span>
                            </button>
                        ))}
                    </div>
                )}

                {/* Show task plans from skill_output events */}
                {taskPlans.length > 0 && (
                    <div className="mt-4 space-y-4">
                        {taskPlans.map((plan, index) => (
                            <TaskPlanPanel
                                key={`plan-${index}`}
                                plan={plan}
                                defaultExpanded={taskPlans.length === 1}
                            />
                        ))}
                    </div>
                )}

                {/* Show slide generation outputs */}
                {slideOutputs.length > 0 && (
                    <div className="mt-4 space-y-4">
                        {slideOutputs.map((output, index) => (
                            <SlideOutputPanel key={`slide-${index}`} output={output} />
                        ))}
                    </div>
                )}

                {/* Show app previews from tool results */}
                {appPreviews.length > 0 && (
                    <div className="mt-4 space-y-4">
                        {appPreviews.map((preview, index) => (
                            <InlineAppPreview
                                key={`app-preview-${index}`}
                                previewUrl={preview.url}
                                template={preview.template}
                            />
                        ))}
                    </div>
                )}

                {/* Show saved agent events (progress steps) if available - only when not streaming */}
                {!isStreaming && parsedMetadata?.agentEvents && parsedMetadata.agentEvents.length > 0 && (
                    <TaskProgressPanel
                        events={parsedMetadata.agentEvents}
                        isStreaming={false}
                        agentType={agentType}
                    />
                )}

                {/* Live token usage during streaming */}
                {isStreaming && streamingEvents && streamingEvents.length > 0 && (
                    <div className="mt-3 flex items-center justify-end">
                        <CostIndicator events={streamingEvents as AgentEvent[]} />
                    </div>
                )}

                {/* Action buttons for assistant message - only show when not streaming */}
                {!isStreaming && (
                    <MessageActions
                        copied={copied}
                        onCopy={handleCopyMessage}
                        onRegenerate={onRegenerate}
                        events={agentEvents}
                    />
                )}
            </div>
        </div>
    );
}, (prevProps, nextProps) => {
    // During streaming, throttle re-renders for small content changes
    if (nextProps.isStreaming || prevProps.isStreaming) {
        // Always re-render when streaming state changes
        if (prevProps.isStreaming !== nextProps.isStreaming) return false;
        // Always re-render when streamingEvents changes (e.g. skill_output arrives)
        if (prevProps.streamingEvents !== nextProps.streamingEvents) {
            const prevEvtLen = prevProps.streamingEvents?.length ?? 0;
            const nextEvtLen = nextProps.streamingEvents?.length ?? 0;
            if (prevEvtLen !== nextEvtLen) return false;
        }
        // Throttle small content-only changes
        const prevLen = prevProps.message.content?.length ?? 0;
        const nextLen = nextProps.message.content?.length ?? 0;
        if (Math.abs(nextLen - prevLen) < 5) {
            return true; // Skip re-render for tiny content changes
        }
        return false;
    }
    // For non-streaming messages, use default shallow comparison
    return (
        prevProps.message.id === nextProps.message.id &&
        prevProps.message.content === nextProps.message.content &&
        prevProps.message.role === nextProps.message.role &&
        prevProps.isStreaming === nextProps.isStreaming &&
        prevProps.streamingEvents === nextProps.streamingEvents &&
        prevProps.streamingSources === nextProps.streamingSources
    );
});

/**
 * Assistant header with logo and name
 */
function AssistantHeader(): JSX.Element {
    return (
        <div className="flex items-center gap-2 mb-3">
            <div className="w-5 h-5 flex items-center justify-center">
                <Image
                    src="/images/logo-light.svg"
                    alt="HyperAgent"
                    width={20}
                    height={20}
                    className="dark:hidden"
                />
                <Image
                    src="/images/logo-dark.svg"
                    alt="HyperAgent"
                    width={20}
                    height={20}
                    className="hidden dark:block"
                />
            </div>
            <span className="text-sm font-semibold text-foreground">
                HyperAgent
            </span>
        </div>
    );
}

interface MessageActionsProps {
    copied: boolean;
    onCopy: () => void;
    onRegenerate?: () => void;
    events?: AgentEvent[];
}

/**
 * Action buttons for assistant messages (copy, regenerate)
 */
function MessageActions({ copied, onCopy, onRegenerate, events }: MessageActionsProps): JSX.Element {
    const t = useTranslations("chat");

    return (
        <div className="mt-3 flex items-center gap-1">
            <button
                onClick={onCopy}
                className={cn(
                    "flex items-center gap-1.5",
                    "px-2.5 py-1.5",
                    "text-xs font-medium",
                    "rounded-lg",
                    "transition-colors",
                    copied
                        ? "text-success bg-success/10"
                        : "text-muted-foreground hover:text-foreground hover:bg-secondary"
                )}
            >
                {copied ? (
                    <>
                        <Check className="w-3.5 h-3.5" strokeWidth={2.5} />
                        <span>{t("copied")}</span>
                    </>
                ) : (
                    <>
                        <Copy className="w-3.5 h-3.5" />
                        <span>{t("copy")}</span>
                    </>
                )}
            </button>

            {onRegenerate && (
                <button
                    onClick={onRegenerate}
                    className={cn(
                        "flex items-center gap-1.5",
                        "px-2.5 py-1.5",
                        "text-xs font-medium",
                        "rounded-lg",
                        "transition-colors",
                        "text-muted-foreground hover:text-foreground hover:bg-secondary"
                    )}
                >
                    <RotateCcw className="w-3.5 h-3.5" />
                    <span>{t("regenerate")}</span>
                </button>
            )}

            {events && events.length > 0 && (
                <>
                    <div className="flex-1" />
                    <CostIndicator events={events} />
                </>
            )}
        </div>
    );
}
