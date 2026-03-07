import type { AgentEvent, Source, InterruptEvent } from "@/lib/types";
import type { ComputerMode } from "@/lib/stores/computer-store";
import type { SlideOutput } from "@/components/chat/slide-output-panel";
import type { ExternalFileEntry } from "@/lib/stores/computer-store";
import {
    type StreamEvent,
    type StreamingContext,
    getEventKey,
    createTimestampedEvent,
    mergeTokenContent,
    parseSourceFromEvent,
    parseImageFromEvent,
    filterEventsForSaving,
    isSearchTool,
    generatedImageToFileAttachment,
    fileAttachmentToExternalEntry,
} from "@/lib/utils/streaming-helpers";
import { reduceStreamEvent, type StreamReducerHandlers } from "@/lib/utils/stream-reducer";
import type { TimestampedEvent } from "@/lib/stores/agent-progress-store";
import { getTranslatedSkillName } from "@/lib/utils/skill-i18n";

export interface StreamHandlerDeps {
    ctx: StreamingContext;
    updateStreamingContent: (content: string) => void;
    flushTokenBatch: () => void;
    updateAgentStage: (description: string | null) => void;
    addAgentEvent: (event: AgentEvent) => void;
    addExecutionEvent: (event: AgentEvent) => void;
    setStreamingEvents: React.Dispatch<React.SetStateAction<TimestampedEvent[]>>;
    setStreamingSources: React.Dispatch<React.SetStateAction<Source[]>>;
    setActiveInterrupt: React.Dispatch<React.SetStateAction<InterruptEvent | null>>;
    setCurrentCommand: (command: string | null, cwd?: string) => void;
    addTerminalLine: (line: { type: "command" | "output" | "error"; content: string; cwd?: string }) => void;
    smartOpenComputer: (mode: ComputerMode, force?: boolean) => void;
    setWorkspaceContext: (sandboxType: "execution" | "app", sandboxId: string, conversationId: string) => void;
    handleWorkspaceUpdate: (event: {
        type: "workspace_update";
        operation: "create" | "modify" | "delete";
        path: string;
        name: string;
        is_directory: boolean;
        size: number | undefined;
        sandbox_type: "execution" | "app";
        sandbox_id: string;
        timestamp: number | undefined;
    }) => void;
    refreshFiles: (paths: string[]) => void;
    openFileInBrowser: (entry: ExternalFileEntry) => void;
    workspaceSandboxId: string | null;
    conversationId: string;
    hasOpenedTerminalRef: { current: boolean };
    tChat: (key: string, values?: Record<string, string | number>) => string;
    tSkills: (key: string) => string;
    getTranslatedToolName: (toolName: string) => string;
}

export function createStreamHandlers(deps: StreamHandlerDeps): StreamReducerHandlers {
    const {
        ctx,
        updateStreamingContent,
        flushTokenBatch,
        updateAgentStage,
        addAgentEvent,
        addExecutionEvent,
        setStreamingEvents,
        setStreamingSources,
        setActiveInterrupt,
        setCurrentCommand,
        addTerminalLine,
        smartOpenComputer,
        setWorkspaceContext,
        handleWorkspaceUpdate,
        refreshFiles,
        openFileInBrowser,
        workspaceSandboxId,
        conversationId,
        hasOpenedTerminalRef,
        tChat,
        tSkills,
        getTranslatedToolName,
    } = deps;

    const handleTokenContent = (tokenContent: string): void => {
        ctx.fullContent = mergeTokenContent(ctx.fullContent, tokenContent);
        updateStreamingContent(ctx.fullContent);
    };

    const addStreamingEvent = (event: AgentEvent): void => {
        const eventKey = getEventKey(event);
        if (ctx.seenEventKeys.has(eventKey)) return;
        ctx.seenEventKeys.add(eventKey);

        addAgentEvent(event);
        addExecutionEvent(event);
        setStreamingEvents(prev => [...prev, createTimestampedEvent(event)]);
        ctx.collectedEvents.push(event);
    };

    const addStreamingSource = (source: Source): void => {
        ctx.collectedSources.push(source);
        setStreamingSources(prev => [...prev, source]);
    };

    return {
        onToken: (token: string) => handleTokenContent(token),
        onStage: (event: StreamEvent) => {
            flushTokenBatch();
            const eventData = typeof event.data === "object" && event.data !== null ? event.data : null;
            const stageDescription = event.description || eventData?.description || null;
            updateAgentStage(stageDescription);
            addStreamingEvent(event);
        },
        onToolCall: (event: StreamEvent) => {
            const eventData = typeof event.data === "object" && event.data !== null ? event.data : null;
            const toolName = event.tool || eventData?.tool || "tool";
            const args = event.args || eventData?.args || {};
            const rawQuery = typeof args === "object" && args !== null ? (args as Record<string, unknown>).query : undefined;
            const queryArg = typeof rawQuery === "string" ? rawQuery : "web";
            if (isSearchTool(toolName)) {
                updateAgentStage(tChat("agent.searching", { query: queryArg }));
            } else {
                updateAgentStage(tChat("agent.executing", { tool: getTranslatedToolName(toolName) }));
            }
            addStreamingEvent(event);
        },
        onToolResult: (event: StreamEvent) => addStreamingEvent(event),
        onRouting: (event: StreamEvent) => addStreamingEvent(event),
        onHandoff: (event: StreamEvent) => {
            const target = event.target || "";
            updateAgentStage(tChat("agent.handoffTo", { target }));
            addStreamingEvent(event);
        },
        onSource: (event: StreamEvent) => {
            addStreamingEvent(event);
            const newSource = parseSourceFromEvent(event, ctx.collectedSources.length);
            if (newSource) addStreamingSource(newSource);
        },
        onCodeResult: (event: StreamEvent) => addStreamingEvent(event),
        onImage: (event: StreamEvent) => {
            const imageData = parseImageFromEvent(event, ctx.collectedImages.length);
            if (imageData) {
                const isDuplicate = ctx.collectedImages.some((img) => img.index === imageData.index);
                if (!isDuplicate) {
                    ctx.collectedImages.push(imageData);
                    ctx.collectedEvents.push(event);
                    const previewFile = generatedImageToFileAttachment(imageData);
                    if (previewFile) {
                        const entry = fileAttachmentToExternalEntry(previewFile, "generated-image");
                        if (previewFile.previewUrl?.startsWith("data:")) {
                            entry.base64Data = previewFile.previewUrl;
                        }
                        openFileInBrowser(entry);
                    }
                }
            }
        },
        onBrowserStream: (event: StreamEvent) => addStreamingEvent(event),
        onBrowserActionStage: (event: AgentEvent) => addStreamingEvent(event),
        onTerminalCommand: (event: StreamEvent) => {
            const command = event.command as string;
            const cwd = event.cwd as string | undefined;
            if (command) {
                setCurrentCommand(command, cwd);
                addTerminalLine({ type: "command", content: command, cwd });
                if (!hasOpenedTerminalRef.current) {
                    smartOpenComputer("terminal", false);
                    hasOpenedTerminalRef.current = true;
                }
                addAgentEvent(event);
            }
        },
        onTerminalOutput: (event: StreamEvent) => {
            const output = (event.content || event.data) as string;
            if (output) addTerminalLine({ type: "output", content: output });
        },
        onTerminalError: (event: StreamEvent) => {
            const errorOutput = (event.content || event.data) as string;
            if (errorOutput) addTerminalLine({ type: "error", content: errorOutput });
        },
        onTerminalComplete: (event: StreamEvent) => {
            const exitCode = event.exit_code as number | undefined;
            setCurrentCommand(null);
            if (exitCode !== undefined && exitCode !== 0) {
                addTerminalLine({ type: "error", content: `Exit code: ${exitCode}` });
            }
            addAgentEvent(event);
        },
        onSkillOutput: (event: StreamEvent) => {
            const skillId = event.skill_id as string;
            addStreamingEvent(event);
            updateAgentStage(tChat("agent.skillCompleted", { skill: getTranslatedSkillName(skillId, skillId, tSkills) }));

            const output = event.output as Record<string, unknown> | undefined;
            if (!output) return;

            if (skillId === "slide_generation") {
                const slideData = output as unknown as SlideOutput;
                if (slideData.download_url) {
                    const slideEntry: ExternalFileEntry = {
                        id: `slide-${Date.now()}`,
                        name: slideData.title || "Presentation.pptx",
                        source: "generated-slide",
                        contentType: "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                        fileSize: 0,
                        downloadUrl: slideData.download_url,
                        slideOutput: slideData,
                        timestamp: Date.now(),
                    };
                    openFileInBrowser(slideEntry);
                }
            } else if (typeof output.download_url === "string" && output.download_url) {
                // Generic artifact download for any skill that produces a file
                const nameMap: Record<string, string> = {
                    task_planning: "Plan.md",
                    code_generation: "Code",
                    deep_research: "Report.md",
                    data_analysis: "Analysis.md",
                    web_research: "Research.md",
                };
                const contentTypeMap: Record<string, string> = {
                    task_planning: "text/markdown",
                    code_generation: "text/plain",
                    deep_research: "text/markdown",
                    data_analysis: "text/markdown",
                    web_research: "text/markdown",
                };
                const fileName = nameMap[skillId] || "Artifact";
                const contentType = contentTypeMap[skillId] || "text/markdown";
                const artifactEntry: ExternalFileEntry = {
                    id: `artifact-${skillId}-${Date.now()}`,
                    name: fileName,
                    source: "generated-artifact",
                    contentType,
                    fileSize: 0,
                    downloadUrl: output.download_url as string,
                    timestamp: Date.now(),
                };
                openFileInBrowser(artifactEntry);
            }
        },
        onWorkspaceUpdate: (event: StreamEvent) => {
            const workspaceEvent = {
                type: "workspace_update" as const,
                operation: event.operation as "create" | "modify" | "delete",
                path: event.path as string,
                name: event.name as string,
                is_directory: event.is_directory as boolean,
                size: event.size as number | undefined,
                sandbox_type: event.sandbox_type as "execution" | "app",
                sandbox_id: event.sandbox_id as string,
                timestamp: event.timestamp as number | undefined,
            };
            if (!workspaceSandboxId) {
                setWorkspaceContext(workspaceEvent.sandbox_type, workspaceEvent.sandbox_id, conversationId);
            }
            handleWorkspaceUpdate(workspaceEvent);
            refreshFiles([workspaceEvent.path]);
            if (
                (workspaceEvent.operation === "create" || workspaceEvent.operation === "modify")
                && workspaceEvent.sandbox_type === "app"
            ) {
                smartOpenComputer("file", false);
            }
        },
        onPlanOverview: (event: StreamEvent) => {
            flushTokenBatch();
            const totalSteps = (event as unknown as Record<string, unknown>).total_steps as number || 0;
            updateAgentStage(tChat("agent.planning", { steps: String(totalSteps) }));
            addStreamingEvent(event);
        },
        onPlanStep: (event: StreamEvent) => {
            const raw = event as unknown as Record<string, unknown>;
            const action = (raw.action as string) || "";
            const stepNum = (raw.step_number as number) || 0;
            const total = (raw.total_steps as number) || 0;
            updateAgentStage(tChat("agent.executingStep", { step: String(stepNum), total: String(total), action }));
            addStreamingEvent(event);
        },
        onPlanStepCompleted: (event: StreamEvent) => addStreamingEvent(event),
        onStepActivity: (event: StreamEvent) => {
            const title = (event.step_title as string) || "";
            const status = (event.status as string) || "";
            if (status === "running") {
                updateAgentStage(title);
            }
            addStreamingEvent(event);
        },
        onTodoUpdate: (event: StreamEvent) => addStreamingEvent(event),
        onInterrupt: (interruptEvent: InterruptEvent, rawEvent: StreamEvent) => {
            setActiveInterrupt(interruptEvent);
            addStreamingEvent(rawEvent);
        },
        onUsage: (event: StreamEvent) => addStreamingEvent(event),
        onError: (event: StreamEvent) => {
            const errorData = typeof event.data === "string" ? event.data : "Unknown error";
            ctx.fullContent = tChat("agent.error", { error: errorData });
            updateStreamingContent(ctx.fullContent);
            addAgentEvent(event);
        },
    };
}

export interface ReadSSEStreamResult {
    streamComplete: boolean;
}

export async function readSSEStream(
    reader: ReadableStreamDefaultReader<Uint8Array>,
    handlers: StreamReducerHandlers,
): Promise<ReadSSEStreamResult> {
    const decoder = new TextDecoder();
    let buffer = "";
    let streamComplete = false;

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
            if (!line.trim()) continue;

            if (line.startsWith("data: ")) {
                const jsonStr = line.slice(6).trim();
                if (jsonStr === "[DONE]" || !jsonStr) continue;

                try {
                    const event = JSON.parse(jsonStr) as StreamEvent;
                    const reduced = reduceStreamEvent(event, handlers);
                    if (reduced.shouldBreak) {
                        streamComplete = true;
                        break;
                    }
                } catch (e) {
                    console.error("[SSE Parse Error]", e, "Line:", line);
                }
            } else if (line.startsWith("event: ")) {
                continue;
            }
        }
        if (streamComplete) break;
    }

    return { streamComplete };
}

export function buildSavedMessageMetadata(ctx: StreamingContext) {
    const savedEvents = filterEventsForSaving(ctx.collectedEvents);
    return {
        ...(ctx.collectedImages.length ? { images: ctx.collectedImages } : {}),
        ...(savedEvents.length ? { agentEvents: savedEvents } : {}),
    };
}
