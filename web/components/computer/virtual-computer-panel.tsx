"use client";

import React, { useEffect, useCallback, useState, useMemo } from "react";
import { cn } from "@/lib/utils";
import { useComputerStore, COMPUTER_PANEL_MIN_WIDTH, COMPUTER_PANEL_MAX_WIDTH } from "@/lib/stores/computer-store";
import type { ComputerMode, TimelineEventType, TimelineEvent } from "@/lib/stores/computer-store";
import { useAgentProgressStore } from "@/lib/stores/agent-progress-store";
import { ComputerTabBar } from "./computer-tab-bar";
import { ComputerTerminalView } from "./computer-terminal-view";
import { ComputerBrowserView } from "./computer-browser-view";
import { ComputerFileView } from "./computer-file-view";
import { ComputerErrorBoundary } from "./computer-error-boundary";
import { ChevronLeft, ChevronRight } from "lucide-react";
import { useTranslations } from "next-intl";

// Module-level empty array constant for referential stability
const EMPTY_ARRAY: never[] = [];
const EMPTY_TIMELINE: TimelineEvent[] = [];

export function VirtualComputerPanel() {
    const t = useTranslations("computer");
    const getModeAriaLabel = useCallback((mode: ComputerMode) => {
        switch (mode) {
            case "terminal":
                return t("terminal");
            case "browser":
                return t("browser");
            case "file":
                return t("files");
            default:
                return mode;
        }
    }, [t]);

    // Global UI state
    const isOpen = useComputerStore((state) => state.isOpen);
    const panelWidth = useComputerStore((state) => state.panelWidth);
    const activeMode = useComputerStore((state) => state.activeMode);

    // Per-conversation state via direct state access (stable references)
    const terminalLines = useComputerStore((state) => {
        const id = state.activeConversationId;
        return id ? state.conversationStates[id]?.terminalLines ?? EMPTY_ARRAY : EMPTY_ARRAY;
    });
    const currentCommand = useComputerStore((state) => {
        const id = state.activeConversationId;
        return id ? state.conversationStates[id]?.currentCommand ?? null : null;
    });
    const currentCwd = useComputerStore((state) => {
        const id = state.activeConversationId;
        return id ? state.conversationStates[id]?.currentCwd ?? "/home/user" : "/home/user";
    });
    const browserStream = useComputerStore((state) => {
        const id = state.activeConversationId;
        return id ? state.conversationStates[id]?.browserStream ?? null : null;
    });
    const isLive = useComputerStore((state) => {
        const id = state.activeConversationId;
        return id ? state.conversationStates[id]?.isLive ?? true : true;
    });
    const currentStep = useComputerStore((state) => {
        const id = state.activeConversationId;
        return id ? state.conversationStates[id]?.currentStep ?? 0 : 0;
    });
    const totalSteps = useComputerStore((state) => {
        const id = state.activeConversationId;
        return id ? state.conversationStates[id]?.totalSteps ?? 0 : 0;
    });
    const timeline = useComputerStore((state) => {
        const id = state.activeConversationId;
        return id ? state.conversationStates[id]?.timeline ?? EMPTY_TIMELINE : EMPTY_TIMELINE;
    });

    // Compute visible data slices based on timeline position
    const visibleCounts = useMemo(() => {
        if (isLive) {
            return {
                terminal: terminalLines.length,
                browser: true,
                file: true,
            };
        }

        const maxIndices: Record<TimelineEventType, number> = {
            terminal: -1,
            plan: -1,
            browser: -1,
            file: -1,
        };

        for (let i = 0; i < Math.min(currentStep, timeline.length); i++) {
            const event = timeline[i];
            if (event.dataIndex > maxIndices[event.type]) {
                maxIndices[event.type] = event.dataIndex;
            }
        }

        return {
            terminal: maxIndices.terminal + 1,
            browser: maxIndices.browser >= 0,
            file: maxIndices.file >= 0,
        };
    }, [isLive, currentStep, timeline, terminalLines.length]);

    const visibleTerminalLines = useMemo(
        () => isLive ? terminalLines : terminalLines.slice(0, visibleCounts.terminal),
        [isLive, terminalLines, visibleCounts.terminal]
    );

    // Actions (stable references via getState)
    const closePanel = useComputerStore.getState().closePanel;
    const setModeByUser = useComputerStore.getState().setModeByUser;
    const setPanelWidth = useComputerStore.getState().setPanelWidth;
    const setBrowserStream = useComputerStore.getState().setBrowserStream;
    const setCurrentStep = useComputerStore.getState().setCurrentStep;
    const setIsLive = useComputerStore.getState().setIsLive;
    const nextStep = useComputerStore.getState().nextStep;
    const prevStep = useComputerStore.getState().prevStep;
    const clearTerminal = useComputerStore.getState().clearTerminal;
    const addTerminalLine = useComputerStore.getState().addTerminalLine;
    const setMode = useComputerStore.getState().setMode;

    const activeProgress = useAgentProgressStore((state) => state.activeProgress);
    const setAgentBrowserStream = useAgentProgressStore.getState().setBrowserStream;

    const [isResizing, setIsResizing] = useState(false);
    const [isDesktop, setIsDesktop] = useState(false);

    // Check if we're on desktop (lg breakpoint = 1024px)
    useEffect(() => {
        const mq = window.matchMedia("(min-width: 1024px)");
        setIsDesktop(mq.matches);
        const handler = (e: MediaQueryListEvent) => setIsDesktop(e.matches);
        mq.addEventListener("change", handler);
        return () => mq.removeEventListener("change", handler);
    }, []);

    // Sync browser stream from agent progress store
    const agentStreamUrl = activeProgress?.browserStream?.streamUrl ?? null;
    const agentSandboxId = activeProgress?.browserStream?.sandboxId ?? null;
    useEffect(() => {
        if (activeProgress?.browserStream && agentStreamUrl && agentSandboxId) {
            setBrowserStream(activeProgress.browserStream);
        }
    }, [agentStreamUrl, agentSandboxId, activeProgress?.browserStream, setBrowserStream]);

    // Handle close
    const handleClose = useCallback(() => {
        closePanel();
        setAgentBrowserStream(null);
    }, [closePanel, setAgentBrowserStream]);

    // Resize handlers
    const startResizing = useCallback((e: React.MouseEvent) => {
        e.preventDefault();
        setIsResizing(true);
    }, []);

    const stopResizing = useCallback(() => {
        setIsResizing(false);
    }, []);

    const resize = useCallback(
        (e: MouseEvent) => {
            if (isResizing) {
                const newWidth = window.innerWidth - e.clientX;
                setPanelWidth(newWidth);
            }
        },
        [isResizing, setPanelWidth]
    );

    // Handle keyboard shortcuts
    useEffect(() => {
        const handleKeyDown = (e: KeyboardEvent) => {
            if (e.key === "Escape" && isOpen) {
                if (document.fullscreenElement) return;
                const fullscreenOverlay = document.querySelector('.fixed.inset-0.z-\\[100\\]');
                if (fullscreenOverlay) return;
                handleClose();
            }
        };

        window.addEventListener("keydown", handleKeyDown);
        return () => window.removeEventListener("keydown", handleKeyDown);
    }, [isOpen, handleClose]);

    // Add/remove event listeners for resize
    useEffect(() => {
        if (isResizing) {
            document.addEventListener("mousemove", resize);
            document.addEventListener("mouseup", stopResizing);
            document.body.style.cursor = "col-resize";
            document.body.style.userSelect = "none";
        }

        return () => {
            document.removeEventListener("mousemove", resize);
            document.removeEventListener("mouseup", stopResizing);
            document.body.style.cursor = "";
            document.body.style.userSelect = "";
        };
    }, [isResizing, resize, stopResizing]);

    // Go to live mode
    const handleGoLive = useCallback(() => {
        setIsLive(true);
    }, [setIsLive]);

    // Terminal command handler
    const handleSendTerminalCommand = useCallback(async (command: string) => {
        if (!command.trim()) return;

        const cs = useComputerStore.getState();
        const id = cs.activeConversationId;
        if (!id) return;
        const conv = cs.conversationStates[id];
        if (!conv?.workspaceTaskId) return;

        addTerminalLine({ type: "command", content: command, cwd: conv.currentCwd });

        try {
            const response = await fetch(`/api/v1/sandbox/exec`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    command,
                    task_id: conv.workspaceTaskId,
                    sandbox_type: conv.workspaceSandboxType || "execution",
                }),
            });
            const data = await response.json();

            if (data.stdout) {
                addTerminalLine({ type: "output", content: data.stdout });
            }
            if (data.stderr) {
                addTerminalLine({ type: "error", content: data.stderr });
            }
        } catch (error) {
            addTerminalLine({ type: "error", content: `Error: ${error}` });
        }
    }, [addTerminalLine]);

    // Replay bar: auto-switch view when scrubbing
    const handleStepChange = useCallback((step: number) => {
        setCurrentStep(step);
        if (step > 0 && step <= timeline.length) {
            const event = timeline[step - 1];
            if (event) {
                const viewMode = event.type === "browser" ? "browser" as const
                    : event.type === "terminal" ? "terminal" as const
                    : "file" as const;
                setMode(viewMode);
            }
        }
    }, [setCurrentStep, timeline, setMode]);

    // Helper to render a view for a given mode
    const renderViewForMode = useCallback((mode: ComputerMode) => {
        switch (mode) {
            case "terminal":
                return (
                    <ComputerTerminalView
                        lines={visibleTerminalLines}
                        isLive={isLive}
                        currentCommand={currentCommand}
                        currentCwd={currentCwd}
                        onClear={clearTerminal}
                        onSendCommand={handleSendTerminalCommand}
                        className="flex-1"
                    />
                );
            case "browser":
                return (
                    <ComputerBrowserView
                        stream={browserStream}
                        className="flex-1"
                    />
                );
            case "file":
                return (
                    <ComputerFileView className="flex-1" />
                );
            default:
                return null;
        }
    }, [visibleTerminalLines, isLive, currentCommand, currentCwd, clearTerminal, handleSendTerminalCommand, browserStream]);

    if (!isOpen) return null;

    return (
        <>
            {/* Mobile backdrop */}
            <div
                className={cn(
                    "fixed inset-0 bg-black/40 z-40 lg:hidden transition-opacity",
                    isOpen ? "opacity-100" : "opacity-0 pointer-events-none"
                )}
                onClick={handleClose}
                aria-hidden="true"
            />

            {/* Panel */}
            <div
                className={cn(
                    "fixed right-0 top-0 bottom-0 z-50 flex flex-col",
                    "bg-background border-l border-border",
                    "w-full lg:w-auto",
                    !isDesktop && "pt-safe pb-safe",
                    !isResizing && "transition-transform duration-300",
                    isOpen ? "translate-x-0" : "translate-x-full"
                )}
                style={{
                    width: isDesktop ? panelWidth : "100%",
                }}
                role="complementary"
                aria-label={t("panelTitle")}
            >
                {/* Resize handle (left edge) */}
                <div
                    className={cn(
                        "absolute top-0 left-0 w-1 h-full cursor-col-resize z-10",
                        "hover:bg-primary/50 active:bg-primary/70",
                        "transition-colors duration-150 hidden lg:block",
                        isResizing && "bg-primary/70"
                    )}
                    onMouseDown={startResizing}
                    role="separator"
                    aria-orientation="vertical"
                    aria-valuenow={panelWidth}
                    aria-valuemin={COMPUTER_PANEL_MIN_WIDTH}
                    aria-valuemax={COMPUTER_PANEL_MAX_WIDTH}
                    aria-label={t("resizeHandle")}
                    tabIndex={0}
                />

                {/* Tab bar */}
                <ComputerTabBar
                    activeMode={activeMode}
                    onModeChange={setModeByUser}
                    onClose={handleClose}
                />

                {/* View area with error boundary */}
                <ComputerErrorBoundary
                    translations={{
                        title: t("errorBoundary.title"),
                        maxRetries: t("errorBoundary.maxRetries"),
                        retry: (count: number) => t("errorBoundary.retry", { count }),
                        fallbackErrorMessage: t("errorBoundary.fallbackMessage"),
                    }}
                >
                    <div
                        className="flex-1 overflow-hidden flex flex-col relative"
                        role="tabpanel"
                        aria-label={getModeAriaLabel(activeMode)}
                    >
                        {renderViewForMode(activeMode)}

                        {/* Floating live indicator */}
                        {isLive && totalSteps > 0 && (
                            <div className="absolute bottom-2 right-2 z-10">
                                <button
                                    className={cn(
                                        "flex items-center gap-1 px-2 py-1 rounded-full",
                                        "bg-primary/10 text-primary text-xs font-medium",
                                        "focus-visible:ring-2 focus-visible:ring-primary focus-visible:outline-none"
                                    )}
                                    onClick={handleGoLive}
                                    aria-label={t("live")}
                                >
                                    <span className="w-1.5 h-1.5 rounded-full bg-primary animate-pulse" />
                                    {t("live")}
                                </button>
                            </div>
                        )}
                    </div>
                </ComputerErrorBoundary>

                {/* Conditional replay bar (only when not live) */}
                {!isLive && totalSteps > 0 && (
                    <div className="flex items-center gap-1.5 px-3 h-9 border-t border-border shrink-0 bg-background">
                        {/* Previous step */}
                        <button
                            className={cn(
                                "h-7 w-7 inline-flex items-center justify-center rounded-md",
                                "text-muted-foreground hover:text-foreground hover:bg-secondary/50 transition-colors",
                                "focus-visible:ring-2 focus-visible:ring-primary focus-visible:outline-none",
                                "disabled:opacity-40 disabled:pointer-events-none"
                            )}
                            onClick={prevStep}
                            disabled={currentStep <= 0}
                            title={t("previousStep")}
                            aria-label={t("previousStep")}
                        >
                            <ChevronLeft className="w-3.5 h-3.5" />
                        </button>

                        {/* Range slider */}
                        <div className="flex-1 relative h-4 flex items-center">
                            <div className="absolute inset-x-0 h-1 rounded-full bg-border" />
                            {totalSteps > 0 && (
                                <div
                                    className="absolute left-0 h-1 rounded-full bg-foreground/30"
                                    style={{ width: `${(currentStep / totalSteps) * 100}%` }}
                                />
                            )}
                            <input
                                type="range"
                                min={0}
                                max={totalSteps}
                                value={currentStep}
                                onChange={(e) => handleStepChange(parseInt(e.target.value, 10))}
                                aria-label={t("stepOf", { current: currentStep, total: totalSteps })}
                                aria-valuemin={0}
                                aria-valuemax={totalSteps}
                                aria-valuenow={currentStep}
                                className={cn(
                                    "absolute inset-0 w-full h-full opacity-0 cursor-pointer z-[2]",
                                    "[&::-webkit-slider-thumb]:appearance-none",
                                    "[&::-webkit-slider-thumb]:w-3",
                                    "[&::-webkit-slider-thumb]:h-3",
                                    "[&::-moz-range-thumb]:w-3",
                                    "[&::-moz-range-thumb]:h-3"
                                )}
                            />
                            {totalSteps > 0 && (
                                <div
                                    className="absolute w-2.5 h-2.5 rounded-full bg-foreground border-2 border-background -translate-x-1/2 z-[3] pointer-events-none"
                                    style={{ left: `${(currentStep / totalSteps) * 100}%` }}
                                />
                            )}
                        </div>

                        {/* Next step */}
                        <button
                            className={cn(
                                "h-7 w-7 inline-flex items-center justify-center rounded-md",
                                "text-muted-foreground hover:text-foreground hover:bg-secondary/50 transition-colors",
                                "focus-visible:ring-2 focus-visible:ring-primary focus-visible:outline-none",
                                "disabled:opacity-40 disabled:pointer-events-none"
                            )}
                            onClick={nextStep}
                            disabled={currentStep >= totalSteps}
                            title={t("nextStep")}
                            aria-label={t("nextStep")}
                        >
                            <ChevronRight className="w-3.5 h-3.5" />
                        </button>

                        {/* Step counter */}
                        <span className="text-xs text-muted-foreground tabular-nums min-w-[3ch] text-center">
                            {currentStep}/{totalSteps}
                        </span>

                        {/* Go to Live button */}
                        <button
                            onClick={handleGoLive}
                            className={cn(
                                "flex items-center gap-1 px-1.5 py-0.5 rounded",
                                "text-xs text-muted-foreground hover:text-foreground",
                                "hover:bg-secondary transition-colors",
                                "focus-visible:ring-2 focus-visible:ring-primary focus-visible:outline-none"
                            )}
                            title={t("goToLive")}
                            aria-label={t("goToLive")}
                        >
                            <span className="w-1.5 h-1.5 rounded-full bg-muted-foreground/50" />
                            {t("live")}
                        </button>
                    </div>
                )}
            </div>
        </>
    );
}
