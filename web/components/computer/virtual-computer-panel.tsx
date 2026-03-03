"use client";

import React, { useEffect, useCallback, useState, useMemo } from "react";
import { cn } from "@/lib/utils";
import { useComputerStore, COMPUTER_PANEL_MIN_WIDTH, COMPUTER_PANEL_MAX_WIDTH } from "@/lib/stores/computer-store";
import type { ComputerMode, TimelineEventType, TimelineEvent, TaskPlan } from "@/lib/stores/computer-store";
import { useAgentProgressStore } from "@/lib/stores/agent-progress-store";
import { ComputerPanelHeader } from "./computer-panel-header";
import { ComputerTerminalView } from "./computer-terminal-view";
import { ComputerBrowserView } from "./computer-browser-view";
import { ComputerFileView } from "./computer-file-view";
import { ComputerErrorBoundary } from "./computer-error-boundary";
import { ChevronLeft, ChevronRight, ChevronDown, ChevronUp, Check, Loader2, Circle } from "lucide-react";
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
        return id ? state.conversationStates[id]?.isLive ?? false : false;
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
    const taskPlan = useComputerStore((state) => {
        const id = state.activeConversationId;
        return id ? state.conversationStates[id]?.taskPlan ?? null : null;
    }) as TaskPlan | null;

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
    const setCurrentStep = useComputerStore.getState().setCurrentStep;
    const setIsLive = useComputerStore.getState().setIsLive;
    const nextStep = useComputerStore.getState().nextStep;
    const prevStep = useComputerStore.getState().prevStep;
    const setMode = useComputerStore.getState().setMode;

    const activeProgress = useAgentProgressStore((state) => state.activeProgress);
    const setAgentBrowserStream = useAgentProgressStore.getState().setBrowserStream;

    const [isResizing, setIsResizing] = useState(false);
    const [isDesktop, setIsDesktop] = useState(false);
    const [stepsExpanded, setStepsExpanded] = useState(false);

    // Check if we're on desktop (lg breakpoint = 1024px)
    useEffect(() => {
        const mq = window.matchMedia("(min-width: 1024px)");
        setIsDesktop(mq.matches);
        const handler = (e: MediaQueryListEvent) => setIsDesktop(e.matches);
        mq.addEventListener("change", handler);
        return () => mq.removeEventListener("change", handler);
    }, []);

    // Browser stream sync is handled by agent-progress-store's addEvent handler
    // (browser_stream events call openWithBrowser on computer-store directly).

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
    }, [visibleTerminalLines, isLive, currentCommand, currentCwd, browserStream]);

    // Activity description from agent progress
    const activityDescription = activeProgress?.currentStageDescription ?? null;

    // Current running or last completed step from taskPlan
    const currentPlanStep = useMemo(() => {
        if (!taskPlan?.steps?.length) return null;
        const running = taskPlan.steps.find((s) => s.status === "running");
        if (running) return running;
        // Find last completed
        const completed = [...taskPlan.steps].reverse().find((s) => s.status === "completed");
        return completed ?? taskPlan.steps[0];
    }, [taskPlan]);

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

                {/* Panel header (title + activity status) */}
                <ComputerPanelHeader
                    activeMode={activeMode}
                    onModeChange={setModeByUser}
                    onClose={handleClose}
                    activityDescription={activityDescription}
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
                    </div>
                </ComputerErrorBoundary>

                {/* Always-visible timeline scrubber */}
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
                        disabled={isLive || currentStep <= 0}
                        title={t("previousStep")}
                        aria-label={t("previousStep")}
                    >
                        <ChevronLeft className="w-3.5 h-3.5" />
                    </button>

                    {/* Range slider */}
                    <div className="flex-1 relative h-4 flex items-center">
                        <div className="absolute inset-x-0 h-0.5 rounded-full bg-border" />
                        {totalSteps > 0 && (
                            <div
                                className="absolute left-0 h-0.5 rounded-full bg-info"
                                style={{ width: `${isLive ? 100 : (currentStep / totalSteps) * 100}%` }}
                            />
                        )}
                        <input
                            type="range"
                            min={0}
                            max={totalSteps}
                            value={isLive ? totalSteps : currentStep}
                            onChange={(e) => {
                                const val = parseInt(e.target.value, 10);
                                if (val >= totalSteps) {
                                    handleGoLive();
                                } else {
                                    setIsLive(false);
                                    handleStepChange(val);
                                }
                            }}
                            aria-label={t("stepOf", { current: isLive ? totalSteps : currentStep, total: totalSteps })}
                            aria-valuemin={0}
                            aria-valuemax={totalSteps}
                            aria-valuenow={isLive ? totalSteps : currentStep}
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
                                className="absolute w-2.5 h-2.5 rounded-full bg-info border-2 border-background -translate-x-1/2 z-[3] pointer-events-none"
                                style={{ left: `${isLive ? 100 : (currentStep / totalSteps) * 100}%` }}
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
                        disabled={isLive || currentStep >= totalSteps}
                        title={t("nextStep")}
                        aria-label={t("nextStep")}
                    >
                        <ChevronRight className="w-3.5 h-3.5" />
                    </button>

                    {/* Live indicator / button */}
                    <button
                        onClick={handleGoLive}
                        className={cn(
                            "flex items-center gap-1 px-1.5 py-0.5 rounded",
                            "text-xs font-medium transition-colors",
                            "focus-visible:ring-2 focus-visible:ring-primary focus-visible:outline-none",
                            isLive
                                ? "text-info"
                                : "text-muted-foreground hover:text-foreground hover:bg-secondary"
                        )}
                        title={isLive ? t("live") : t("goToLive")}
                        aria-label={isLive ? t("live") : t("goToLive")}
                    >
                        <span
                            className={cn(
                                "w-1.5 h-1.5 rounded-full",
                                isLive ? "bg-info animate-pulse" : "bg-muted-foreground/50"
                            )}
                        />
                        {t("live")}
                    </button>
                </div>

                {/* Step progress bar (when taskPlan exists) */}
                {taskPlan && taskPlan.steps.length > 0 && (
                    <div className="border-t border-border shrink-0 bg-background">
                        {/* Summary row */}
                        <button
                            onClick={() => setStepsExpanded((v) => !v)}
                            className={cn(
                                "flex items-center gap-2 w-full px-3 h-8 text-left",
                                "hover:bg-secondary/30 transition-colors",
                                "focus-visible:ring-2 focus-visible:ring-primary focus-visible:outline-none"
                            )}
                        >
                            {/* Status icon */}
                            {currentPlanStep?.status === "completed" && (
                                <Check className="w-3.5 h-3.5 text-success shrink-0" />
                            )}
                            {currentPlanStep?.status === "running" && (
                                <Loader2 className="w-3.5 h-3.5 text-info animate-spin shrink-0" />
                            )}
                            {currentPlanStep?.status === "pending" && (
                                <Circle className="w-3.5 h-3.5 text-muted-foreground shrink-0" />
                            )}
                            {currentPlanStep?.status === "failed" && (
                                <Circle className="w-3.5 h-3.5 text-destructive shrink-0" />
                            )}

                            {/* Step title */}
                            <span className="text-xs text-foreground truncate flex-1">
                                {currentPlanStep?.title}
                            </span>

                            {/* Counter */}
                            <span className="text-xs text-muted-foreground tabular-nums shrink-0">
                                {taskPlan.completedSteps}/{taskPlan.totalSteps}
                            </span>

                            {/* Expand chevron */}
                            {stepsExpanded ? (
                                <ChevronUp className="w-3.5 h-3.5 text-muted-foreground shrink-0" />
                            ) : (
                                <ChevronDown className="w-3.5 h-3.5 text-muted-foreground shrink-0" />
                            )}
                        </button>

                        {/* Expanded steps list */}
                        {stepsExpanded && (
                            <div className="px-3 pb-2 space-y-0.5 max-h-48 overflow-y-auto">
                                {taskPlan.steps.map((step) => (
                                    <div
                                        key={step.id}
                                        className="flex items-center gap-2 py-1 px-1 rounded text-xs"
                                    >
                                        {step.status === "completed" && (
                                            <Check className="w-3 h-3 text-success shrink-0" />
                                        )}
                                        {step.status === "running" && (
                                            <Loader2 className="w-3 h-3 text-info animate-spin shrink-0" />
                                        )}
                                        {step.status === "pending" && (
                                            <Circle className="w-3 h-3 text-muted-foreground/40 shrink-0" />
                                        )}
                                        {step.status === "failed" && (
                                            <Circle className="w-3 h-3 text-destructive shrink-0" />
                                        )}
                                        <span
                                            className={cn(
                                                "truncate",
                                                step.status === "completed" && "text-muted-foreground",
                                                step.status === "running" && "text-foreground font-medium",
                                                step.status === "pending" && "text-muted-foreground/60",
                                                step.status === "failed" && "text-destructive"
                                            )}
                                        >
                                            {step.title}
                                        </span>
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>
                )}
            </div>
        </>
    );
}
