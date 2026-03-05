"use client";

import React, { useCallback, useRef, useMemo } from "react";
import { TerminalSquare, Monitor, Folder, X, ExternalLink } from "lucide-react";
import { useTranslations } from "next-intl";
import { cn } from "@/lib/utils";
import type { ComputerMode } from "@/lib/stores/computer-store";

interface ComputerPanelHeaderProps {
    activeMode: ComputerMode;
    onModeChange: (mode: ComputerMode) => void;
    onClose: () => void;
    /** Stage name for i18n lookup (e.g. "plan", "execute", "search") */
    activityStage?: string | null;
    /** Backend description used as fallback when no translation found */
    activityDescriptionFallback?: string | null;
}

const modeConfig: { mode: ComputerMode; icon: React.ElementType; labelKey: string }[] = [
    { mode: "terminal", icon: TerminalSquare, labelKey: "terminal" },
    { mode: "browser", icon: Monitor, labelKey: "browser" },
    { mode: "file", icon: Folder, labelKey: "files" },
];

export function ComputerPanelHeader({
    activeMode,
    onModeChange,
    onClose,
    activityStage,
    activityDescriptionFallback,
}: ComputerPanelHeaderProps) {
    const t = useTranslations("computer");
    const tStages = useTranslations("chat.agent.stages");
    const tabListRef = useRef<HTMLDivElement>(null);

    const handleTabKeyDown = useCallback(
        (e: React.KeyboardEvent<HTMLDivElement>) => {
            const currentIndex = modeConfig.findIndex(({ mode }) => mode === activeMode);
            let nextIndex: number | null = null;

            if (e.key === "ArrowRight" || e.key === "ArrowDown") {
                e.preventDefault();
                nextIndex = (currentIndex + 1) % modeConfig.length;
            } else if (e.key === "ArrowLeft" || e.key === "ArrowUp") {
                e.preventDefault();
                nextIndex = (currentIndex - 1 + modeConfig.length) % modeConfig.length;
            } else if (e.key === "Home") {
                e.preventDefault();
                nextIndex = 0;
            } else if (e.key === "End") {
                e.preventDefault();
                nextIndex = modeConfig.length - 1;
            }

            if (nextIndex !== null) {
                onModeChange(modeConfig[nextIndex].mode);
                const buttons =
                    tabListRef.current?.querySelectorAll<HTMLButtonElement>('[role="tab"]');
                buttons?.[nextIndex]?.focus();
            }
        },
        [activeMode, onModeChange]
    );

    const handleOpenNewTab = useCallback(() => {
        // Placeholder for open-in-new-tab / fullscreen action
    }, []);

    // Resolve the activity label via i18n, falling back to the backend description
    const activityLabel = useMemo(() => {
        if (!activityStage && !activityDescriptionFallback) return null;
        if (activityStage) {
            // Try translating the stage name via chat.agent.stages.{name}.running
            const key = `${activityStage}.running` as Parameters<typeof tStages>[0];
            if (typeof tStages.has === "function" && tStages.has(key)) {
                try {
                    return tStages(key);
                } catch {
                    // fall through
                }
            }
        }
        return activityDescriptionFallback ?? null;
    }, [activityStage, activityDescriptionFallback, tStages]);

    return (
        <div className="shrink-0">
            {/* Row 1: Segmented tabs + action buttons */}
            <div className="flex items-center justify-between gap-2 px-3 h-10 border-b border-border">
                <div
                    className="flex items-center gap-0.5 flex-1 min-w-0"
                    role="tablist"
                    aria-label={t("panelTitle")}
                    ref={tabListRef}
                    onKeyDown={handleTabKeyDown}
                >
                    {modeConfig.map(({ mode, icon: Icon, labelKey }) => {
                        const isActive = activeMode === mode;
                        return (
                            <button
                                key={mode}
                                className={cn(
                                    "relative h-8 px-2 flex items-center gap-1.5",
                                    "text-xs transition-colors",
                                    "cursor-pointer",
                                    "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary",
                                    isActive
                                        ? "text-foreground font-medium"
                                        : "text-muted-foreground hover:text-foreground"
                                )}
                                onClick={() => onModeChange(mode)}
                                role="tab"
                                aria-selected={isActive}
                                tabIndex={isActive ? 0 : -1}
                            >
                                <Icon className="w-3.5 h-3.5 shrink-0" />
                                <span>{t(labelKey)}</span>
                                {isActive && (
                                    <span className="absolute bottom-0 left-1 right-1 h-0.5 bg-primary rounded-full" />
                                )}
                            </button>
                        );
                    })}
                </div>
                <div className="flex items-center gap-0.5 shrink-0">
                    <button
                        className={cn(
                            "h-7 w-7 inline-flex items-center justify-center rounded-md shrink-0",
                            "text-muted-foreground hover:text-foreground hover:bg-secondary/50 transition-colors",
                            "focus-visible:ring-2 focus-visible:ring-primary focus-visible:outline-none"
                        )}
                        onClick={handleOpenNewTab}
                        title={t("openInNewTab")}
                        aria-label={t("openInNewTab")}
                    >
                        <ExternalLink className="w-3.5 h-3.5" />
                    </button>
                    <button
                        className={cn(
                            "h-7 w-7 inline-flex items-center justify-center rounded-md shrink-0",
                            "text-muted-foreground hover:text-foreground hover:bg-secondary/50 transition-colors",
                            "focus-visible:ring-2 focus-visible:ring-primary focus-visible:outline-none"
                        )}
                        onClick={onClose}
                        title={t("close")}
                        aria-label={t("close")}
                    >
                        <X className="w-3.5 h-3.5" />
                    </button>
                </div>
            </div>

            {/* Row 2: Activity status */}
            <div className="flex items-center gap-2 px-3 h-7 border-b border-border bg-muted/30 overflow-hidden">
                <span
                    className={cn(
                        "w-1.5 h-1.5 rounded-full shrink-0",
                        activityLabel ? "bg-info animate-pulse" : "bg-muted-foreground/40"
                    )}
                />
                <span className="text-xs text-muted-foreground truncate font-mono">
                    {activityLabel ?? t("activityIdle")}
                </span>
            </div>
        </div>
    );
}
