"use client";

import React, { useCallback } from "react";
import { TerminalSquare, Monitor, Folder, X, ExternalLink } from "lucide-react";
import { useTranslations } from "next-intl";
import { cn } from "@/lib/utils";
import type { ComputerMode } from "@/lib/stores/computer-store";

interface ComputerPanelHeaderProps {
    activeMode: ComputerMode;
    onModeChange: (mode: ComputerMode) => void;
    onClose: () => void;
    activityDescription?: string | null;
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
    activityDescription,
}: ComputerPanelHeaderProps) {
    const t = useTranslations("computer");

    const currentModeConfig = modeConfig.find((m) => m.mode === activeMode) ?? modeConfig[0];
    const ModeIcon = currentModeConfig.icon;

    const cycleMode = useCallback(() => {
        const currentIndex = modeConfig.findIndex((m) => m.mode === activeMode);
        const nextIndex = (currentIndex + 1) % modeConfig.length;
        onModeChange(modeConfig[nextIndex].mode);
    }, [activeMode, onModeChange]);

    const handleOpenNewTab = useCallback(() => {
        // Placeholder for open-in-new-tab / fullscreen action
    }, []);

    return (
        <div className="shrink-0">
            {/* Row 1: Title + action buttons */}
            <div className="flex items-center justify-between px-3 h-10 border-b border-border">
                <span className="text-sm font-semibold text-foreground truncate">
                    {t("panelTitle")}
                </span>
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
            <div className="flex items-center gap-2 px-3 h-8 border-b border-border bg-muted/30 overflow-hidden">
                <ModeIcon className="w-3.5 h-3.5 text-muted-foreground shrink-0" />
                <span className="text-xs text-muted-foreground shrink-0">
                    {t("activityUsing")}
                </span>
                <button
                    onClick={cycleMode}
                    className={cn(
                        "text-xs font-medium text-foreground hover:text-primary transition-colors",
                        "focus-visible:ring-2 focus-visible:ring-primary focus-visible:outline-none rounded px-0.5"
                    )}
                >
                    {t(currentModeConfig.labelKey)}
                </button>
                {activityDescription && (
                    <>
                        <span className="text-muted-foreground/50 text-xs shrink-0">|</span>
                        <span className="text-xs text-muted-foreground truncate font-mono">
                            {activityDescription}
                        </span>
                    </>
                )}
                {!activityDescription && (
                    <>
                        <span className="text-muted-foreground/50 text-xs shrink-0">|</span>
                        <span className="text-xs text-muted-foreground/50">
                            {t("activityIdle")}
                        </span>
                    </>
                )}
            </div>
        </div>
    );
}
