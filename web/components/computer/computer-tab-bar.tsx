"use client";

import React, { useCallback, useRef } from "react";
import { TerminalSquare, Monitor, Folder, X } from "lucide-react";
import { useTranslations } from "next-intl";
import { cn } from "@/lib/utils";
import type { ComputerMode } from "@/lib/stores/computer-store";

interface ComputerTabBarProps {
    activeMode: ComputerMode;
    onModeChange: (mode: ComputerMode) => void;
    onClose: () => void;
}

const tabConfig: { mode: ComputerMode; icon: React.ElementType; labelKey: string }[] = [
    { mode: "terminal", icon: TerminalSquare, labelKey: "terminal" },
    { mode: "browser", icon: Monitor, labelKey: "browser" },
    { mode: "file", icon: Folder, labelKey: "files" },
];

export function ComputerTabBar({
    activeMode,
    onModeChange,
    onClose,
}: ComputerTabBarProps) {
    const t = useTranslations("computer");
    const tabListRef = useRef<HTMLDivElement>(null);

    const handleTabKeyDown = useCallback((e: React.KeyboardEvent<HTMLDivElement>) => {
        const currentIndex = tabConfig.findIndex(({ mode }) => mode === activeMode);
        let nextIndex: number | null = null;

        if (e.key === "ArrowRight" || e.key === "ArrowDown") {
            e.preventDefault();
            nextIndex = (currentIndex + 1) % tabConfig.length;
        } else if (e.key === "ArrowLeft" || e.key === "ArrowUp") {
            e.preventDefault();
            nextIndex = (currentIndex - 1 + tabConfig.length) % tabConfig.length;
        } else if (e.key === "Home") {
            e.preventDefault();
            nextIndex = 0;
        } else if (e.key === "End") {
            e.preventDefault();
            nextIndex = tabConfig.length - 1;
        }

        if (nextIndex !== null) {
            onModeChange(tabConfig[nextIndex].mode);
            const buttons = tabListRef.current?.querySelectorAll<HTMLButtonElement>('[role="tab"]');
            buttons?.[nextIndex]?.focus();
        }
    }, [activeMode, onModeChange]);

    return (
        <div className="h-10 px-2 flex items-center border-b border-border bg-background shrink-0">
            <div
                className="flex items-center gap-0.5 flex-1"
                role="tablist"
                aria-label={t("panelTitle")}
                ref={tabListRef}
                onKeyDown={handleTabKeyDown}
            >
                {tabConfig.map(({ mode, icon: Icon, labelKey }) => {
                    const isActive = activeMode === mode;
                    return (
                        <button
                            key={mode}
                            className={cn(
                                "relative flex items-center gap-1.5 px-3 h-8 text-xs font-medium transition-colors",
                                "focus-visible:ring-2 focus-visible:ring-primary focus-visible:outline-none rounded-md",
                                isActive
                                    ? "text-foreground"
                                    : "text-muted-foreground hover:text-foreground"
                            )}
                            onClick={() => onModeChange(mode)}
                            role="tab"
                            aria-selected={isActive}
                            tabIndex={isActive ? 0 : -1}
                        >
                            <Icon className="w-3.5 h-3.5 flex-shrink-0" />
                            <span>{t(labelKey)}</span>
                            {isActive && (
                                <span className="absolute bottom-0 left-2 right-2 h-0.5 bg-primary rounded-full" />
                            )}
                        </button>
                    );
                })}
            </div>

            <button
                className={cn(
                    "h-8 w-8 inline-flex items-center justify-center rounded-md shrink-0",
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
    );
}
