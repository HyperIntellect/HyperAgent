"use client";

import React from "react";
import { Monitor } from "lucide-react";
import { useTranslations } from "next-intl";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { useComputerStore } from "@/lib/stores/computer-store";
import { useAgentProgressStore } from "@/lib/stores/agent-progress-store";

interface ComputerToggleButtonProps {
    className?: string;
}

export function ComputerToggleButton({ className }: ComputerToggleButtonProps) {
    const t = useTranslations("computer");

    // Global UI state
    const isOpen = useComputerStore((state) => state.isOpen);
    const togglePanel = useComputerStore((state) => state.togglePanel);
    const unreadActivityCount = useComputerStore((state) => state.unreadActivityCount);

    // Per-conversation state via selector functions (cached to avoid infinite loops)
    const terminalLines = useComputerStore((state) => state.getTerminalLines());
    const browserStream = useComputerStore((state) => state.getBrowserStream());

    const { activeProgress } = useAgentProgressStore();

    // Show activity indicator if there's terminal activity or browser stream
    const hasActivity = terminalLines.length > 0 || browserStream !== null || activeProgress?.browserStream !== null;
    const isStreaming = activeProgress?.isStreaming;

    // Only show button when panel is open, has activity, or has unread count
    if (!isOpen && !hasActivity && unreadActivityCount === 0) return null;

    return (
        <Button
            variant="ghost"
            size="icon"
            onClick={togglePanel}
            className={cn(
                "h-9 w-9 relative",
                isOpen && "bg-secondary",
                className
            )}
            title={t("title")}
            aria-label={t("title")}
            aria-pressed={isOpen}
        >
            <Monitor className="w-4 h-4" />

            {/* Unread activity count badge */}
            {!isOpen && unreadActivityCount > 0 && (
                <span className="absolute -top-0.5 -right-0.5 min-w-[16px] h-4 px-1 flex items-center justify-center rounded-full bg-primary text-primary-foreground text-xs font-medium leading-none">
                    {unreadActivityCount > 99 ? "99+" : unreadActivityCount}
                </span>
            )}

            {/* Activity indicator (when no unread count) */}
            {hasActivity && (unreadActivityCount === 0 || isOpen) && (
                <span
                    className={cn(
                        "absolute top-1 right-1 w-2 h-2 rounded-full",
                        isStreaming
                            ? "bg-primary animate-pulse"
                            : "bg-muted-foreground/50"
                    )}
                />
            )}
        </Button>
    );
}
