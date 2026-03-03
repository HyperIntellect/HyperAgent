"use client";

import React, { useRef, useEffect, useState, useCallback } from "react";
import { useCopyToClipboard } from "@/lib/hooks/use-copy-to-clipboard";
import { cn } from "@/lib/utils";
import {
    Tooltip,
    TooltipContent,
    TooltipProvider,
    TooltipTrigger,
} from "@/components/ui/tooltip";
import {
    Copy,
    Check,
    Loader2,
    ArrowDown,
} from "lucide-react";
import { useTranslations } from "next-intl";
import type { TerminalLine } from "@/lib/stores/computer-store";

interface ComputerTerminalViewProps {
    lines: TerminalLine[];
    isLive?: boolean;
    currentCommand?: string | null;
    currentCwd?: string;
    className?: string;
}

function CopyButton({ text }: { text: string }) {
    const { copied, copy } = useCopyToClipboard(1500);
    const t = useTranslations("computer");

    const handleCopy = useCallback(
        (e: React.MouseEvent) => {
            e.stopPropagation();
            copy(text);
        },
        [text, copy]
    );

    return (
        <TooltipProvider delayDuration={300}>
            <Tooltip>
                <TooltipTrigger asChild>
                    <button
                        onClick={handleCopy}
                        className={cn(
                            "h-8 w-8 flex items-center justify-center rounded transition-colors",
                            "opacity-0 group-hover:opacity-100",
                            "text-terminal-output/50 hover:text-terminal-fg",
                            "hover:bg-terminal-output/15",
                            "focus-visible:ring-2 focus-visible:ring-primary focus-visible:outline-none"
                        )}
                        aria-label={t("copyCommand")}
                    >
                        {copied ? (
                            <Check className="w-3.5 h-3.5 text-terminal-prompt" />
                        ) : (
                            <Copy className="w-3.5 h-3.5" />
                        )}
                    </button>
                </TooltipTrigger>
                <TooltipContent side="top">
                    {copied ? t("copied") : t("copyCommand")}
                </TooltipContent>
            </Tooltip>
        </TooltipProvider>
    );
}

function TerminalPrompt({ cwd }: { cwd?: string }) {
    const t = useTranslations("computer");
    return (
        <>
            <span className="text-terminal-prompt font-semibold">
                {t("terminalPrompt")}
            </span>
            <span className="text-terminal-fg">:</span>
            <span className="text-terminal-command">{cwd || "~"}</span>
            <span className="text-terminal-fg">$ </span>
        </>
    );
}

function TerminalLineContent({
    line,
    wordWrap,
}: {
    line: TerminalLine;
    wordWrap: boolean;
}) {
    const wrapClass = wordWrap
        ? "whitespace-pre-wrap break-all"
        : "whitespace-pre overflow-x-auto";

    switch (line.type) {
        case "prompt":
            return (
                <div className="flex items-start gap-0">
                    <TerminalPrompt cwd={line.cwd} />
                </div>
            );

        case "command":
            return (
                <div className="group flex items-start justify-between gap-2">
                    <div className="flex items-start min-w-0">
                        <TerminalPrompt cwd={line.cwd} />
                        <span
                            className={cn("text-terminal-command", wrapClass)}
                        >
                            {line.content}
                        </span>
                    </div>
                    <CopyButton text={line.content} />
                </div>
            );

        case "output":
            return (
                <div className={cn("text-terminal-output", wrapClass)}>
                    {line.content}
                </div>
            );

        case "error":
            return (
                <div className={cn("text-terminal-error", wrapClass)}>
                    {line.content}
                </div>
            );

        default:
            return (
                <div className={cn("text-terminal-fg", wrapClass)}>
                    {line.content}
                </div>
            );
    }
}

export function ComputerTerminalView({
    lines,
    isLive = true,
    currentCommand,
    currentCwd = "/home/user",
    className,
}: ComputerTerminalViewProps) {
    const scrollRef = useRef<HTMLDivElement>(null);
    const bottomRef = useRef<HTMLDivElement>(null);
    const wordWrap = true;
    const t = useTranslations("computer");

    const [isNearBottom, setIsNearBottom] = useState(true);

    // Track scroll position to decide whether to auto-scroll
    const handleScroll = useCallback(() => {
        const el = scrollRef.current;
        if (!el) return;
        const threshold = 60;
        const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < threshold;
        setIsNearBottom(atBottom);
    }, []);

    // Auto-scroll to bottom only when user is near the bottom
    useEffect(() => {
        if (isLive && isNearBottom && bottomRef.current) {
            bottomRef.current.scrollIntoView({ behavior: "smooth" });
        }
    }, [lines.length, isLive, isNearBottom]);

    // Scroll-to-bottom handler for the button
    const scrollToBottom = useCallback(() => {
        if (bottomRef.current) {
            bottomRef.current.scrollIntoView({ behavior: "smooth" });
            setIsNearBottom(true);
        }
    }, []);

    return (
        <div className={cn("flex flex-col flex-1 bg-terminal-bg", className)}>
            <div className="flex-1 min-h-0 relative">
                <div
                    ref={scrollRef}
                    onScroll={handleScroll}
                    className="absolute inset-0 overflow-y-auto px-3 py-2 font-mono text-sm leading-relaxed"
                    role="log"
                    aria-live="polite"
                    aria-label={t("terminalOutput")}
                >
                    {lines.length === 0 && !currentCommand ? (
                        <div className="text-terminal-output/40 text-xs py-6 px-1">
                            {t("noTerminalOutput")}
                        </div>
                    ) : (
                        lines.map((line) => (
                            <div
                                key={line.id}
                                className="py-0.5 hover:bg-terminal-output/5 transition-colors rounded px-1 -mx-1"
                            >
                                <TerminalLineContent
                                    line={line}
                                    wordWrap={wordWrap}
                                />
                            </div>
                        ))
                    )}

                    {/* Running command indicator */}
                    {isLive && currentCommand && (
                        <div className="py-0.5 px-1 -mx-1 group flex items-start justify-between gap-2">
                            <div className="flex items-start min-w-0">
                                <TerminalPrompt cwd={currentCwd} />
                                <span className="text-terminal-command">
                                    {currentCommand}
                                </span>
                                <Loader2 className="w-3.5 h-3.5 text-terminal-output/60 animate-spin ml-2 mt-0.5 shrink-0" />
                            </div>
                        </div>
                    )}

                    {/* Idle cursor when live and no command running */}
                    {isLive && !currentCommand && lines.length > 0 && (
                        <div className="py-0.5 px-1 -mx-1 flex items-center">
                            <TerminalPrompt cwd={currentCwd} />
                            <span className="w-2 h-4 bg-terminal-fg/70 animate-terminal-cursor ml-0.5" />
                        </div>
                    )}

                    {/* Scroll anchor */}
                    <div ref={bottomRef} />
                </div>

                {/* Scroll to bottom button */}
                {!isNearBottom && lines.length > 0 && (
                    <button
                        onClick={scrollToBottom}
                        className={cn(
                            "absolute bottom-2 right-3 z-10",
                            "h-7 w-7 rounded-full flex items-center justify-center",
                            "bg-terminal-fg/15 hover:bg-terminal-fg/25 backdrop-blur-sm",
                            "text-terminal-fg/70 hover:text-terminal-fg",
                            "transition-all shadow-sm",
                            "focus-visible:ring-2 focus-visible:ring-primary focus-visible:outline-none"
                        )}
                        aria-label={t("scrollToBottom")}
                    >
                        <ArrowDown className="w-3.5 h-3.5" />
                    </button>
                )}
            </div>
        </div>
    );
}
