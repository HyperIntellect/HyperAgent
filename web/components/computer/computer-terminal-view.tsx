"use client";

import React, { useRef, useEffect, useState, useCallback } from "react";
import { useCopyToClipboard } from "@/lib/hooks/use-copy-to-clipboard";
import { cn } from "@/lib/utils";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
    Tooltip,
    TooltipContent,
    TooltipProvider,
    TooltipTrigger,
} from "@/components/ui/tooltip";
import {
    Copy,
    Check,
    Trash2,
    WrapText,
    Loader2,
    TerminalSquare,
} from "lucide-react";
import { useTranslations } from "next-intl";
import type { TerminalLine } from "@/lib/stores/computer-store";

interface ComputerTerminalViewProps {
    lines: TerminalLine[];
    isLive?: boolean;
    currentCommand?: string | null;
    currentCwd?: string;
    onClear?: () => void;
    onSendCommand?: (command: string) => void;
    className?: string;
}

const PLACEHOLDER_COMMANDS = ["ls", "pwd", "cat README.md", "npm run dev"];

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
                            "hover:bg-accent",
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
    return (
        <>
            <span className="text-terminal-prompt font-semibold">
                ubuntu@sandbox
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
    onClear,
    onSendCommand,
    className,
}: ComputerTerminalViewProps) {
    const scrollRef = useRef<HTMLDivElement>(null);
    const bottomRef = useRef<HTMLDivElement>(null);
    const inputRef = useRef<HTMLInputElement>(null);
    const [wordWrap, setWordWrap] = useState(true);
    const [inputValue, setInputValue] = useState("");
    const [commandHistory, setCommandHistory] = useState<string[]>([]);
    const [historyIndex, setHistoryIndex] = useState(-1);
    const [isInputFocused, setIsInputFocused] = useState(false);
    const [showControlBanner, setShowControlBanner] = useState(false);
    const [placeholderIndex, setPlaceholderIndex] = useState(0);
    const t = useTranslations("computer");

    // Auto-scroll to bottom when new lines are added (only when live)
    useEffect(() => {
        if (isLive && bottomRef.current) {
            bottomRef.current.scrollIntoView({ behavior: "smooth" });
        }
    }, [lines.length, isLive]);

    // Cycle placeholder commands every 3 seconds
    useEffect(() => {
        const interval = setInterval(() => {
            setPlaceholderIndex((prev) => (prev + 1) % PLACEHOLDER_COMMANDS.length);
        }, 3000);
        return () => clearInterval(interval);
    }, []);

    // Handle clicking the terminal content area to focus input
    const handleContentClick = useCallback(() => {
        if (inputRef.current) {
            inputRef.current.focus();
            setShowControlBanner(true);
        }
    }, []);

    // Handle keyboard input for the terminal input
    const handleKeyDown = useCallback((e: React.KeyboardEvent<HTMLInputElement>) => {
        // Ctrl+C interrupt
        if (e.key === "c" && (e.ctrlKey || e.metaKey)) {
            e.preventDefault();
            if (onSendCommand) {
                onSendCommand("\x03"); // Send interrupt signal
            }
            setInputValue("");
            return;
        }

        if (e.key === "Enter" && inputValue.trim() && onSendCommand) {
            e.preventDefault();
            onSendCommand(inputValue.trim());
            setCommandHistory((prev) => {
                const next = [inputValue.trim(), ...prev];
                return next.slice(0, 50);
            });
            setInputValue("");
            setHistoryIndex(-1);
        } else if (e.key === "ArrowUp") {
            e.preventDefault();
            if (commandHistory.length > 0) {
                const nextIndex = Math.min(historyIndex + 1, commandHistory.length - 1);
                setHistoryIndex(nextIndex);
                setInputValue(commandHistory[nextIndex]);
            }
        } else if (e.key === "ArrowDown") {
            e.preventDefault();
            if (historyIndex > 0) {
                const nextIndex = historyIndex - 1;
                setHistoryIndex(nextIndex);
                setInputValue(commandHistory[nextIndex]);
            } else {
                setHistoryIndex(-1);
                setInputValue("");
            }
        }
    }, [inputValue, onSendCommand, commandHistory, historyIndex]);

    // Lines are already filtered by the parent panel based on timeline position
    const visibleLines = !isLive
            ? lines
            : lines;

    // Shorten cwd for display (replace /home/user with ~)
    const displayCwd = currentCwd.replace(/^\/home\/ubuntu/, "~") || "~";

    return (
        <div className={cn("flex flex-col flex-1 bg-terminal-bg", className)}>
            {/* Terminal header bar */}
            <div
                className={cn(
                    "flex items-center justify-between px-3 h-8 shrink-0",
                    "bg-[hsl(var(--terminal-bg))] border-b border-terminal-border"
                )}
            >
                <div className="flex items-center gap-2 min-w-0">
                    <TerminalSquare className="w-3.5 h-3.5 text-terminal-output/60 shrink-0" />
                    <span className="text-xs font-mono text-terminal-output/60 truncate">
                        ubuntu@sandbox:{displayCwd}
                    </span>
                </div>
                <div className="flex items-center gap-0.5 shrink-0">
                    <TooltipProvider delayDuration={300}>
                        <Tooltip>
                            <TooltipTrigger asChild>
                                <button
                                    onClick={() => setWordWrap((v) => !v)}
                                    className={cn(
                                        "h-8 w-8 flex items-center justify-center rounded transition-colors",
                                        "text-terminal-output/50 hover:text-terminal-fg hover:bg-accent",
                                        wordWrap && "text-terminal-fg bg-accent",
                                        "focus-visible:ring-2 focus-visible:ring-primary focus-visible:outline-none"
                                    )}
                                    aria-label={t("wordWrap")}
                                >
                                    <WrapText className="w-3.5 h-3.5" />
                                </button>
                            </TooltipTrigger>
                            <TooltipContent side="bottom">
                                {t("wordWrap")}
                            </TooltipContent>
                        </Tooltip>
                    </TooltipProvider>

                    {onClear && (
                        <TooltipProvider delayDuration={300}>
                            <Tooltip>
                                <TooltipTrigger asChild>
                                    <button
                                        onClick={onClear}
                                        className={cn(
                                            "h-8 w-8 flex items-center justify-center rounded transition-colors",
                                            "text-terminal-output/50 hover:text-terminal-fg hover:bg-accent",
                                            "focus-visible:ring-2 focus-visible:ring-primary focus-visible:outline-none"
                                        )}
                                        aria-label={t("clearTerminal")}
                                    >
                                        <Trash2 className="w-3.5 h-3.5" />
                                    </button>
                                </TooltipTrigger>
                                <TooltipContent side="bottom">
                                    {t("clearTerminal")}
                                </TooltipContent>
                            </Tooltip>
                        </TooltipProvider>
                    )}
                </div>
            </div>

            {/* Terminal content */}
            <ScrollArea className="flex-1">
                <div
                    ref={scrollRef}
                    className="p-3 font-mono text-sm leading-relaxed min-h-full cursor-text"
                    role="log"
                    aria-live="polite"
                    aria-label={t("terminalOutput")}
                    onClick={handleContentClick}
                >
                    {visibleLines.length === 0 && !currentCommand ? (
                        <div className="flex flex-col items-center justify-center h-full min-h-[120px] gap-3">
                            <div className="font-mono text-sm">
                                <span className="text-terminal-prompt font-semibold">ubuntu@sandbox</span>
                                <span className="text-terminal-fg">:</span>
                                <span className="text-terminal-command">~</span>
                                <span className="text-terminal-fg">$ </span>
                                <span className="inline-block w-2 h-4 bg-terminal-fg/70 animate-terminal-cursor align-middle" />
                            </div>
                            <span className="text-terminal-output/40 text-xs">
                                {t("noTerminalOutput")}
                            </span>
                        </div>
                    ) : (
                        visibleLines.map((line) => (
                            <div
                                key={line.id}
                                className="py-0.5 hover:bg-[hsl(var(--terminal-output)/0.05)] transition-colors rounded px-1 -mx-1"
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
                        <div className="py-0.5 px-1 -mx-1 border-l-2 border-l-[hsl(var(--terminal-prompt))] group flex items-start justify-between gap-2">
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
                    {isLive && !currentCommand && visibleLines.length > 0 && (
                        <div className="py-0.5 px-1 -mx-1 flex items-center">
                            <TerminalPrompt cwd={currentCwd} />
                            <span className="w-2 h-4 bg-terminal-fg/70 animate-terminal-cursor ml-0.5" />
                        </div>
                    )}

                    {/* Scroll anchor */}
                    <div ref={bottomRef} />
                </div>
            </ScrollArea>

            {/* Terminal input */}
            {isLive && onSendCommand && (
                <div className="shrink-0">
                    {/* "You are in control" banner */}
                    {showControlBanner && isInputFocused && (
                        <div className="bg-primary/10 text-primary text-xs py-1 px-3 rounded">
                            You are in control
                        </div>
                    )}
                    <div
                        className={cn(
                            "flex items-center border-t border-border/50 px-3 py-1.5",
                            "bg-[hsl(var(--terminal-input-bg))]",
                            isInputFocused && "border-l-2 border-l-primary"
                        )}
                    >
                        <span className="text-terminal-prompt text-xs mr-2 flex-shrink-0 font-mono">
                            ubuntu@sandbox:{displayCwd}$
                        </span>
                        <input
                            ref={inputRef}
                            value={inputValue}
                            onChange={(e) => setInputValue(e.target.value)}
                            onKeyDown={handleKeyDown}
                            onFocus={() => {
                                setIsInputFocused(true);
                                setShowControlBanner(true);
                            }}
                            onBlur={() => {
                                setIsInputFocused(false);
                                setShowControlBanner(false);
                            }}
                            className={cn(
                                "flex-1 bg-transparent text-terminal-fg text-xs outline-none caret-terminal-fg font-mono placeholder:text-terminal-output/30",
                                "focus-visible:ring-2 focus-visible:ring-primary focus-visible:outline-none"
                            )}
                            placeholder={PLACEHOLDER_COMMANDS[placeholderIndex]}
                            autoFocus
                        />
                    </div>
                </div>
            )}
        </div>
    );
}
