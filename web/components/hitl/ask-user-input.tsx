"use client";

import React, { useState, useEffect, useRef, useCallback } from "react";
import { useTranslations } from "next-intl";
import { Send, Clock, MessageSquare, CheckCircle2, Check, X } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import type { InterruptEvent, InterruptResponse, InterruptOption } from "@/lib/types";

interface AskUserInputProps {
    interrupt: InterruptEvent;
    onRespond: (response: InterruptResponse) => void;
    onCancel: () => void;
}

// Countdown display - Aligned with design guide
function CountdownBadge({ seconds, onExpire }: { seconds: number; onExpire: () => void }) {
    const [remaining, setRemaining] = useState(seconds);

    // Reset remaining when seconds prop changes (new interrupt)
    useEffect(() => {
        setRemaining(seconds);
    }, [seconds]);

    useEffect(() => {
        if (remaining <= 0) {
            onExpire();
            return;
        }

        const timer = setInterval(() => {
            setRemaining((prev) => {
                if (prev <= 1) {
                    clearInterval(timer);
                    return 0;
                }
                return prev - 1;
            });
        }, 1000);

        return () => clearInterval(timer);
    }, [remaining, onExpire]);

    const minutes = Math.floor(remaining / 60);
    const secs = remaining % 60;
    const isLow = remaining <= 30;

    return (
        <span className={cn(
            "inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-medium tabular-nums transition-colors",
            "border",
            isLow
                ? "text-destructive border-destructive/30 bg-destructive/5"
                : "text-muted-foreground border-border/50 bg-secondary/30"
        )}>
            <Clock className="w-4 h-4" />
            {minutes}:{secs.toString().padStart(2, "0")}
        </span>
    );
}

// Decision/Choice buttons - Aligned with design guide
function DecisionInput({
    options,
    onSelect,
}: {
    options: InterruptOption[];
    onSelect: (value: string) => void;
}) {
    const [selected, setSelected] = useState<string | null>(null);

    return (
        <div className="flex flex-wrap gap-2">
            {options.map((option) => (
                <button
                    key={option.value}
                    className={cn(
                        "px-4 py-2.5 text-sm font-medium rounded-lg border transition-colors",
                        "focus-visible:ring-2 focus-visible:ring-primary focus-visible:ring-offset-2 focus-visible:outline-none",
                        selected === option.value
                            ? "bg-foreground text-background border-foreground"
                            : "bg-transparent text-foreground border-border hover:bg-secondary hover:text-foreground"
                    )}
                    onClick={() => {
                        setSelected(option.value);
                        onSelect(option.value);
                    }}
                >
                    <span className="flex items-center gap-2">
                        {selected === option.value && (
                            <CheckCircle2 className="w-4 h-4" />
                        )}
                        <span>{option.label}</span>
                    </span>
                </button>
            ))}
        </div>
    );
}

// Boolean Yes/No input - Aligned with design guide
function BooleanInput({
    onSelect,
}: {
    onSelect: (value: boolean) => void;
}) {
    const t = useTranslations("hitl");

    return (
        <div className="flex gap-2">
            <button
                className={cn(
                    "flex items-center gap-2 px-4 py-2.5 text-sm font-medium rounded-lg border transition-colors",
                    "focus-visible:ring-2 focus-visible:ring-primary focus-visible:ring-offset-2 focus-visible:outline-none",
                    "bg-transparent text-foreground border-border hover:bg-foreground hover:text-background hover:border-foreground"
                )}
                onClick={() => onSelect(true)}
            >
                <Check className="w-4 h-4" />
                <span>{t("yes")}</span>
            </button>
            <button
                className={cn(
                    "flex items-center gap-2 px-4 py-2.5 text-sm font-medium rounded-lg border transition-colors",
                    "focus-visible:ring-2 focus-visible:ring-primary focus-visible:ring-offset-2 focus-visible:outline-none",
                    "bg-transparent text-foreground border-border hover:bg-foreground hover:text-background hover:border-foreground"
                )}
                onClick={() => onSelect(false)}
            >
                <X className="w-4 h-4" />
                <span>{t("no")}</span>
            </button>
        </div>
    );
}

// Approval Approve/Deny input - For tool/action approval requests
function ApprovalInput({
    onApprove,
    onDeny,
}: {
    onApprove: () => void;
    onDeny: () => void;
}) {
    const t = useTranslations("hitl");

    return (
        <div className="flex gap-2">
            <button
                className={cn(
                    "flex items-center gap-2 px-4 py-2.5 text-sm font-medium rounded-lg border transition-colors",
                    "focus-visible:ring-2 focus-visible:ring-primary focus-visible:ring-offset-2 focus-visible:outline-none",
                    "bg-transparent text-foreground border-border hover:bg-foreground hover:text-background hover:border-foreground"
                )}
                onClick={onApprove}
            >
                <Check className="w-4 h-4" />
                <span>{t("approve")}</span>
            </button>
            <button
                className={cn(
                    "flex items-center gap-2 px-4 py-2.5 text-sm font-medium rounded-lg border transition-colors",
                    "focus-visible:ring-2 focus-visible:ring-primary focus-visible:ring-offset-2 focus-visible:outline-none",
                    "bg-transparent text-foreground border-border hover:bg-foreground hover:text-background hover:border-foreground"
                )}
                onClick={onDeny}
            >
                <X className="w-4 h-4" />
                <span>{t("deny")}</span>
            </button>
        </div>
    );
}

// Text input field - Aligned with design guide
function TextInput({
    onSubmit,
    placeholder,
}: {
    onSubmit: (value: string) => void;
    placeholder?: string;
}) {
    const [value, setValue] = useState("");
    const inputRef = useRef<HTMLTextAreaElement>(null);

    useEffect(() => {
        // Auto-focus with slight delay for animation
        const timer = setTimeout(() => inputRef.current?.focus(), 100);
        return () => clearTimeout(timer);
    }, []);

    const handleSubmit = () => {
        if (value.trim()) {
            onSubmit(value.trim());
        }
    };

    const handleKeyDown = (e: React.KeyboardEvent) => {
        if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
            e.preventDefault();
            handleSubmit();
        }
    };

    return (
        <div className="flex gap-2 items-end">
            <Textarea
                ref={inputRef}
                value={value}
                onChange={(e) => setValue(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder={placeholder}
                rows={1}
                className={cn(
                    "min-h-[42px] max-h-[120px] resize-none flex-1 text-sm",
                    "border-border bg-transparent rounded-lg",
                    "focus-visible:ring-2 focus-visible:ring-primary focus-visible:ring-offset-2",
                    "transition-colors"
                )}
            />
            <Button
                variant="primary"
                onClick={handleSubmit}
                disabled={!value.trim()}
                className="h-[42px] px-4"
            >
                <Send className="w-4 h-4" />
            </Button>
        </div>
    );
}

// Generate localized message for tool approval from tool_info
function useApprovalMessage(interrupt: InterruptEvent): string {
    const t = useTranslations("hitl");

    if (interrupt.interrupt_type !== "approval" || !interrupt.tool_info) {
        return interrupt.message;
    }

    const { name, args } = interrupt.tool_info;
    const lowerName = name.toLowerCase();

    if (lowerName.includes("browser_navigate") || lowerName === "computer_tool") {
        const url = (args.url as string) || (args.action as Record<string, string>)?.text || "unknown";
        return t("approvalBrowserNav", { url });
    }
    if (lowerName.includes("browser_click")) {
        const target = (args.selector as string) || (args.text as string) || "element";
        return t("approvalBrowserClick", { target });
    }
    if (lowerName.includes("browser_type")) {
        const target = (args.selector as string) || (args.text as string) || "element";
        return t("approvalBrowserType", { target });
    }
    if (["execute_code", "code_interpreter", "python_repl", "sandbox_execute"].includes(name)) {
        return t("approvalCodeExecution");
    }
    if (["sandbox_file", "file_write", "file_delete", "file_str_replace"].includes(name)) {
        const path = (args.path as string) || (args.filename as string) || "unknown";
        if (name.includes("delete")) {
            return t("approvalFileDelete", { path });
        }
        return t("approvalFileModify", { path });
    }

    return t("approvalGenericTool", { toolName: name });
}

export function AskUserInput({ interrupt, onRespond, onCancel }: AskUserInputProps) {
    const t = useTranslations("hitl");
    const approvalMessage = useApprovalMessage(interrupt);

    const handleExpire = useCallback(() => {
        onRespond({
            interrupt_id: interrupt.interrupt_id,
            action: interrupt.default_action as InterruptResponse["action"] || "skip",
        });
    }, [interrupt, onRespond]);

    const handleSelect = useCallback((value: string) => {
        onRespond({
            interrupt_id: interrupt.interrupt_id,
            action: "select",
            value,
        });
    }, [interrupt, onRespond]);

    const handleInput = useCallback((value: string) => {
        onRespond({
            interrupt_id: interrupt.interrupt_id,
            action: "input",
            value,
        });
    }, [interrupt, onRespond]);

    const handleSkip = useCallback(() => {
        onRespond({
            interrupt_id: interrupt.interrupt_id,
            action: "skip",
        });
    }, [interrupt, onRespond]);

    const handleBoolean = useCallback((value: boolean) => {
        onRespond({
            interrupt_id: interrupt.interrupt_id,
            action: value ? "approve" : "deny",
            value: value ? "yes" : "no",
        });
    }, [interrupt, onRespond]);

    const handleApprove = useCallback(() => {
        onRespond({
            interrupt_id: interrupt.interrupt_id,
            action: "approve",
        });
    }, [interrupt, onRespond]);

    const handleDeny = useCallback(() => {
        onRespond({
            interrupt_id: interrupt.interrupt_id,
            action: "deny",
        });
    }, [interrupt, onRespond]);

    const isApproval = interrupt.interrupt_type === "approval";
    const isDecision = interrupt.interrupt_type === "decision" && interrupt.options;
    const isInput = interrupt.interrupt_type === "input";
    const isConfirm = interrupt.interrupt_type === "confirm";

    return (
        <div className="bg-secondary/30 border border-border/50 rounded-xl p-5 space-y-4 animate-fade-in">
            {/* Header with question and timer */}
            <div className="flex items-start justify-between gap-4">
                <div className="flex items-start gap-3 flex-1 min-w-0">
                    <MessageSquare className="w-4 h-4 text-foreground mt-0.5 flex-shrink-0" />
                    <p className="text-sm text-foreground leading-relaxed font-medium whitespace-pre-wrap">
                        {isApproval ? approvalMessage : interrupt.message}
                    </p>
                </div>
                <CountdownBadge
                    seconds={interrupt.timeout_seconds}
                    onExpire={handleExpire}
                />
            </div>

            {/* Input area */}
            <div className="pl-7">
                {isApproval && (
                    <ApprovalInput
                        onApprove={handleApprove}
                        onDeny={handleDeny}
                    />
                )}
                {isDecision && interrupt.options && (
                    <DecisionInput
                        options={interrupt.options}
                        onSelect={handleSelect}
                    />
                )}
                {isInput && (
                    <TextInput
                        onSubmit={handleInput}
                        placeholder={t("inputPlaceholder")}
                    />
                )}
                {isConfirm && (
                    <BooleanInput onSelect={handleBoolean} />
                )}
            </div>

            {/* Skip option */}
            <div className="pl-7 pt-1">
                <button
                    onClick={handleSkip}
                    className={cn(
                        "text-xs text-muted-foreground hover:text-foreground transition-colors",
                        "focus-visible:ring-2 focus-visible:ring-primary focus-visible:outline-none rounded px-1 py-0.5"
                    )}
                >
                    {t("skip")}
                </button>
            </div>
        </div>
    );
}
