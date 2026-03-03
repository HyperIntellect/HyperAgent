"use client";

import React, { useState, useEffect, useCallback } from "react";
import { useTranslations } from "next-intl";
import {
    X,
    Shield,
    AlertTriangle,
    MessageSquare,
    Check,
    Clock,
    Code,
    Globe,
    FileText,
    Sparkles,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import type { InterruptEvent, InterruptResponse, InterruptOption } from "@/lib/types";
import { getTranslatedSkillName } from "@/lib/utils/skill-i18n";

interface InterruptDialogProps {
    interrupt: InterruptEvent;
    onRespond: (response: InterruptResponse) => void;
    onCancel: () => void;
}

// Get icon for tool type
function getToolIcon(toolName: string) {
    const lowerName = toolName.toLowerCase();
    if (lowerName.includes("browser") || lowerName.includes("navigate")) {
        return Globe;
    }
    if (lowerName.includes("code") || lowerName.includes("execute") || lowerName.includes("python")) {
        return Code;
    }
    if (lowerName.includes("file") || lowerName.includes("sandbox")) {
        return FileText;
    }
    if (lowerName === "invoke_skill" || lowerName.startsWith("invoke_skill:")) {
        return Sparkles;
    }
    return AlertTriangle;
}

// Get display name for tool, with translated skill names
function getToolDisplayName(
    toolName: string,
    args: Record<string, unknown>,
    tSkills: (key: string) => string,
): string {
    const lowerName = toolName.toLowerCase();
    if (lowerName === "invoke_skill" || lowerName.startsWith("invoke_skill:")) {
        const skillId = (args.skill_id as string) || toolName.split(":")[1] || "unknown";
        return getTranslatedSkillName(skillId, skillId, tSkills);
    }
    return toolName.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

// Format tool arguments for display
function formatToolArgs(args: Record<string, unknown>, noArgsLabel: string): string {
    const entries = Object.entries(args);
    if (entries.length === 0) return noArgsLabel;

    return entries
        .filter(([key]) => key !== "user_id" && key !== "task_id") // Hide internal args
        .map(([key, value]) => {
            const displayValue = typeof value === "string"
                ? value.length > 100 ? value.slice(0, 100) + "..." : value
                : JSON.stringify(value).slice(0, 100);
            return `${key}: ${displayValue}`;
        })
        .join("\n");
}

// Countdown timer component
function CountdownTimer({ seconds, onExpire }: { seconds: number; onExpire: () => void }) {
    const [remaining, setRemaining] = useState(seconds);

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
        <div className={cn(
            "flex items-center gap-1.5 text-sm",
            isLow ? "text-destructive" : "text-muted-foreground"
        )}>
            <Clock className="w-3.5 h-3.5" />
            <span className="tabular-nums">
                {minutes}:{secs.toString().padStart(2, "0")}
            </span>
        </div>
    );
}

// Generate localized approval message for tool
function useDialogApprovalMessage(interrupt: InterruptEvent): string {
    const t = useTranslations("hitl");
    const tSkills = useTranslations("skills");

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
    if (lowerName === "invoke_skill" || lowerName.startsWith("invoke_skill:")) {
        const skillId = (args.skill_id as string) || name.split(":")[1] || "unknown";
        const skillName = getTranslatedSkillName(skillId, skillId, tSkills);
        return t("approvalSkillInvocation", { skillName });
    }

    return t("approvalGenericTool", { toolName: name });
}

// Approval dialog content
function ApprovalContent({
    interrupt,
    onRespond,
}: {
    interrupt: InterruptEvent;
    onRespond: (response: InterruptResponse) => void;
}) {
    const t = useTranslations("hitl");
    const tSkills = useTranslations("skills");
    const toolInfo = interrupt.tool_info;
    const ToolIcon = toolInfo ? getToolIcon(toolInfo.name) : AlertTriangle;
    const approvalMessage = useDialogApprovalMessage(interrupt);

    return (
        <div className="space-y-4">
            {/* Tool info card */}
            {toolInfo && (
                <div className="bg-secondary/50 border border-border/50 rounded-lg p-4">
                    <div className="flex items-start gap-3">
                        <div className="w-10 h-10 rounded-lg bg-background flex items-center justify-center flex-shrink-0">
                            <ToolIcon className="w-5 h-5 text-muted-foreground" />
                        </div>
                        <div className="flex-1 min-w-0">
                            <h4 className="text-sm font-medium text-foreground">
                                {getToolDisplayName(toolInfo.name, toolInfo.args, tSkills)}
                            </h4>
                            <pre className="mt-2 text-xs text-muted-foreground whitespace-pre-wrap break-all font-mono bg-background/50 rounded p-2 max-h-32 overflow-auto">
                                {formatToolArgs(toolInfo.args, t("noArguments"))}
                            </pre>
                        </div>
                    </div>
                </div>
            )}

            {/* Message */}
            <p className="text-sm text-foreground/80 whitespace-pre-wrap">
                {approvalMessage}
            </p>

            {/* Action buttons */}
            <div className="flex flex-col gap-2">
                <div className="flex gap-2">
                    <Button
                        variant="primary"
                        className="flex-1"
                        onClick={() => onRespond({
                            interrupt_id: interrupt.interrupt_id,
                            action: "approve",
                        })}
                    >
                        <Check className="w-4 h-4" />
                        {t("approve")}
                    </Button>
                    <Button
                        variant="destructive"
                        className="flex-1"
                        onClick={() => onRespond({
                            interrupt_id: interrupt.interrupt_id,
                            action: "deny",
                        })}
                    >
                        <X className="w-4 h-4" />
                        {t("deny")}
                    </Button>
                </div>
                <Button
                    variant="ghost"
                    size="sm"
                    className="text-muted-foreground"
                    onClick={() => onRespond({
                        interrupt_id: interrupt.interrupt_id,
                        action: "approve_always",
                    })}
                >
                    {t("approveAlways")}
                </Button>
            </div>
        </div>
    );
}

// Decision dialog content
function DecisionContent({
    interrupt,
    onRespond,
}: {
    interrupt: InterruptEvent;
    onRespond: (response: InterruptResponse) => void;
}) {
    const t = useTranslations("hitl");
    const [selected, setSelected] = useState<string | null>(null);

    const options = interrupt.options || [];

    return (
        <div className="space-y-4">
            {/* Message */}
            <p className="text-sm text-foreground/80 whitespace-pre-wrap">
                {interrupt.message}
            </p>

            {/* Options */}
            <div className="space-y-2">
                {options.map((option: InterruptOption) => (
                    <button
                        key={option.value}
                        className={cn(
                            "w-full text-left p-3 rounded-lg border transition-colors",
                            selected === option.value
                                ? "border-accent-cyan bg-accent-cyan/10"
                                : "border-border hover:border-border/80 hover:bg-secondary/50"
                        )}
                        onClick={() => setSelected(option.value)}
                    >
                        <div className="font-medium text-sm">{option.label}</div>
                        {option.description && (
                            <div className="text-xs text-muted-foreground mt-1">
                                {option.description}
                            </div>
                        )}
                    </button>
                ))}
            </div>

            {/* Action buttons */}
            <div className="flex gap-2">
                <Button
                    variant="primary"
                    className="flex-1"
                    disabled={!selected}
                    onClick={() => selected && onRespond({
                        interrupt_id: interrupt.interrupt_id,
                        action: "select",
                        value: selected,
                    })}
                >
                    {t("confirm")}
                </Button>
                <Button
                    variant="secondary"
                    onClick={() => onRespond({
                        interrupt_id: interrupt.interrupt_id,
                        action: "skip",
                    })}
                >
                    {t("skip")}
                </Button>
            </div>
        </div>
    );
}

// Input dialog content
function InputContent({
    interrupt,
    onRespond,
}: {
    interrupt: InterruptEvent;
    onRespond: (response: InterruptResponse) => void;
}) {
    const t = useTranslations("hitl");
    const [value, setValue] = useState("");
    const textareaRef = React.useRef<HTMLTextAreaElement>(null);

    // Auto-focus the textarea when mounted
    useEffect(() => {
        const timer = setTimeout(() => {
            textareaRef.current?.focus();
        }, 100);
        return () => clearTimeout(timer);
    }, []);

    const handleSubmit = () => {
        if (value.trim()) {
            onRespond({
                interrupt_id: interrupt.interrupt_id,
                action: "input",
                value: value.trim(),
            });
        }
    };

    const handleKeyDown = (e: React.KeyboardEvent) => {
        // Submit on Ctrl+Enter or Cmd+Enter
        if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
            e.preventDefault();
            handleSubmit();
        }
    };

    return (
        <div className="space-y-4">
            {/* Message */}
            <p className="text-sm text-foreground/80 whitespace-pre-wrap">
                {interrupt.message}
            </p>

            {/* Input */}
            <Textarea
                ref={textareaRef}
                value={value}
                onChange={(e) => setValue(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder={t("inputPlaceholder")}
                rows={3}
                className="resize-none"
                autoFocus
            />

            {/* Action buttons */}
            <div className="flex gap-2">
                <Button
                    variant="primary"
                    className="flex-1"
                    disabled={!value.trim()}
                    onClick={handleSubmit}
                >
                    {t("submit")}
                </Button>
                <Button
                    variant="secondary"
                    onClick={() => onRespond({
                        interrupt_id: interrupt.interrupt_id,
                        action: "skip",
                    })}
                >
                    {t("skip")}
                </Button>
            </div>
        </div>
    );
}

// Confirm dialog content (Yes/No)
function ConfirmContent({
    interrupt,
    onRespond,
}: {
    interrupt: InterruptEvent;
    onRespond: (response: InterruptResponse) => void;
}) {
    const t = useTranslations("hitl");

    return (
        <div className="space-y-4">
            {/* Message */}
            <p className="text-sm text-foreground/80 whitespace-pre-wrap">
                {interrupt.message}
            </p>

            {/* Action buttons */}
            <div className="flex flex-col gap-2">
                <div className="flex gap-2">
                    <Button
                        variant="primary"
                        className="flex-1"
                        onClick={() => onRespond({
                            interrupt_id: interrupt.interrupt_id,
                            action: "approve",
                            value: "yes",
                        })}
                    >
                        <Check className="w-4 h-4" />
                        {t("yes")}
                    </Button>
                    <Button
                        variant="secondary"
                        className="flex-1"
                        onClick={() => onRespond({
                            interrupt_id: interrupt.interrupt_id,
                            action: "deny",
                            value: "no",
                        })}
                    >
                        <X className="w-4 h-4" />
                        {t("no")}
                    </Button>
                </div>
                <Button
                    variant="ghost"
                    size="sm"
                    className="text-muted-foreground"
                    onClick={() => onRespond({
                        interrupt_id: interrupt.interrupt_id,
                        action: "skip",
                    })}
                >
                    {t("skip")}
                </Button>
            </div>
        </div>
    );
}

export function InterruptDialog({ interrupt, onRespond, onCancel }: InterruptDialogProps) {
    const t = useTranslations("hitl");

    // Handle timeout expiration
    const handleExpire = useCallback(() => {
        onRespond({
            interrupt_id: interrupt.interrupt_id,
            action: interrupt.default_action as InterruptResponse["action"] || "deny",
        });
    }, [interrupt, onRespond]);

    // Get dialog icon based on type
    const DialogIcon = interrupt.interrupt_type === "approval"
        ? Shield
        : interrupt.interrupt_type === "decision"
            ? AlertTriangle
            : interrupt.interrupt_type === "confirm"
                ? Check
                : MessageSquare;

    // Prevent body scroll when dialog is open
    useEffect(() => {
        document.body.style.overflow = "hidden";
        return () => {
            document.body.style.overflow = "";
        };
    }, []);

    return (
        <>
            {/* Backdrop - higher z-index to be above everything */}
            <div
                className="fixed inset-0 bg-black/40 z-[100] animate-in fade-in duration-200"
                onClick={onCancel}
            />

            {/* Dialog - higher z-index */}
            <div className="fixed inset-0 z-[101] flex items-center justify-center p-4 pointer-events-none">
                <div
                    className={cn(
                        "bg-background border border-border rounded-xl shadow-sm pointer-events-auto",
                        "w-full max-w-md max-h-[85vh] overflow-hidden",
                        "animate-in zoom-in-95 fade-in duration-200"
                    )}
                    onClick={(e) => e.stopPropagation()}
                >
                    {/* Header */}
                    <div className="flex items-center justify-between px-4 py-3 border-b border-border">
                        <div className="flex items-center gap-3">
                            <div className={cn(
                                "w-8 h-8 rounded-lg flex items-center justify-center",
                                interrupt.interrupt_type === "approval"
                                    ? "bg-amber-500/10 text-amber-500"
                                    : "bg-accent-cyan/10 text-accent-cyan"
                            )}>
                                <DialogIcon className="w-4 h-4" />
                            </div>
                            <h2 className="text-sm font-medium text-foreground">
                                {interrupt.interrupt_type === "approval"
                                    ? t("toolApproval")
                                    : interrupt.interrupt_type === "decision"
                                        ? t("decisionRequired")
                                        : interrupt.interrupt_type === "input"
                                            ? t("inputRequired")
                                            : interrupt.interrupt_type === "confirm"
                                                ? t("confirmRequired")
                                                : interrupt.title}
                            </h2>
                        </div>
                        <div className="flex items-center gap-3">
                            <CountdownTimer
                                seconds={interrupt.timeout_seconds}
                                onExpire={handleExpire}
                            />
                            <button
                                onClick={onCancel}
                                className="p-1.5 -m-1.5 rounded-md hover:bg-secondary transition-colors"
                            >
                                <X className="w-4 h-4 text-muted-foreground" />
                            </button>
                        </div>
                    </div>

                    {/* Content */}
                    <div className="p-4 overflow-y-auto max-h-[calc(85vh-60px)]">
                        {interrupt.interrupt_type === "approval" && (
                            <ApprovalContent interrupt={interrupt} onRespond={onRespond} />
                        )}
                        {interrupt.interrupt_type === "decision" && (
                            <DecisionContent interrupt={interrupt} onRespond={onRespond} />
                        )}
                        {interrupt.interrupt_type === "input" && (
                            <InputContent interrupt={interrupt} onRespond={onRespond} />
                        )}
                        {interrupt.interrupt_type === "confirm" && (
                            <ConfirmContent interrupt={interrupt} onRespond={onRespond} />
                        )}
                    </div>
                </div>
            </div>
        </>
    );
}
