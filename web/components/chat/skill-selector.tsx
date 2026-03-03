"use client";

import React, { useEffect, useState, useRef, useCallback } from "react";
import {
    Sparkles,
    ChevronDown,
    Check,
} from "lucide-react";
import { useTranslations } from "next-intl";
import { cn } from "@/lib/utils";
import { listSkills } from "@/lib/api/skills";
import type { SkillMetadata } from "@/lib/types/skills";
import { getTranslatedSkillName, getTranslatedSkillDescription } from "@/lib/utils/skill-i18n";
import { SKILL_ICONS, SKILL_ACCENT, CATEGORY_ICONS, CATEGORY_ACCENT, DEFAULT_ACCENT } from "@/lib/utils/skill-categories";

type IconComponent = React.ComponentType<{ className?: string }>;

function getSkillIcon(skill: SkillMetadata): IconComponent {
    return SKILL_ICONS[skill.id] || CATEGORY_ICONS[skill.category] || Sparkles;
}

function getSkillStyle(skill: SkillMetadata) {
    const accent = SKILL_ACCENT[skill.id] || CATEGORY_ACCENT[skill.category as keyof typeof CATEGORY_ACCENT];
    if (accent) return { bg: accent.bg, text: accent.icon };
    return DEFAULT_STYLE;
}

const DEFAULT_STYLE = { bg: "bg-muted", text: "text-muted-foreground" };

interface SkillSelectorProps {
    value: string | null;
    onChange: (skillId: string | null) => void;
    disabled?: boolean;
}

export function SkillSelector({ value, onChange, disabled }: SkillSelectorProps) {
    const t = useTranslations("chat.skillSelector");
    const tSkills = useTranslations("skills");
    const [skills, setSkills] = useState<SkillMetadata[]>([]);
    const [isOpen, setIsOpen] = useState(false);
    const containerRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        listSkills()
            .then((data) => setSkills(data.filter((s) => s.enabled)))
            .catch(() => {});
    }, []);

    // Close on click outside
    useEffect(() => {
        if (!isOpen) return;
        const handleClick = (e: MouseEvent) => {
            if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
                setIsOpen(false);
            }
        };
        document.addEventListener("mousedown", handleClick);
        return () => document.removeEventListener("mousedown", handleClick);
    }, [isOpen]);

    // Close on Escape
    useEffect(() => {
        if (!isOpen) return;
        const handleKeyDown = (e: KeyboardEvent) => {
            if (e.key === "Escape") setIsOpen(false);
        };
        document.addEventListener("keydown", handleKeyDown);
        return () => document.removeEventListener("keydown", handleKeyDown);
    }, [isOpen]);

    const selectedSkill = skills.find((s) => s.id === value);
    const displayLabel = selectedSkill
        ? getTranslatedSkillName(selectedSkill.id, selectedSkill.name, tSkills)
        : t("auto");

    const TriggerIcon = selectedSkill
        ? getSkillIcon(selectedSkill)
        : Sparkles;

    const triggerStyle = selectedSkill
        ? getSkillStyle(selectedSkill)
        : null;

    const handleSelect = useCallback((skillId: string | null) => {
        onChange(skillId);
        setIsOpen(false);
    }, [onChange]);

    return (
        <div ref={containerRef} className="relative">
            {/* Trigger */}
            <button
                type="button"
                onClick={() => !disabled && setIsOpen(!isOpen)}
                disabled={disabled}
                className={cn(
                    "inline-flex items-center gap-1.5 rounded-lg px-2 py-1 h-auto",
                    "text-xs font-medium transition-all duration-150 cursor-pointer select-none",
                    "hover:bg-secondary/80 active:scale-[0.97]",
                    "disabled:opacity-50 disabled:cursor-not-allowed",
                    "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40 focus-visible:ring-offset-1",
                    isOpen && "bg-secondary/80",
                    value
                        ? triggerStyle ? triggerStyle.text : "text-foreground"
                        : "text-muted-foreground"
                )}
            >
                <TriggerIcon className="w-3.5 h-3.5" />
                <span className="max-w-[80px] truncate">{displayLabel}</span>
                <ChevronDown
                    className={cn(
                        "w-3 h-3 opacity-60 transition-transform duration-200",
                        isOpen && "rotate-180"
                    )}
                />
            </button>

            {/* Dropdown panel — 2-column grid */}
            {isOpen && (
                <div
                    className={cn(
                        "absolute bottom-full left-0 mb-2 z-50",
                        "w-[420px] rounded-xl overflow-hidden",
                        "bg-popover border border-border/80 shadow-xl shadow-black/8 dark:shadow-black/25",
                        "animate-in fade-in-0 slide-in-from-bottom-2 duration-150"
                    )}
                >
                    <div className="p-2 grid grid-cols-2 gap-1">
                        {/* Auto option */}
                        <button
                            type="button"
                            onClick={() => handleSelect(null)}
                            className={cn(
                                "flex items-center gap-2.5 px-2.5 py-2.5 rounded-lg text-left transition-colors",
                                "hover:bg-accent/60",
                                !value && "bg-primary/[0.06] dark:bg-primary/[0.08] ring-1 ring-primary/15"
                            )}
                        >
                            <div
                                className={cn(
                                    "flex items-center justify-center w-7 h-7 rounded-md shrink-0",
                                    !value
                                        ? "bg-primary/12 text-primary"
                                        : "bg-muted text-muted-foreground"
                                )}
                            >
                                <Sparkles className="w-3.5 h-3.5" />
                            </div>
                            <div className="flex-1 min-w-0">
                                <span className="text-[13px] font-medium text-foreground block truncate">
                                    {t("auto")}
                                </span>
                                <span className="text-[11px] text-muted-foreground block truncate leading-tight mt-0.5">
                                    {t("autoDescription")}
                                </span>
                            </div>
                            {!value && (
                                <Check className="w-3.5 h-3.5 text-primary shrink-0" />
                            )}
                        </button>

                        {/* Skill cards */}
                        {skills.map((skill) => {
                            const Icon = getSkillIcon(skill);
                            const style = getSkillStyle(skill);
                            const isSelected = value === skill.id;

                            return (
                                <button
                                    key={skill.id}
                                    type="button"
                                    onClick={() => handleSelect(skill.id)}
                                    className={cn(
                                        "flex items-center gap-2.5 px-2.5 py-2.5 rounded-lg text-left transition-colors",
                                        "hover:bg-accent/60",
                                        isSelected && "bg-primary/[0.06] dark:bg-primary/[0.08] ring-1 ring-primary/15"
                                    )}
                                >
                                    <div
                                        className={cn(
                                            "flex items-center justify-center w-7 h-7 rounded-md shrink-0",
                                            style.bg,
                                            style.text
                                        )}
                                    >
                                        <Icon className="w-3.5 h-3.5" />
                                    </div>
                                    <div className="flex-1 min-w-0">
                                        <span className="text-[13px] font-medium text-foreground block truncate">
                                            {getTranslatedSkillName(skill.id, skill.name, tSkills)}
                                        </span>
                                        <span className="text-[11px] text-muted-foreground block truncate leading-tight mt-0.5">
                                            {getTranslatedSkillDescription(skill.id, skill.description, tSkills)}
                                        </span>
                                    </div>
                                    {isSelected && (
                                        <Check className="w-3.5 h-3.5 text-primary shrink-0" />
                                    )}
                                </button>
                            );
                        })}
                    </div>
                </div>
            )}
        </div>
    );
}
