"use client";

import { useRouter } from "next/navigation";
import { useTranslations } from "next-intl";
import { SkillMetadata } from "@/lib/types/skills";
import { Button } from "@/components/ui/button";
import { ArrowRight, Code } from "lucide-react";
import { cn } from "@/lib/utils";
import { SKILL_ICONS, SKILL_ACCENT, CATEGORY_ICONS, CATEGORY_ACCENT, DEFAULT_ACCENT } from "@/lib/utils/skill-categories";
import { getTranslatedSkillName, getTranslatedSkillDescription, getTranslatedCategory } from "@/lib/utils/skill-i18n";

interface SkillCardProps {
  skill: SkillMetadata;
  onExecute?: (skillId: string) => void;
  index?: number;
}

export function SkillCard({ skill, onExecute, index = 0 }: SkillCardProps) {
  const t = useTranslations("skills");
  const router = useRouter();
  const Icon =
    SKILL_ICONS[skill.id] || CATEGORY_ICONS[skill.category as keyof typeof CATEGORY_ICONS] || Code;
  const accent =
    SKILL_ACCENT[skill.id] || CATEGORY_ACCENT[skill.category as keyof typeof CATEGORY_ACCENT] || DEFAULT_ACCENT;

  const paramCount = skill.parameters?.length ?? 0;

  return (
    <div
      className={cn(
        "group relative border border-border/50 rounded-xl p-4",
        "hover:border-border hover:shadow-sm",
        "transition-all duration-200",
        "cursor-pointer",
        accent.border,
        "animate-fade-in"
      )}
      style={{ animationDelay: `${index * 40}ms`, animationFillMode: "backwards" }}
      onClick={() => router.push(`/skills/${skill.id}`)}
    >
      {/* Header row: icon + name + version */}
      <div className="flex items-start gap-3">
        <div className={cn("shrink-0 p-2 rounded-lg", accent.bg)}>
          <Icon className={cn("w-4 h-4", accent.icon)} />
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <h4 className="font-semibold text-sm text-foreground truncate">
              {getTranslatedSkillName(skill.id, skill.name, t)}
            </h4>
            <span className="shrink-0 text-[10px] text-muted-foreground/60 font-mono">
              v{skill.version}
            </span>
          </div>
          <p className="text-xs text-muted-foreground mt-1 line-clamp-2 leading-relaxed">
            {getTranslatedSkillDescription(skill.id, skill.description, t)}
          </p>
        </div>
      </div>

      {/* Tags */}
      {skill.tags && skill.tags.length > 0 && (
        <div className="flex flex-wrap gap-1 mt-3">
          {skill.tags.slice(0, 4).map((tag) => (
            <span
              key={tag}
              className={cn(
                "inline-flex text-[10px] font-medium px-1.5 py-0.5 rounded",
                accent.tag
              )}
            >
              {tag}
            </span>
          ))}
          {skill.tags.length > 4 && (
            <span className="text-[10px] text-muted-foreground/50 px-1 py-0.5">
              +{skill.tags.length - 4}
            </span>
          )}
        </div>
      )}

      {/* Footer: metadata + action */}
      <div className="flex items-center justify-between mt-3 pt-3 border-t border-border/30">
        <div className="flex items-center gap-2 text-[10px] text-muted-foreground">
          <span className="uppercase tracking-wider font-medium">
            {getTranslatedCategory(skill.category, t)}
          </span>
          {paramCount > 0 && (
            <>
              <span className="text-border">|</span>
              <span>{t("parameters", { count: paramCount })}</span>
            </>
          )}
        </div>

        {onExecute && (
          <Button
            size="sm"
            variant="ghost"
            className="h-7 px-2.5 text-xs gap-1 cursor-pointer opacity-0 group-hover:opacity-100 transition-opacity duration-150"
            onClick={(e) => {
              e.stopPropagation();
              onExecute(skill.id);
            }}
          >
            {t("useSkill")}
            <ArrowRight className="w-3 h-3" />
          </Button>
        )}
      </div>
    </div>
  );
}
