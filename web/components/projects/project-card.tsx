"use client";

import { useRouter } from "next/navigation";
import { useTranslations } from "next-intl";
import { Layers } from "lucide-react";
import { cn } from "@/lib/utils";
import { COLOR_MAP } from "@/lib/utils/project-colors";
import type { Project } from "@/lib/types/projects";

interface ProjectCardProps {
  project: Project;
  index?: number;
}

export function ProjectCard({ project, index = 0 }: ProjectCardProps) {
  const router = useRouter();
  const t = useTranslations("projects");

  const colorClass = project.color
    ? COLOR_MAP[project.color] || "bg-muted-foreground"
    : "bg-muted-foreground";

  return (
    <div
      onClick={() => router.push(`/projects/${project.id}`)}
      className={cn(
        "group relative border border-border/50 rounded-xl p-4",
        "hover:border-border hover:shadow-sm hover:bg-secondary/30",
        "transition-all duration-200 cursor-pointer",
        "animate-fade-in"
      )}
      style={{
        animationDelay: `${index * 40}ms`,
        animationFillMode: "backwards",
      }}
    >
      {/* Header */}
      <div className="flex items-start gap-3">
        <div
          className={cn("w-3 h-3 rounded-full shrink-0 mt-1", colorClass)}
        />
        <div className="flex-1 min-w-0">
          <h4 className="font-semibold text-sm text-foreground truncate">
            {project.name}
          </h4>
          {project.description && (
            <p className="text-xs text-muted-foreground mt-1 line-clamp-2 leading-relaxed">
              {project.description}
            </p>
          )}
        </div>
      </div>

      {/* Footer: unified item count */}
      <div className="flex items-center gap-3 mt-3 pt-3 border-t border-border/30">
        <div className="flex items-center gap-1 text-[10px] text-muted-foreground">
          <Layers className="w-3 h-3" />
          <span>
            {t("itemCount", {
              count:
                (project.conversation_count ?? 0) +
                (project.research_task_count ?? 0),
            })}
          </span>
        </div>
      </div>
    </div>
  );
}
