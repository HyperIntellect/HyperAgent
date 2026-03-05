"use client";

import { useState, useEffect } from "react";
import { useRouter, usePathname } from "next/navigation";
import { useSession } from "next-auth/react";
import { Plus, ChevronDown } from "lucide-react";
import { useTranslations } from "next-intl";
import { cn } from "@/lib/utils";
import { useProjectStore } from "@/lib/stores/project-store";
import { COLOR_MAP } from "@/lib/utils/project-colors";
import { CreateProjectDialog } from "@/components/projects/create-project-dialog";

interface SidebarProjectsProps {
    variant: "desktop" | "mobile";
    onNavigate?: () => void;
}

export function SidebarProjects({ variant: _variant, onNavigate }: SidebarProjectsProps) {
    const router = useRouter();
    const pathname = usePathname();
    const t = useTranslations("sidebar");
    const { status } = useSession();
    const [collapsed, setCollapsed] = useState(false);
    const [createDialogOpen, setCreateDialogOpen] = useState(false);

    const projects = useProjectStore((state) => state.projects);
    const hasHydrated = useProjectStore((state) => state.hasHydrated);
    const loadProjects = useProjectStore.getState().loadProjects;
    const createProject = useProjectStore.getState().createProject;

    useEffect(() => {
        if (status === "authenticated" && !hasHydrated) {
            loadProjects();
        } else if (status === "unauthenticated") {
            useProjectStore.setState({ hasHydrated: true });
        }
    }, [status, hasHydrated, loadProjects]);

    if (!hasHydrated) return null;

    const handleProjectClick = (projectId: string) => {
        router.push(`/projects/${projectId}`);
        onNavigate?.();
    };

    const handleCreateProject = async (data: {
        name: string;
        description?: string;
        color?: string;
    }) => {
        try {
            const id = await createProject(data);
            setCreateDialogOpen(false);
            router.push(`/projects/${id}`);
            onNavigate?.();
        } catch (error) {
            console.error("Failed to create project:", error);
        }
    };

    return (
        <>
            <div className="flex flex-col">
                {/* Section header — matches RecentTasks header style */}
                <div className="px-3 py-2 flex items-center justify-between">
                    <span className="text-xs font-medium text-muted-foreground">
                        {t("projects")}
                    </span>
                    <div className="flex items-center gap-0.5">
                        <button
                            onClick={() => setCreateDialogOpen(true)}
                            className="w-5 h-5 rounded-md flex items-center justify-center text-muted-foreground hover:text-foreground hover:bg-accent transition-colors"
                        >
                            <Plus className="w-4 h-4" strokeWidth={2} />
                        </button>
                        <button
                            onClick={() => setCollapsed(!collapsed)}
                            className="w-5 h-5 rounded-md flex items-center justify-center text-muted-foreground hover:text-foreground transition-colors"
                        >
                            <ChevronDown
                                className={cn(
                                    "w-3.5 h-3.5 transition-transform duration-200",
                                    collapsed && "-rotate-90"
                                )}
                            />
                        </button>
                    </div>
                </div>

                {/* Project list */}
                {!collapsed && (
                    <div className="px-3 pb-3 max-h-[240px] overflow-y-auto">
                        <div className="space-y-0.5">
                            {projects.map((project) => {
                                const isActive = pathname === `/projects/${project.id}`;
                                const itemCount =
                                    (project.conversation_count || 0) +
                                    (project.research_task_count || 0);

                                return (
                                    <div
                                        key={project.id}
                                        className={cn(
                                            "group relative flex items-center gap-3 px-2.5 py-2.5 rounded-lg cursor-pointer transition-colors",
                                            isActive
                                                ? "bg-accent text-accent-foreground"
                                                : "hover:bg-accent"
                                        )}
                                        onClick={() => handleProjectClick(project.id)}
                                    >
                                        {/* Color dot in icon container */}
                                        <div className={cn(
                                            "flex-shrink-0 w-8 h-8 rounded-lg flex items-center justify-center transition-colors",
                                            isActive ? "bg-secondary" : "bg-muted group-hover:bg-secondary"
                                        )}>
                                            <span
                                                className={cn(
                                                    "w-3 h-3 rounded-full",
                                                    COLOR_MAP[project.color || "blue"] || "bg-blue-500"
                                                )}
                                            />
                                        </div>

                                        {/* Name */}
                                        <span
                                            className={cn(
                                                "flex-1 text-sm truncate",
                                                isActive
                                                    ? "text-foreground font-medium"
                                                    : "text-foreground/80 group-hover:text-foreground"
                                            )}
                                        >
                                            {project.name}
                                        </span>

                                        {/* Item count */}
                                        {itemCount > 0 && (
                                            <span className="flex-shrink-0 text-xs text-muted-foreground">
                                                {itemCount}
                                            </span>
                                        )}
                                    </div>
                                );
                            })}

                        </div>
                    </div>
                )}
            </div>

            <CreateProjectDialog
                open={createDialogOpen}
                onClose={() => setCreateDialogOpen(false)}
                onSubmit={handleCreateProject}
            />
        </>
    );
}
