"use client";

import { useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { useTranslations } from "next-intl";
import {
  ArrowLeft,
  Loader2,
  MessageSquare,
  Search,
  Trash2,
  Plus,
  X,
  CheckCircle2,
  AlertCircle,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { COLOR_MAP } from "@/lib/utils/project-colors";
import { formatRelativeTime } from "@/lib/utils/relative-time";
import { useProjectStore } from "@/lib/stores/project-store";
import { useChatStore } from "@/lib/stores/chat-store";
import { useTaskStore } from "@/lib/stores/task-store";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  PROJECT_COLORS,
  type ProjectColor,
  type ProjectItem,
  getProjectItemDate,
  getProjectItemTitle,
} from "@/lib/types/projects";

interface ProjectDetailViewProps {
  projectId: string;
}

function ProjectItemRow({
  item,
  t,
  onRemove,
  onClick,
}: {
  item: ProjectItem;
  t: (key: string, params?: Record<string, unknown>) => string;
  onRemove: () => void;
  onClick: () => void;
}) {
  const isConversation = item.kind === "conversation";
  const title = getProjectItemTitle(item);
  const dateStr = getProjectItemDate(item);

  return (
    <div
      onClick={onClick}
      className="flex items-center justify-between px-3 py-2.5 rounded-lg hover:bg-secondary/30 transition-colors group cursor-pointer"
    >
      <div className="flex items-center gap-2.5 min-w-0 flex-1">
        {isConversation ? (
          <MessageSquare className="w-3.5 h-3.5 text-muted-foreground shrink-0" />
        ) : (
          <Search className="w-3.5 h-3.5 text-muted-foreground shrink-0" />
        )}
        <span className="text-sm truncate">{title}</span>
        <span className="text-[10px] uppercase px-1.5 py-0.5 rounded-full bg-secondary text-muted-foreground shrink-0">
          {isConversation ? t("conversationsSection") : t("tasksSection")}
        </span>
        {!isConversation && (
          <span className="shrink-0">
            {item.data.status === "running" && (
              <Loader2 className="w-3.5 h-3.5 animate-spin text-muted-foreground" />
            )}
            {item.data.status === "completed" && (
              <CheckCircle2 className="w-3.5 h-3.5 text-green-500" />
            )}
            {item.data.status === "failed" && (
              <AlertCircle className="w-3.5 h-3.5 text-destructive" />
            )}
          </span>
        )}
      </div>
      <div className="flex items-center gap-2 shrink-0">
        <span className="text-xs text-muted-foreground">
          {formatRelativeTime(dateStr, t as (key: string, params?: Record<string, number>) => string)}
        </span>
        <button
          onClick={(e) => {
            e.stopPropagation();
            onRemove();
          }}
          className="text-xs text-muted-foreground hover:text-destructive opacity-0 group-hover:opacity-100 transition-all cursor-pointer"
        >
          {t("removeFromProject")}
        </button>
      </div>
    </div>
  );
}

export function ProjectDetailView({ projectId }: ProjectDetailViewProps) {
  const t = useTranslations("projects");
  const router = useRouter();

  const {
    activeProject,
    isLoading,
    loadProject,
    updateProject,
    deleteProject,
    removeItems,
    assignItems,
  } = useProjectStore();

  const conversations = useChatStore((state) => state.conversations);
  const chatHydrated = useChatStore((state) => state.hasHydrated);

  const allTasks = useTaskStore((state) => state.tasks);
  const taskHydrated = useTaskStore((state) => state.hasHydrated);

  const [isEditing, setIsEditing] = useState(false);
  const [editName, setEditName] = useState("");
  const [editDescription, setEditDescription] = useState("");
  const [editColor, setEditColor] = useState<ProjectColor>("blue");
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const [showAddItems, setShowAddItems] = useState(false);
  const [selectedConvIds, setSelectedConvIds] = useState<string[]>([]);
  const [selectedTaskIds, setSelectedTaskIds] = useState<string[]>([]);

  useEffect(() => {
    loadProject(projectId);
  }, [projectId]);

  useEffect(() => {
    if (activeProject) {
      setEditName(activeProject.name);
      setEditDescription(activeProject.description || "");
      setEditColor((activeProject.color as ProjectColor) || "blue");
    }
  }, [activeProject]);

  // Unified timeline: merge conversations and research tasks, sorted by date descending
  const timelineItems = useMemo<ProjectItem[]>(() => {
    if (!activeProject) return [];
    const items: ProjectItem[] = [
      ...activeProject.conversations.map(
        (c) => ({ kind: "conversation" as const, data: c })
      ),
      ...activeProject.research_tasks.map(
        (t) => ({ kind: "research_task" as const, data: t })
      ),
    ];
    items.sort(
      (a, b) =>
        new Date(getProjectItemDate(b)).getTime() -
        new Date(getProjectItemDate(a)).getTime()
    );
    return items;
  }, [activeProject]);

  const handleSave = async () => {
    if (!editName.trim()) return;
    try {
      await updateProject(projectId, {
        name: editName.trim(),
        description: editDescription.trim() || undefined,
        color: editColor,
      });
      setIsEditing(false);
    } catch (err) {
      console.error("Failed to update project:", err);
    }
  };

  const handleDelete = async () => {
    try {
      await deleteProject(projectId);
      router.push("/projects");
    } catch (err) {
      console.error("Failed to delete project:", err);
    }
  };

  const handleRemoveItem = async (item: ProjectItem) => {
    if (item.kind === "conversation") {
      await removeItems(projectId, { conversation_ids: [item.data.id] });
    } else {
      await removeItems(projectId, { research_task_ids: [item.data.id] });
    }
  };

  const handleItemClick = (item: ProjectItem) => {
    if (item.kind === "conversation") {
      router.push(`/c/${item.data.id}`);
    } else {
      router.push(`/tasks/${item.data.id}`);
    }
  };

  const handleAssignItems = async () => {
    if (selectedConvIds.length === 0 && selectedTaskIds.length === 0) return;
    await assignItems(projectId, {
      conversation_ids: selectedConvIds.length > 0 ? selectedConvIds : undefined,
      research_task_ids: selectedTaskIds.length > 0 ? selectedTaskIds : undefined,
    });
    setSelectedConvIds([]);
    setSelectedTaskIds([]);
    setShowAddItems(false);
  };

  // Conversations not assigned to this project
  const unassignedConversations = chatHydrated
    ? conversations.filter(
        (c) =>
          !activeProject?.conversations.some((pc) => pc.id === c.id)
      )
    : [];

  // Research tasks not assigned to this project
  const unassignedTasks = taskHydrated
    ? allTasks.filter(
        (t) =>
          !activeProject?.research_tasks.some((pt) => pt.id === t.id)
      )
    : [];

  const totalSelectedCount = selectedConvIds.length + selectedTaskIds.length;

  if (isLoading && !activeProject) {
    return (
      <div className="flex flex-col items-center justify-center py-24 gap-3">
        <Loader2 className="w-5 h-5 animate-spin text-muted-foreground" />
      </div>
    );
  }

  if (!activeProject) {
    return (
      <div className="flex flex-col items-center justify-center py-24 gap-3">
        <p className="text-sm text-muted-foreground">Project not found</p>
        <Button
          variant="ghost"
          onClick={() => router.push("/projects")}
          className="cursor-pointer"
        >
          <ArrowLeft className="w-4 h-4 mr-1.5" />
          Back to projects
        </Button>
      </div>
    );
  }

  const colorClass = activeProject.color
    ? COLOR_MAP[activeProject.color] || "bg-muted-foreground"
    : "bg-muted-foreground";

  return (
    <div className="space-y-6">
      {/* Back button */}
      <button
        onClick={() => router.push("/projects")}
        className="flex items-center gap-1.5 text-sm text-muted-foreground hover:text-foreground transition-colors cursor-pointer"
      >
        <ArrowLeft className="w-4 h-4" />
        {t("title")}
      </button>

      {/* Project header */}
      {isEditing ? (
        <div className="space-y-4 border border-border/50 rounded-xl p-4">
          <div>
            <label className="text-sm font-medium mb-1.5 block">
              {t("name")}
            </label>
            <Input
              value={editName}
              onChange={(e) => setEditName(e.target.value)}
              autoFocus
            />
          </div>
          <div>
            <label className="text-sm font-medium mb-1.5 block">
              {t("description")}
            </label>
            <textarea
              value={editDescription}
              onChange={(e) => setEditDescription(e.target.value)}
              rows={2}
              className="flex w-full rounded-lg border border-border bg-transparent px-3 py-2 text-sm transition-colors placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary focus-visible:ring-offset-2 resize-none"
            />
          </div>
          <div>
            <label className="text-sm font-medium mb-1.5 block">
              {t("color")}
            </label>
            <div className="flex items-center gap-2">
              {PROJECT_COLORS.map((c) => (
                <button
                  key={c}
                  type="button"
                  onClick={() => setEditColor(c)}
                  className={cn(
                    "w-6 h-6 rounded-full transition-all cursor-pointer",
                    COLOR_MAP[c],
                    editColor === c
                      ? "ring-2 ring-offset-2 ring-foreground ring-offset-card scale-110"
                      : "hover:scale-110"
                  )}
                />
              ))}
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Button onClick={handleSave} size="sm" className="cursor-pointer">
              {t("save")}
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setIsEditing(false)}
              className="cursor-pointer"
            >
              {t("cancel")}
            </Button>
          </div>
        </div>
      ) : (
        <div className="flex items-start justify-between">
          <div className="flex items-start gap-3">
            <div className={cn("w-4 h-4 rounded-full mt-1", colorClass)} />
            <div>
              <h1 className="text-2xl font-bold">{activeProject.name}</h1>
              {activeProject.description && (
                <p className="text-muted-foreground mt-1">
                  {activeProject.description}
                </p>
              )}
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setIsEditing(true)}
              className="cursor-pointer"
            >
              {t("editProject")}
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setShowDeleteConfirm(true)}
              className="text-destructive hover:text-destructive cursor-pointer"
            >
              <Trash2 className="w-4 h-4" />
            </Button>
          </div>
        </div>
      )}

      {/* Delete confirmation */}
      {showDeleteConfirm && (
        <div className="border border-destructive/50 rounded-xl p-4 bg-destructive/5">
          <p className="text-sm text-foreground">{t("deleteConfirm")}</p>
          <div className="flex items-center gap-2 mt-3">
            <Button
              variant="destructive"
              size="sm"
              onClick={handleDelete}
              className="cursor-pointer"
            >
              {t("deleteProject")}
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setShowDeleteConfirm(false)}
              className="cursor-pointer"
            >
              {t("cancel")}
            </Button>
          </div>
        </div>
      )}

      {/* Unified Items section */}
      <section>
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-sm font-semibold uppercase tracking-wide flex items-center gap-2">
            {t("items")}
            <span className="text-xs text-muted-foreground font-normal">
              ({timelineItems.length})
            </span>
          </h2>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setShowAddItems(true)}
            className="cursor-pointer"
          >
            <Plus className="w-4 h-4 mr-1" />
            {t("addItems")}
          </Button>
        </div>

        {timelineItems.length === 0 ? (
          <p className="text-sm text-muted-foreground py-4">
            {t("noItems")}
          </p>
        ) : (
          <div className="space-y-1">
            {timelineItems.map((item) => (
              <ProjectItemRow
                key={item.data.id}
                item={item}
                t={t as (key: string, params?: Record<string, unknown>) => string}
                onRemove={() => handleRemoveItem(item)}
                onClick={() => handleItemClick(item)}
              />
            ))}
          </div>
        )}
      </section>

      {/* Add Items Dialog */}
      {showAddItems && (
        <div className="fixed inset-0 z-50 flex items-center justify-center">
          <div
            className="absolute inset-0 bg-black/40"
            onClick={() => setShowAddItems(false)}
            aria-hidden="true"
          />
          <div className="relative bg-card border border-border rounded-xl shadow-lg w-full max-w-md mx-4 p-6 max-h-[80vh] overflow-y-auto">
            <button
              onClick={() => setShowAddItems(false)}
              className="absolute top-4 right-4 text-muted-foreground hover:text-foreground transition-colors cursor-pointer"
            >
              <X className="w-4 h-4" />
            </button>

            <h2 className="text-lg font-semibold mb-1">{t("addItems")}</h2>
            <p className="text-sm text-muted-foreground mb-4">
              {t("selectItemsDescription")}
            </p>

            {/* Conversations section */}
            <h3 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground mb-2 flex items-center gap-1.5">
              <MessageSquare className="w-3.5 h-3.5" />
              {t("conversationsSection")}
            </h3>
            {unassignedConversations.length === 0 ? (
              <p className="text-sm text-muted-foreground mb-4">
                {t("noConversations")}
              </p>
            ) : (
              <div className="space-y-1 mb-4">
                {unassignedConversations.map((conv) => (
                  <label
                    key={conv.id}
                    className="flex items-center gap-2 px-3 py-2 rounded-lg hover:bg-secondary/30 transition-colors cursor-pointer"
                  >
                    <input
                      type="checkbox"
                      checked={selectedConvIds.includes(conv.id)}
                      onChange={(e) => {
                        if (e.target.checked) {
                          setSelectedConvIds((prev) => [...prev, conv.id]);
                        } else {
                          setSelectedConvIds((prev) =>
                            prev.filter((id) => id !== conv.id)
                          );
                        }
                      }}
                      className="rounded"
                    />
                    <MessageSquare className="w-3.5 h-3.5 text-muted-foreground shrink-0" />
                    <span className="text-sm truncate">{conv.title}</span>
                  </label>
                ))}
              </div>
            )}

            {/* Research Tasks section */}
            <h3 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground mb-2 flex items-center gap-1.5">
              <Search className="w-3.5 h-3.5" />
              {t("tasksSection")}
            </h3>
            {unassignedTasks.length === 0 ? (
              <p className="text-sm text-muted-foreground mb-4">
                {t("noTasks")}
              </p>
            ) : (
              <div className="space-y-1 mb-4">
                {unassignedTasks.map((task) => (
                  <label
                    key={task.id}
                    className="flex items-center gap-2 px-3 py-2 rounded-lg hover:bg-secondary/30 transition-colors cursor-pointer"
                  >
                    <input
                      type="checkbox"
                      checked={selectedTaskIds.includes(task.id)}
                      onChange={(e) => {
                        if (e.target.checked) {
                          setSelectedTaskIds((prev) => [...prev, task.id]);
                        } else {
                          setSelectedTaskIds((prev) =>
                            prev.filter((id) => id !== task.id)
                          );
                        }
                      }}
                      className="rounded"
                    />
                    <Search className="w-3.5 h-3.5 text-muted-foreground shrink-0" />
                    <span className="text-sm truncate">{task.query}</span>
                    <span className="text-[10px] text-muted-foreground uppercase">
                      {task.status}
                    </span>
                  </label>
                ))}
              </div>
            )}

            <div className="flex items-center justify-end gap-2 mt-4 pt-4 border-t border-border/30">
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setShowAddItems(false)}
                className="cursor-pointer"
              >
                {t("cancel")}
              </Button>
              <Button
                size="sm"
                disabled={totalSelectedCount === 0}
                onClick={handleAssignItems}
                className="cursor-pointer"
              >
                {t("addItems")} ({totalSelectedCount})
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
