"use client";

import { useMemo, useState, useRef, useEffect } from "react";
import { useTranslations } from "next-intl";

// Configuration for virtual/lazy rendering
const INITIAL_RENDER_COUNT = 30; // Items to render initially
const BATCH_SIZE = 20; // Additional items to render on scroll
import {
  MessageCircle,
  Search,
  BookOpenText,
  Trash2,
  Loader2,
  CheckCircle2,
  AlertCircle,
  BarChart3,
  ListFilter,
  ChevronDown,
  Check,
  ImageIcon,
  AppWindow,
  Presentation,
} from "lucide-react";
import { cn } from "@/lib/utils";
import type { Conversation, ConversationType } from "@/lib/types";
import type { ResearchTask } from "@/lib/stores/task-store";

// Unified item type
export type RecentItem =
  | { type: "conversation"; data: Conversation }
  | { type: "task"; data: ResearchTask };

// Filter types
type FilterType = "all" | ConversationType;

// Per-type icon and color for sidebar items
const CONVERSATION_TYPE_CONFIG: Record<ConversationType, {
  icon: React.ReactNode;
  filterIcon: React.ReactNode;
  color: string;
}> = {
  task: {
    icon: <MessageCircle className="w-4 h-4" />,
    filterIcon: <MessageCircle className="w-3.5 h-3.5" />,
    color: "text-muted-foreground",
  },
  research: {
    icon: <BookOpenText className="w-4 h-4" />,
    filterIcon: <BookOpenText className="w-3.5 h-3.5" />,
    color: "text-indigo-500 dark:text-indigo-400",
  },
  data: {
    icon: <BarChart3 className="w-4 h-4" />,
    filterIcon: <BarChart3 className="w-3.5 h-3.5" />,
    color: "text-amber-500 dark:text-amber-400",
  },
  app: {
    icon: <AppWindow className="w-4 h-4" />,
    filterIcon: <AppWindow className="w-3.5 h-3.5" />,
    color: "text-rose-500 dark:text-rose-400",
  },
  image: {
    icon: <ImageIcon className="w-4 h-4" />,
    filterIcon: <ImageIcon className="w-3.5 h-3.5" />,
    color: "text-purple-500 dark:text-purple-400",
  },
  slide: {
    icon: <Presentation className="w-4 h-4" />,
    filterIcon: <Presentation className="w-3.5 h-3.5" />,
    color: "text-teal-500 dark:text-teal-400",
  },
};

// Legacy task items (from old research worker) use Search icon
const TASK_ITEM_CONFIG = {
  icon: <Search className="w-4 h-4" />,
  color: "text-blue-500 dark:text-blue-400",
};

const FILTER_ICONS: Record<FilterType, React.ReactNode> = {
  all: <ListFilter className="w-3.5 h-3.5" />,
  task: CONVERSATION_TYPE_CONFIG.task.filterIcon,
  research: CONVERSATION_TYPE_CONFIG.research.filterIcon,
  data: CONVERSATION_TYPE_CONFIG.data.filterIcon,
  app: CONVERSATION_TYPE_CONFIG.app.filterIcon,
  image: CONVERSATION_TYPE_CONFIG.image.filterIcon,
  slide: CONVERSATION_TYPE_CONFIG.slide.filterIcon,
};

// Group items by time period
function groupByTimePeriod(items: RecentItem[]): Record<string, RecentItem[]> {
  const now = new Date();
  const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  const yesterday = new Date(today.getTime() - 86400000);
  const lastWeek = new Date(today.getTime() - 7 * 86400000);

  const groups: Record<string, RecentItem[]> = {
    today: [],
    yesterday: [],
    thisWeek: [],
    earlier: [],
  };

  items.forEach((item) => {
    const itemDate = new Date(item.data.updatedAt);
    if (itemDate >= today) {
      groups.today.push(item);
    } else if (itemDate >= yesterday) {
      groups.yesterday.push(item);
    } else if (itemDate >= lastWeek) {
      groups.thisWeek.push(item);
    } else {
      groups.earlier.push(item);
    }
  });

  return groups;
}

interface RecentTasksProps {
  conversations: Conversation[];
  tasks: ResearchTask[];
  activeConversationId: string | null;
  activeTaskId: string | null;
  onSelect: (item: RecentItem) => void;
  onDelete: (item: RecentItem) => void;
  className?: string;
}

export function RecentTasks({
  conversations,
  tasks,
  activeConversationId,
  activeTaskId,
  onSelect,
  onDelete,
  className,
}: RecentTasksProps) {
  const t = useTranslations("sidebar");
  const tCommon = useTranslations("common");
  const [activeFilter, setActiveFilter] = useState<FilterType>("all");
  const [visibleCount, setVisibleCount] = useState(INITIAL_RENDER_COUNT);
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const loadMoreRef = useRef<HTMLDivElement>(null);

  // Combine and sort items
  const allItems = useMemo(() => {
    const conversationItems: RecentItem[] = conversations.map((conv) => ({
      type: "conversation" as const,
      data: conv,
    }));
    const taskItems: RecentItem[] = tasks.map((task) => ({
      type: "task" as const,
      data: task,
    }));
    return [...conversationItems, ...taskItems].sort(
      (a, b) =>
        new Date(b.data.updatedAt).getTime() -
        new Date(a.data.updatedAt).getTime()
    );
  }, [conversations, tasks]);

  // Filter items based on active filter
  const filteredItems = useMemo(() => {
    if (activeFilter === "all") return allItems;
    if (activeFilter === "task") {
      return allItems.filter((item) => item.type === "task");
    }
    // Filter by conversation type
    return allItems.filter(
      (item) =>
        item.type === "conversation" && item.data.type === activeFilter
    );
  }, [allItems, activeFilter]);

  // Reset visible count when filter changes
  useEffect(() => {
    setVisibleCount(INITIAL_RENDER_COUNT);
  }, [activeFilter]);

  // Lazy render: only render visible items for better performance with 100+ items
  const visibleItems = useMemo(
    () => filteredItems.slice(0, visibleCount),
    [filteredItems, visibleCount]
  );

  const hasMore = visibleCount < filteredItems.length;

  const groupedItems = useMemo(
    () => groupByTimePeriod(visibleItems),
    [visibleItems]
  );

  // Intersection observer for infinite scroll
  useEffect(() => {
    const loadMore = loadMoreRef.current;
    if (!loadMore || !hasMore) return;

    const observer = new IntersectionObserver(
      (entries) => {
        if (entries[0].isIntersecting) {
          setVisibleCount((prev) => Math.min(prev + BATCH_SIZE, filteredItems.length));
        }
      },
      { rootMargin: "100px" }
    );

    observer.observe(loadMore);
    return () => observer.disconnect();
  }, [hasMore, filteredItems.length]);

  // Get available filters based on existing items
  const availableFilters = useMemo(() => {
    const filters = new Set<FilterType>(["all"]);
    allItems.forEach((item) => {
      if (item.type === "task") {
        filters.add("task");
      } else {
        filters.add(item.data.type);
      }
    });
    return Array.from(filters);
  }, [allItems]);

  const periodLabels: Record<string, string> = {
    today: t("today"),
    yesterday: t("yesterday"),
    thisWeek: t("thisWeek"),
    earlier: t("earlier"),
  };

  const filterLabels: Record<FilterType, string> = {
    all: t("all") || "All",
    task: t("task") || "Task",
    research: t("research"),
    data: t("data"),
    app: t("app") || "App",
    image: t("image") || "Image",
    slide: t("slide") || "Slides",
  };

  const [isFilterOpen, setIsFilterOpen] = useState(false);
  const filterRef = useRef<HTMLDivElement>(null);

  // Close filter menu when clicking outside
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (filterRef.current && !filterRef.current.contains(event.target as Node)) {
        setIsFilterOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  return (
    <div className={cn("flex flex-col", className)}>
      {/* Header with Filter Dropdown */}
      {allItems.length > 0 && (
        <div className="px-3 py-2 flex items-center justify-between">
          <span className="text-xs font-medium text-muted-foreground">
            {t("recentTasks")}
          </span>

          {/* Filter Dropdown */}
          {availableFilters.length > 1 && (
            <div ref={filterRef} className="relative">
              <button
                onClick={() => setIsFilterOpen(!isFilterOpen)}
                className={cn(
                  "flex items-center gap-1.5 px-2 py-1 rounded-md text-xs font-medium transition-colors",
                  isFilterOpen || activeFilter !== "all"
                    ? "bg-accent text-accent-foreground"
                    : "text-muted-foreground hover:text-foreground hover:bg-accent"
                )}
              >
                {FILTER_ICONS[activeFilter]}
                <span>{filterLabels[activeFilter]}</span>
                <ChevronDown className={cn(
                  "w-3 h-3",
                  isFilterOpen && "rotate-180"
                )} />
              </button>

              {/* Dropdown Menu */}
              {isFilterOpen && (
                <div className="absolute right-0 top-full mt-1 z-50 min-w-[140px] bg-card border border-border rounded-lg overflow-hidden">
                  {availableFilters.map((filter) => (
                    <button
                      key={filter}
                      onClick={() => {
                        setActiveFilter(filter);
                        setIsFilterOpen(false);
                      }}
                      className={cn(
                        "w-full flex items-center gap-2 px-3 py-2.5 text-sm transition-colors text-left",
                        activeFilter === filter
                          ? "bg-accent text-accent-foreground"
                          : "text-muted-foreground hover:text-foreground hover:bg-accent"
                      )}
                    >
                      {FILTER_ICONS[filter]}
                      <span className="flex-1">{filterLabels[filter]}</span>
                      {activeFilter === filter && (
                        <Check className="w-3.5 h-3.5" />
                      )}
                    </button>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* List */}
      <div ref={scrollContainerRef} className="flex-1 overflow-y-auto px-3 pb-3">
        {allItems.length === 0 ? (
          <EmptyState />
        ) : filteredItems.length === 0 ? (
          <div className="py-8 text-center">
            <p className="text-sm text-muted-foreground">{t("noResults") || "No items found"}</p>
          </div>
        ) : (
          <div className="space-y-4">
            {Object.entries(groupedItems).map(
              ([period, items]) =>
                items.length > 0 && (
                  <TimeGroup
                    key={period}
                    label={periodLabels[period]}
                    items={items}
                    activeConversationId={activeConversationId}
                    activeTaskId={activeTaskId}
                    onSelect={onSelect}
                    onDelete={onDelete}
                  />
                )
            )}
            {/* Infinite scroll trigger */}
            {hasMore && (
              <div ref={loadMoreRef} className="py-2 text-center">
                <span className="text-xs text-muted-foreground">
                  {tCommon("loadingMore")}
                </span>
              </div>
            )}
          </div>
        )}
      </div>

    </div>
  );
}

// Empty state component
function EmptyState() {
  const t = useTranslations("sidebar");
  return (
    <div className="flex flex-col items-center justify-center py-12 px-4">
      <div className="w-10 h-10 rounded-sm bg-secondary flex items-center justify-center mb-3">
        <MessageCircle className="w-5 h-5 text-muted-foreground" />
      </div>
      <p className="text-sm text-muted-foreground text-center mb-1">
        {t("noConversations")}
      </p>
      <p className="text-xs text-muted-foreground/60 text-center">
        {t("startChatOrResearch")}
      </p>
    </div>
  );
}

// Time group component
interface TimeGroupProps {
  label: string;
  items: RecentItem[];
  activeConversationId: string | null;
  activeTaskId: string | null;
  onSelect: (item: RecentItem) => void;
  onDelete: (item: RecentItem) => void;
}

function TimeGroup({
  label,
  items,
  activeConversationId,
  activeTaskId,
  onSelect,
  onDelete,
}: TimeGroupProps) {
  return (
    <div>
      {/* Time label */}
      <div className="px-2 py-1.5">
        <span className="text-xs font-medium text-muted-foreground">
          {label}
        </span>
      </div>

      {/* Items */}
      <div className="space-y-0.5">
        {items.map((item) => (
          <RecentItemRow
            key={item.data.id}
            item={item}
            isActive={
              item.type === "conversation"
                ? item.data.id === activeConversationId
                : item.data.id === activeTaskId
            }
            onClick={() => onSelect(item)}
            onDelete={() => onDelete(item)}
          />
        ))}
      </div>
    </div>
  );
}

// Individual item row component
interface RecentItemRowProps {
  item: RecentItem;
  isActive: boolean;
  onClick: () => void;
  onDelete: () => void;
}

function RecentItemRow({
  item,
  isActive,
  onClick,
  onDelete,
}: RecentItemRowProps) {
  const tCommon = useTranslations("common");
  const isTask = item.type === "task";
  const conversationType = item.type === "conversation" ? item.data.type : null;
  const title = isTask ? item.data.query : item.data.title;
  const taskStatus = isTask ? item.data.status : null;

  // Get type icon and color
  const config = isTask
    ? TASK_ITEM_CONFIG
    : CONVERSATION_TYPE_CONFIG[conversationType || "task"];
  const typeIcon = config.icon;
  const iconColor = config.color;

  // Get status indicator for tasks
  const getStatusIndicator = () => {
    if (!isTask) return null;
    if (taskStatus === "running") {
      return <Loader2 className="w-3.5 h-3.5 animate-spin text-muted-foreground" />;
    }
    if (taskStatus === "completed") {
      return <CheckCircle2 className="w-3.5 h-3.5 text-success" />;
    }
    if (taskStatus === "failed") {
      return <AlertCircle className="w-3.5 h-3.5 text-destructive" />;
    }
    return null;
  };

  return (
    <div
      className={cn(
        "group relative flex items-center gap-3 px-2.5 py-2.5 rounded-lg cursor-pointer transition-colors",
        isActive
          ? "bg-accent text-accent-foreground"
          : "hover:bg-accent"
      )}
      onClick={onClick}
    >
      {/* Type Icon Container */}
      <div className={cn(
        "flex-shrink-0 w-8 h-8 rounded-lg flex items-center justify-center transition-colors",
        isActive ? "bg-secondary" : "bg-muted group-hover:bg-secondary"
      )}>
        <span className={cn(
          "transition-colors",
          isActive ? "text-foreground" : iconColor
        )}>
          {typeIcon}
        </span>
      </div>

      {/* Title */}
      <span
        className={cn(
          "flex-1 text-sm truncate",
          isActive ? "text-foreground font-medium" : "text-foreground/80 group-hover:text-foreground"
        )}
      >
        {title.slice(0, 40) + (title.length > 40 ? "..." : "")}
      </span>

      {/* Status indicator for tasks */}
      {getStatusIndicator()}

      {/* Delete button - visible on mobile, hover-reveal on desktop */}
      <button
        className={cn(
          "flex-shrink-0 p-1.5 rounded-lg transition-colors",
          "opacity-60 md:opacity-0 group-hover:opacity-100",
          "text-muted-foreground hover:text-destructive hover:bg-destructive/10",
          "min-h-[36px] min-w-[36px] flex items-center justify-center"
        )}
        onClick={(e) => {
          e.stopPropagation();
          onDelete();
        }}
        aria-label={tCommon("delete")}
      >
        <Trash2 className="w-4 h-4" />
      </button>
    </div>
  );
}
