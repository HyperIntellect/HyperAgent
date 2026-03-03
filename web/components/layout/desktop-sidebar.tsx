"use client";

import React, { useEffect, useRef, useState, useCallback } from "react";
import Image from "next/image";
import { useRouter, usePathname } from "next/navigation";
import { useSession } from "next-auth/react";
import { Sparkles, Library } from "lucide-react";
import { useTranslations } from "next-intl";
import { cn } from "@/lib/utils";
import { useChatStore } from "@/lib/stores/chat-store";
import { useTaskStore } from "@/lib/stores/task-store";
import { RecentTasks, type RecentItem } from "@/components/ui/recent-tasks";
import { CreateMenu } from "@/components/ui/create-menu";
import { MenuToggle } from "@/components/ui/menu-toggle";
import {
    useSidebarStore,
    SIDEBAR_MIN_WIDTH,
    SIDEBAR_MAX_WIDTH,
} from "@/lib/stores/sidebar-store";
import { SidebarProjects } from "@/components/layout/sidebar-projects";

interface DesktopSidebarProps {
    className?: string;
}

export function DesktopSidebar({ className }: DesktopSidebarProps) {
    const desktopSidebarOpen = useSidebarStore((state) => state.desktopSidebarOpen);
    const sidebarWidth = useSidebarStore((state) => state.sidebarWidth);
    const toggleDesktopSidebar = useSidebarStore.getState().toggleDesktopSidebar;
    const setSidebarWidth = useSidebarStore.getState().setSidebarWidth;
    const router = useRouter();
    const pathname = usePathname();
    const t = useTranslations("sidebar");
    const [isResizing, setIsResizing] = useState(false);
    const sidebarRef = useRef<HTMLElement>(null);
    const { status } = useSession();

    const conversations = useChatStore((state) => state.conversations);
    const activeConversationId = useChatStore((state) => state.activeConversationId);
    const chatHydrated = useChatStore((state) => state.hasHydrated);
    const setActiveConversation = useChatStore.getState().setActiveConversation;
    const loadConversations = useChatStore.getState().loadConversations;
    const deleteConversation = useChatStore.getState().deleteConversation;

    const tasks = useTaskStore((state) => state.tasks);
    const activeTaskId = useTaskStore((state) => state.activeTaskId);
    const taskHydrated = useTaskStore((state) => state.hasHydrated);
    const setActiveTask = useTaskStore.getState().setActiveTask;
    const deleteTask = useTaskStore.getState().deleteTask;
    const loadTasks = useTaskStore.getState().loadTasks;

    const hasHydrated = chatHydrated && taskHydrated;

    // Track if we've attempted to load conversations to prevent duplicate calls
    const loadAttempted = useRef(false);

    // Load conversations and tasks on mount (only load conversations if authenticated)
    useEffect(() => {
        // Wait for session to finish loading before making any decisions
        if (status === "loading") {
            return;
        }

        // If we've already attempted to load, don't try again
        if (loadAttempted.current) {
            return;
        }

        // If chat store is already hydrated, don't do anything
        if (chatHydrated) {
            return;
        }

        // Mark that we've attempted to load
        loadAttempted.current = true;

        // Only load from API if authenticated, otherwise just set hydrated
        if (status === "authenticated") {
            loadConversations();
        } else {
            useChatStore.setState({ hasHydrated: true });
        }
    }, [status, chatHydrated, loadConversations]);

    // Load tasks (from API if authenticated, otherwise rely on localStorage)
    useEffect(() => {
        if (status === "loading" || taskHydrated) {
            return;
        }

        if (status === "authenticated") {
            loadTasks();
        } else {
            useTaskStore.setState({ hasHydrated: true });
        }
    }, [status, taskHydrated, loadTasks]);

    // Resize handlers
    const startResizing = useCallback((e: React.MouseEvent) => {
        e.preventDefault();
        setIsResizing(true);
    }, []);

    const stopResizing = useCallback(() => {
        setIsResizing(false);
    }, []);

    const resize = useCallback(
        (e: MouseEvent) => {
            if (isResizing && sidebarRef.current) {
                const newWidth = e.clientX;
                setSidebarWidth(newWidth);
            }
        },
        [isResizing, setSidebarWidth]
    );

    // Add/remove event listeners for resize
    useEffect(() => {
        if (isResizing) {
            document.addEventListener("mousemove", resize);
            document.addEventListener("mouseup", stopResizing);
            document.body.style.cursor = "col-resize";
            document.body.style.userSelect = "none";
        }

        return () => {
            document.removeEventListener("mousemove", resize);
            document.removeEventListener("mouseup", stopResizing);
            document.body.style.cursor = "";
            document.body.style.userSelect = "";
        };
    }, [isResizing, resize, stopResizing]);

    const handleItemSelect = (item: RecentItem) => {
        if (item.type === "conversation") {
            setActiveConversation(item.data.id);
            setActiveTask(null);
            router.push("/");
        } else {
            setActiveTask(item.data.id);
            setActiveConversation(null);
            router.push(`/task/${item.data.id}`);
        }
    };

    const handleItemDelete = async (item: RecentItem) => {
        try {
            if (item.type === "conversation") {
                // If deleting current conversation, go home first
                if (item.data.id === activeConversationId) {
                    router.push("/");
                    setActiveConversation(null);
                }
                await deleteConversation(item.data.id);
            } else {
                // If deleting current task, go home first
                if (item.data.id === activeTaskId) {
                    router.push("/");
                    setActiveTask(null);
                }
                await deleteTask(item.data.id);
            }
        } catch (error) {
            console.error("Failed to delete item:", error);
        }
    };

    return (
        <aside
            ref={sidebarRef}
            className={cn(
                "hidden md:flex",
                "h-full flex-col relative",
                "border-r border-sidebar-border overflow-hidden z-20",
                "bg-sidebar",
                !isResizing && "transition-opacity duration-150",
                !desktopSidebarOpen && "w-0 opacity-0 border-r-0 pointer-events-none",
                className
            )}
            style={{
                width: desktopSidebarOpen ? sidebarWidth : 0,
            }}
            role="navigation"
            aria-label="Main navigation"
        >
            {/* Header */}
            <div className="h-14 px-4 flex items-center justify-between border-b border-sidebar-border">
                <div className="flex items-center gap-2.5">
                    <div className="w-7 h-7 flex items-center justify-center">
                        <Image
                            src="/images/logo-light.svg"
                            alt="HyperAgent"
                            width={28}
                            height={28}
                            className="dark:hidden"
                        />
                        <Image
                            src="/images/logo-dark.svg"
                            alt="HyperAgent"
                            width={28}
                            height={28}
                            className="hidden dark:block"
                        />
                    </div>
                    <span className="brand-title brand-title-sm">HyperAgent</span>
                </div>
                <MenuToggle
                    isOpen={desktopSidebarOpen}
                    onClick={toggleDesktopSidebar}
                />
            </div>

            {/* Create Button */}
            <div className="px-3 py-3">
                <CreateMenu
                    onCreate={() => {
                        setActiveConversation(null);
                        setActiveTask(null);
                        router.push("/");
                    }}
                />
            </div>

            <div className="mx-3 border-t border-sidebar-border" />

            {/* Skills Link */}
            <div className="px-3 pt-2 pb-1">
                <button
                    onClick={() => router.push("/skills")}
                    className={cn(
                        "w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-sm font-medium transition-colors",
                        pathname === "/skills"
                            ? "bg-sidebar-accent text-sidebar-accent-foreground"
                            : "text-sidebar-foreground/70 hover:text-sidebar-foreground hover:bg-sidebar-accent"
                    )}
                >
                    <Sparkles className="w-4 h-4" />
                    <span>{t("skills")}</span>
                </button>
            </div>

            {/* Library Link */}
            <div className="px-3 pb-1">
                <button
                    onClick={() => router.push("/library")}
                    className={cn(
                        "w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-sm font-medium transition-colors",
                        pathname === "/library"
                            ? "bg-sidebar-accent text-sidebar-accent-foreground"
                            : "text-sidebar-foreground/70 hover:text-sidebar-foreground hover:bg-sidebar-accent"
                    )}
                >
                    <Library className="w-4 h-4" />
                    <span>{t("library")}</span>
                </button>
            </div>

            {/* Projects */}
            <SidebarProjects variant="desktop" />

            <div className="mx-3 border-t border-sidebar-border" />

            {/* Recent Items */}
            {hasHydrated && (
                <RecentTasks
                    conversations={conversations}
                    tasks={tasks}
                    activeConversationId={activeConversationId}
                    activeTaskId={activeTaskId}
                    onSelect={handleItemSelect}
                    onDelete={handleItemDelete}
                    className="flex-1 min-h-0"
                />
            )}

            {/* Resize Handle */}
            <div
                className={cn(
                    "absolute top-0 right-0 w-1 h-full cursor-col-resize",
                    "hover:bg-muted-foreground/20 active:bg-muted-foreground/30",
                    "transition-colors duration-150",
                    isResizing && "bg-muted-foreground/30"
                )}
                onMouseDown={startResizing}
                role="separator"
                aria-orientation="vertical"
                aria-valuenow={sidebarWidth}
                aria-valuemin={SIDEBAR_MIN_WIDTH}
                aria-valuemax={SIDEBAR_MAX_WIDTH}
            />
        </aside>
    );
}
