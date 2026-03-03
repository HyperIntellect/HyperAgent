"use client";

import React, { useState, useEffect, useRef } from "react";
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
import { SidebarProjects } from "@/components/layout/sidebar-projects";

interface MobileSidebarProps {
    isOpen: boolean;
    onClose: () => void;
}

export function MobileSidebar({ isOpen, onClose }: MobileSidebarProps) {
    const router = useRouter();
    const pathname = usePathname();
    const t = useTranslations("sidebar");
    const { status } = useSession();
    const {
        conversations,
        activeConversationId,
        hasHydrated: chatHydrated,
        setActiveConversation,
        loadConversations,
        deleteConversation,
    } = useChatStore();
    const {
        tasks,
        activeTaskId,
        hasHydrated: taskHydrated,
        setActiveTask,
        deleteTask,
        loadTasks,
    } = useTaskStore();

    const hasHydrated = chatHydrated && taskHydrated;

    // Track if we've attempted to load conversations to prevent duplicate calls
    const loadAttempted = useRef(false);

    // Touch gesture handling for swipe-to-close
    const [touchStart, setTouchStart] = useState<number | null>(null);
    const [touchOffset, setTouchOffset] = useState(0);

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

    // Close sidebar on escape key
    useEffect(() => {
        if (!isOpen) return;

        const handleEscape = (e: KeyboardEvent) => {
            if (e.key === "Escape" && onClose) {
                onClose();
            }
        };

        document.addEventListener("keydown", handleEscape);
        return () => document.removeEventListener("keydown", handleEscape);
    }, [isOpen, onClose]);

    // Lock body scroll when sidebar is open on mobile
    useEffect(() => {
        if (!isOpen) return;

        const originalOverflow = document.body.style.overflow;
        document.body.style.overflow = 'hidden';
        return () => {
            document.body.style.overflow = originalOverflow;
        };
    }, [isOpen]);

    // Touch gesture handlers
    const handleTouchStart = (e: React.TouchEvent) => {
        setTouchStart(e.touches[0].clientX);
    };

    const handleTouchMove = (e: React.TouchEvent) => {
        if (touchStart === null) return;
        const currentTouch = e.touches[0].clientX;
        const offset = currentTouch - touchStart;

        // Only allow left swipe (negative offset)
        if (offset < 0) {
            setTouchOffset(offset);
        }
    };

    const handleTouchEnd = () => {
        if (touchOffset < -80 && onClose) {
            onClose(); // Close if swiped more than 80px
        }
        setTouchStart(null);
        setTouchOffset(0);
    };

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
        // Close mobile sidebar after selection
        onClose();
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
        <>
            {/* Mobile backdrop */}
            {isOpen && (
                <div
                    className={cn(
                        "fixed inset-0 bg-black/40 z-40 md:hidden",
                        "transition-opacity duration-200",
                        "touch-none",
                        isOpen ? "opacity-100" : "opacity-0"
                    )}
                    onClick={onClose}
                    aria-hidden="true"
                />
            )}

            {/* Sidebar */}
            <aside
                onTouchStart={handleTouchStart}
                onTouchMove={handleTouchMove}
                onTouchEnd={handleTouchEnd}
                className={cn(
                    "md:hidden",
                    "h-full flex flex-col",
                    "bg-card border-r border-border",
                    "fixed inset-y-0 left-0 z-50 w-[300px]",
                    "pt-safe pb-safe",
                    "will-change-transform",
                    "motion-reduce:transition-none",
                    touchOffset === 0 && "transition-transform duration-200",
                    isOpen ? "translate-x-0" : "-translate-x-full",
                    !isOpen && "pointer-events-none"
                )}
                style={{
                    transform: isOpen && touchOffset < 0
                        ? `translateX(${touchOffset}px)`
                        : undefined,
                }}
                role="navigation"
                aria-label="Main navigation"
            >
                {/* Header */}
                <div className="h-14 px-4 flex items-center justify-between border-b border-border">
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

                    {/* Mobile close button */}
                    <MenuToggle
                        isOpen={true}
                        onClick={onClose}
                    />
                </div>

                {/* Create Button */}
                <div className="px-3 py-3">
                    <CreateMenu
                        onCreate={() => {
                            setActiveConversation(null);
                            setActiveTask(null);
                            router.push("/");
                            onClose();
                        }}
                    />
                </div>

                <div className="mx-3 border-t border-border/50" />

                {/* Skills Link */}
                <div className="px-3 pt-2 pb-1">
                    <button
                        onClick={() => {
                            router.push("/skills");
                            onClose();
                        }}
                        className={cn(
                            "w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-sm font-medium transition-colors",
                            pathname === "/skills"
                                ? "bg-secondary text-foreground"
                                : "text-muted-foreground hover:text-foreground hover:bg-secondary/50"
                        )}
                    >
                        <Sparkles className="w-4 h-4" />
                        <span>{t("skills")}</span>
                    </button>
                </div>

                {/* Library Link */}
                <div className="px-3 pb-1">
                    <button
                        onClick={() => {
                            router.push("/library");
                            onClose();
                        }}
                        className={cn(
                            "w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-sm font-medium transition-colors",
                            pathname === "/library"
                                ? "bg-secondary text-foreground"
                                : "text-muted-foreground hover:text-foreground hover:bg-secondary/50"
                        )}
                    >
                        <Library className="w-4 h-4" />
                        <span>{t("library")}</span>
                    </button>
                </div>

                {/* Projects */}
                <SidebarProjects variant="mobile" onNavigate={onClose} />

                <div className="mx-3 border-t border-border/50" />

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
            </aside>
        </>
    );
}
