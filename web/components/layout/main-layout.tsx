"use client";

import React, { useState, useEffect } from "react";
import { DesktopSidebar } from "./desktop-sidebar";
import { MobileSidebar } from "./mobile-sidebar";
import { MenuToggle } from "@/components/ui/menu-toggle";
import { UserProfileMenu } from "@/components/auth/user-profile-menu";
import { useSidebarStore } from "@/lib/stores/sidebar-store";
import { useComputerStore } from "@/lib/stores/computer-store";
import { cn } from "@/lib/utils";

import { FilePreviewSidebar } from "@/components/artifacts/artifacts-preview-panel";
import { ArtifactsToggleButton } from "@/components/artifacts/artifacts-toggle-button";
import { VirtualComputerPanel, ComputerToggleButton } from "@/components/computer";

interface MainLayoutProps {
    children: React.ReactNode;
}

export function MainLayout({ children }: MainLayoutProps) {
    const [mobileSidebarOpen, setMobileSidebarOpen] = useState(false);
    const [isDesktop, setIsDesktop] = useState(false);
    const { desktopSidebarOpen, toggleDesktopSidebar } = useSidebarStore();
    const { isOpen: computerPanelOpen, panelWidth } = useComputerStore();

    // Check if we're on desktop (lg breakpoint = 1024px)
    useEffect(() => {
        const mq = window.matchMedia("(min-width: 1024px)");
        setIsDesktop(mq.matches);
        const handler = (e: MediaQueryListEvent) => setIsDesktop(e.matches);
        mq.addEventListener("change", handler);
        return () => mq.removeEventListener("change", handler);
    }, []);

    // Calculate the right panel width for content adjustment (only on desktop)
    const rightPanelPadding = isDesktop && computerPanelOpen ? panelWidth : 0;

    return (
        <div className="flex h-screen overflow-hidden bg-background">
            {/* Desktop Sidebar - Always visible on desktop */}
            <DesktopSidebar />

            {/* Mobile Sidebar - Overlay on mobile */}
            <MobileSidebar
                isOpen={mobileSidebarOpen}
                onClose={() => setMobileSidebarOpen(false)}
            />

            <main
                className="flex-1 flex flex-col overflow-hidden relative transition-colors duration-150"
                style={{
                    paddingRight: rightPanelPadding || undefined,
                }}
            >
                {/* Mobile header with menu toggle and user profile */}
                <div className="md:hidden h-12 px-3 flex items-center justify-between border-b border-border bg-background sticky top-0 z-30">
                    <div className="flex items-center gap-2">
                        <MenuToggle
                            isOpen={mobileSidebarOpen}
                            onClick={() => setMobileSidebarOpen(!mobileSidebarOpen)}
                        />
                        <span className="text-sm font-semibold text-foreground">HyperAgent</span>
                    </div>
                    <div className="flex items-center gap-1">
                        <ArtifactsToggleButton />
                        <ComputerToggleButton />
                        <UserProfileMenu />
                    </div>
                </div>

                {/* Desktop floating toggle - animated visibility when sidebar is closed */}
                <div
                    className={cn(
                        "hidden md:flex absolute left-4 top-4 z-40",
                        "glass-card rounded-lg",
                        "transition-opacity duration-150",
                        desktopSidebarOpen
                            ? "opacity-0 pointer-events-none"
                            : "opacity-100"
                    )}
                >
                    <MenuToggle
                        isOpen={false}
                        onClick={toggleDesktopSidebar}
                    />
                </div>

                {/* Desktop user profile and computer toggle - top right corner */}
                <div className="hidden md:flex items-center gap-2 absolute right-4 top-4 z-40">
                    <ArtifactsToggleButton />
                    <ComputerToggleButton />
                    <UserProfileMenu />
                </div>

                <div className="flex-1 flex flex-col overflow-hidden w-full">
                    {children}
                </div>

                {/* File Preview Sidebar */}
                <FilePreviewSidebar />

                {/* Virtual Computer Panel - Right side panel for terminal, browser, and file views */}
                <VirtualComputerPanel />
            </main>
        </div>
    );
}
