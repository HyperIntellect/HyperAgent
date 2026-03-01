"use client";

import React, { useState, useCallback, useRef } from "react";
import {
    Monitor,
    Globe,
    ArrowLeft,
    ArrowRight,
    RotateCw,
    Maximize2,
    Minimize2,
    ExternalLink,
    Lock,
    MousePointer2,
    Keyboard,
    ArrowUpDown,
} from "lucide-react";
import { useTranslations } from "next-intl";
import { cn } from "@/lib/utils";
import { useComputerStore } from "@/lib/stores/computer-store";
import type { ComputerStreamInfo } from "@/lib/stores/agent-progress-store";
import { Button } from "@/components/ui/button";
import { ComputerEmptyState } from "./computer-empty-state";

interface ComputerBrowserViewProps {
    stream: ComputerStreamInfo | null;
    className?: string;
}

function getActionIcon(description: string) {
    const lower = description.toLowerCase();
    if (lower.includes("click") || lower.includes("tap") || lower.includes("press")) {
        return <MousePointer2 className="w-3 h-3 flex-shrink-0" />;
    }
    if (lower.includes("type") || lower.includes("input") || lower.includes("enter") || lower.includes("fill")) {
        return <Keyboard className="w-3 h-3 flex-shrink-0" />;
    }
    if (lower.includes("scroll")) {
        return <ArrowUpDown className="w-3 h-3 flex-shrink-0" />;
    }
    return null;
}

export function ComputerBrowserView({
    stream,
    className,
}: ComputerBrowserViewProps) {
    const t = useTranslations("computer");

    // Browser state from store - access underlying state directly for proper Zustand tracking
    // (calling getter methods like getBrowserUrl() inside selectors doesn't subscribe to state changes)
    const browserUrl = useComputerStore((state) => {
        const id = state.activeConversationId;
        return id ? state.conversationStates[id]?.browserUrl ?? null : null;
    });
    const browserAction = useComputerStore((state) => {
        const id = state.activeConversationId;
        return id ? state.conversationStates[id]?.browserAction ?? null : null;
    });
    const browserIsNavigating = useComputerStore((state) => {
        const id = state.activeConversationId;
        return id ? state.conversationStates[id]?.browserIsNavigating ?? false : false;
    });
    const isLive = useComputerStore((state) => {
        const id = state.activeConversationId;
        return id ? state.conversationStates[id]?.isLive ?? true : true;
    });

    const [isFullscreen, setIsFullscreen] = useState(false);
    const iframeRef = useRef<HTMLIFrameElement>(null);

    // Show overlay when action is running
    const visibleAction = browserAction?.status === "running" ? browserAction : null;

    // Send auth key to iframe via postMessage after load.
    // We use "*" as targetOrigin because the iframe sandbox omits allow-same-origin,
    // giving the iframe an opaque (null) origin that cannot be matched by a specific origin string.
    const handleIframeLoad = useCallback(
        (ref: React.RefObject<HTMLIFrameElement | null>) => {
            if (stream?.authKey && ref.current?.contentWindow) {
                ref.current.contentWindow.postMessage(
                    { type: "auth", authKey: stream.authKey },
                    "*"
                );
            }
        },
        [stream]
    );

    const handleScreenshot = useCallback(() => {
        if (stream?.streamUrl) {
            window.open(stream.streamUrl, '_blank', 'noopener,noreferrer');
        }
    }, [stream]);

    // Extract display URL (truncated for the address bar)
    const displayUrl = browserUrl || "";
    const isHttps = displayUrl.startsWith("https://");

    if (!stream) {
        return (
            <ComputerEmptyState
                icon={Monitor}
                title={t("noBrowserStream")}
                subtitle={t("browserStreamWillAppear")}
                className={cn("bg-secondary/30", className)}
            />
        );
    }

    // Extract click coordinates from action if available (may be present in extended action data)
    const actionData = visibleAction as Record<string, unknown> | null;
    const clickCoords = actionData && typeof actionData.x === "number" && typeof actionData.y === "number"
        ? { x: actionData.x as number, y: actionData.y as number }
        : undefined;

    // Action overlay toast
    const actionOverlay = visibleAction ? (
        <div className="absolute bottom-3 left-1/2 -translate-x-1/2 z-10 max-w-[80%]">
            <div className={cn(
                "flex items-center gap-2 px-3 py-1.5 rounded-lg",
                "bg-background/95 border border-border",
                "animate-in fade-in slide-in-from-bottom-2 duration-200"
            )}>
                {getActionIcon(visibleAction.description || "")}
                <span className="w-1.5 h-1.5 rounded-full bg-primary animate-pulse flex-shrink-0" />
                <span className="text-xs text-foreground truncate">
                    {visibleAction.description}
                </span>
            </div>
        </div>
    ) : null;

    // Click position indicator
    const clickIndicator = visibleAction && clickCoords ? (
        <div
            className="absolute z-10 bg-primary/30 rounded-full w-8 h-8 pointer-events-none animate-ping"
            style={{
                left: `${clickCoords.x}%`,
                top: `${clickCoords.y}%`,
                transform: "translate(-50%, -50%)",
            }}
        />
    ) : null;

    // Use a single container with CSS-based fullscreen toggle to avoid
    // destroying/recreating the iframe (which loses iframe state and causes reloads).
    return (
        <div className={cn(
            "flex flex-col overflow-hidden",
            isFullscreen
                ? "fixed inset-0 z-[100] bg-background"
                : "flex-1",
            className
        )}>
            {/* Browser chrome - adapts styling based on fullscreen state */}
            <div className={cn(
                "flex items-center gap-1.5 px-2 py-1.5 border-b",
                isFullscreen
                    ? "px-3 bg-card/95 backdrop-blur-sm border-border"
                    : "bg-secondary/50 border-border/30"
            )}>
                {/* Navigation buttons */}
                <Button
                    variant="ghost"
                    size="icon"
                    className={cn(
                        "h-8 w-8 cursor-default opacity-50",
                        "focus-visible:ring-2 focus-visible:ring-primary focus-visible:outline-none",
                        isFullscreen ? "text-muted-foreground/60" : "text-muted-foreground/50"
                    )}
                    disabled
                    title={t("browserBack")}
                >
                    <ArrowLeft className="w-3.5 h-3.5" />
                </Button>
                <Button
                    variant="ghost"
                    size="icon"
                    className={cn(
                        "h-8 w-8 cursor-default opacity-50",
                        "focus-visible:ring-2 focus-visible:ring-primary focus-visible:outline-none",
                        isFullscreen ? "text-muted-foreground/60" : "text-muted-foreground/50"
                    )}
                    disabled
                    title={t("browserForward")}
                >
                    <ArrowRight className="w-3.5 h-3.5" />
                </Button>
                <Button
                    variant="ghost"
                    size="icon"
                    className={cn(
                        "h-8 w-8 cursor-default opacity-50",
                        "focus-visible:ring-2 focus-visible:ring-primary focus-visible:outline-none",
                        isFullscreen ? "text-muted-foreground/60" : "text-muted-foreground/50",
                        browserIsNavigating && "animate-spin"
                    )}
                    disabled
                    title={t("browserRefresh")}
                >
                    <RotateCw className="w-3.5 h-3.5" />
                </Button>

                {/* URL bar */}
                <div className={cn(
                    "flex-1 flex items-center gap-1.5 h-7 px-2.5 rounded-lg min-w-0 transition-colors",
                    isFullscreen
                        ? "bg-accent/50 border border-border focus-within:border-primary/50"
                        : "bg-background/80 border border-border/40 focus-within:border-primary/50"
                )}>
                    {displayUrl ? (
                        <>
                            {isHttps ? (
                                <Lock className={cn("w-3 h-3 flex-shrink-0", "text-muted-foreground/60")} />
                            ) : (
                                <Globe className={cn("w-3 h-3 flex-shrink-0", "text-muted-foreground/60")} />
                            )}
                            <span className={cn("text-xs truncate", isFullscreen ? "text-muted-foreground" : "text-muted-foreground select-all")}>
                                {displayUrl}
                            </span>
                        </>
                    ) : (
                        <>
                            <Globe className="w-3 h-3 flex-shrink-0 text-muted-foreground/40" />
                            <span className="text-xs truncate text-muted-foreground/40">
                                {t("browserUrlPlaceholder")}
                            </span>
                        </>
                    )}
                </div>

                {/* Right-side controls */}
                {/* Live indicator - uses primary color */}
                {isLive && stream && (
                    <div className={cn(
                        "flex items-center gap-0.5 px-1.5 py-0.5 rounded-full",
                        "bg-primary/10 border border-primary/20"
                    )}>
                        <span className="w-1.5 h-1.5 rounded-full bg-primary animate-pulse" />
                        <span className="text-xs font-medium text-primary">
                            {t("browserLive")}
                        </span>
                    </div>
                )}

                <Button
                    variant="ghost"
                    size="icon"
                    className={cn(
                        "h-8 w-8",
                        "focus-visible:ring-2 focus-visible:ring-primary focus-visible:outline-none",
                        isFullscreen && "text-foreground hover:bg-accent"
                    )}
                    onClick={handleScreenshot}
                    title={t("openInNewTab")}
                    aria-label={t("openInNewTab")}
                >
                    <ExternalLink className="w-3.5 h-3.5" />
                </Button>
                <Button
                    variant="ghost"
                    size="icon"
                    className={cn(
                        "h-8 w-8",
                        "focus-visible:ring-2 focus-visible:ring-primary focus-visible:outline-none",
                        isFullscreen && "text-foreground hover:bg-accent"
                    )}
                    onClick={() => setIsFullscreen(!isFullscreen)}
                    title={isFullscreen ? t("browserExitFullscreen") : t("browserFullscreen")}
                    aria-label={isFullscreen ? t("browserExitFullscreen") : t("browserFullscreen")}
                >
                    {isFullscreen ? (
                        <Minimize2 className="w-3.5 h-3.5" />
                    ) : (
                        <Maximize2 className="w-3.5 h-3.5" />
                    )}
                </Button>
            </div>

            {/* Loading progress bar */}
            {browserIsNavigating && (
                <div className={cn("h-0.5 w-full overflow-hidden", isFullscreen ? "bg-accent/50" : "bg-secondary/30")}>
                    <div className="h-full bg-primary/70 animate-progress-indeterminate" />
                </div>
            )}

            {/* Single iframe area - preserved across fullscreen toggle */}
            <div className="flex-1 relative bg-muted">
                <iframe
                    ref={iframeRef}
                    src={stream.streamUrl}
                    className="w-full h-full border-0"
                    style={{ display: "block" }}
                    sandbox="allow-scripts allow-popups allow-pointer-lock"
                    allow="autoplay; fullscreen"
                    referrerPolicy="no-referrer"
                    onLoad={() => handleIframeLoad(iframeRef)}
                    title={t("browserIframeTitle")}
                />
                {clickIndicator}
                {actionOverlay}
            </div>
        </div>
    );
}
