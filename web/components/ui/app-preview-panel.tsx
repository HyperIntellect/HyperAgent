"use client";

import React, { useState, useCallback, useRef, useEffect } from "react";
import { useTranslations } from "next-intl";
import {
  ExternalLink,
  Maximize2,
  Minimize2,
  RefreshCw,
  Globe,
  Copy,
  Check,
  X,
} from "lucide-react";
import { cn } from "@/lib/utils";

/**
 * Validate that a URL is safe for use in iframes and window.open.
 * Allows http:, https: protocols and relative API proxy paths.
 * Rejects javascript:, data:, blob:, etc.
 */
function isSafeUrl(url: string): boolean {
  // Allow relative API proxy paths (used by BoxLite sandbox provider)
  if (url.startsWith("/api/")) return true;
  try {
    const parsed = new URL(url);
    return parsed.protocol === "http:" || parsed.protocol === "https:";
  } catch {
    return false;
  }
}

interface AppPreviewPanelProps {
  previewUrl: string;
  displayUrl?: string;
  template?: string;
  className?: string;
  onClose?: () => void;
}

/**
 * App Preview Panel - displays a live preview of a running web application
 * Shows an iframe with the app and controls for refresh, external open, and fullscreen
 */
export function AppPreviewPanel({
  previewUrl,
  displayUrl: displayUrlProp,
  template,
  className,
  onClose,
}: AppPreviewPanelProps) {
  const t = useTranslations("appPreview");
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [copied, setCopied] = useState(false);
  const [iframeKey, setIframeKey] = useState(0);
  const copyTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    return () => {
      if (copyTimerRef.current) clearTimeout(copyTimerRef.current);
    };
  }, []);

  const handleRefresh = useCallback(() => {
    setIsLoading(true);
    setIframeKey((prev) => prev + 1);
  }, []);

  const handleCopyUrl = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(previewUrl);
      setCopied(true);
      if (copyTimerRef.current) clearTimeout(copyTimerRef.current);
      copyTimerRef.current = setTimeout(() => setCopied(false), 2000);
    } catch {
      // Silently fail - navigator.clipboard is supported in all modern browsers
    }
  }, [previewUrl]);

  const handleOpenExternal = useCallback(() => {
    if (isSafeUrl(previewUrl)) {
      window.open(previewUrl, "_blank", "noopener,noreferrer");
    }
  }, [previewUrl]);

  const handleIframeLoad = useCallback(() => {
    setIsLoading(false);
  }, []);

  const safeUrl = isSafeUrl(previewUrl);

  // Use friendly display URL when available, falling back to preview URL
  const displayUrl = (displayUrlProp || previewUrl).replace(/^https?:\/\//, "");

  return (
    <div
      className={cn(
        "rounded-lg border border-border/80 bg-card overflow-hidden flex flex-col",
        isFullscreen
          ? "fixed inset-0 md:inset-4 z-50 shadow-sm pt-safe pb-safe"
          : "w-full h-[300px] md:h-[400px]",
        className
      )}
    >
      {/* Header / URL Bar */}
      <div className="flex items-center gap-2 px-3 py-2 bg-muted/50 border-b border-border/60">
        {/* Browser dots (decorative) */}
        <div className="flex items-center gap-1.5 mr-2">
          <div className="w-3 h-3 rounded-full bg-red-500/70 hover:bg-red-500 transition-colors cursor-pointer"
               onClick={onClose}
               title={t("close")} />
          <div className="w-3 h-3 rounded-full bg-yellow-500/70" />
          <div className="w-3 h-3 rounded-full bg-green-500/70" />
        </div>

        {/* URL bar */}
        <div className="flex-1 flex items-center gap-2 px-3 py-1.5 bg-background rounded-md border border-border/60">
          <Globe className="w-3.5 h-3.5 text-muted-foreground/60 flex-shrink-0" />
          <span className="text-xs font-mono text-muted-foreground truncate flex-1">
            {displayUrl}
          </span>
          <button
            onClick={handleCopyUrl}
            className="text-muted-foreground/60 hover:text-foreground transition-colors"
            title={t("copyUrl")}
          >
            {copied ? (
              <Check className="w-3.5 h-3.5 text-emerald-500" />
            ) : (
              <Copy className="w-3.5 h-3.5" />
            )}
          </button>
        </div>

        {/* Template badge */}
        {template && (
          <span className="text-[10px] font-medium text-muted-foreground bg-muted px-2 py-1 rounded">
            {template}
          </span>
        )}

        {/* Actions */}
        <div className="flex items-center gap-1">
          <button
            onClick={handleRefresh}
            className={cn(
              "p-1.5 rounded-md text-muted-foreground hover:text-foreground hover:bg-muted transition-colors",
              isLoading && "animate-spin"
            )}
            title={t("refresh")}
            disabled={isLoading}
          >
            <RefreshCw className="w-4 h-4" />
          </button>
          <button
            onClick={handleOpenExternal}
            className="p-1.5 rounded-md text-muted-foreground hover:text-foreground hover:bg-muted transition-colors"
            title={t("openExternal")}
          >
            <ExternalLink className="w-4 h-4" />
          </button>
          <button
            onClick={() => setIsFullscreen(!isFullscreen)}
            className="p-1.5 rounded-md text-muted-foreground hover:text-foreground hover:bg-muted transition-colors"
            title={isFullscreen ? t("exitFullscreen") : t("fullscreen")}
          >
            {isFullscreen ? (
              <Minimize2 className="w-4 h-4" />
            ) : (
              <Maximize2 className="w-4 h-4" />
            )}
          </button>
          {isFullscreen && (
            <button
              onClick={() => setIsFullscreen(false)}
              className="p-1.5 rounded-md text-muted-foreground hover:text-foreground hover:bg-muted transition-colors ml-1"
              title={t("close")}
            >
              <X className="w-4 h-4" />
            </button>
          )}
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 relative bg-background">
        {/* Loading overlay */}
        {isLoading && (
          <div className="absolute inset-0 flex items-center justify-center bg-background/80 z-10">
            <div className="flex flex-col items-center gap-3">
              <div className="w-8 h-8 border-2 border-primary/30 border-t-primary rounded-full animate-spin" />
              <span className="text-sm text-muted-foreground">{t("loading")}</span>
            </div>
          </div>
        )}

        {/* Iframe */}
        {safeUrl ? (
          <iframe
            key={iframeKey}
            src={previewUrl}
            className="w-full h-full border-0"
            onLoad={handleIframeLoad}
            sandbox="allow-scripts allow-forms allow-popups allow-modals"
            title={t("appPreview")}
          />
        ) : (
          <div className="flex items-center justify-center h-full text-sm text-muted-foreground">
            Invalid URL
          </div>
        )}
      </div>

      {/* Fullscreen backdrop */}
      {isFullscreen && (
        <div
          className="fixed inset-0 bg-black/40 -z-10"
          onClick={() => setIsFullscreen(false)}
        />
      )}
    </div>
  );
}

/**
 * Inline app preview that appears within message bubbles
 * Smaller, more compact version for chat context
 */
export function InlineAppPreview({
  previewUrl,
  displayUrl: displayUrlProp,
  template,
  className,
}: {
  previewUrl: string;
  displayUrl?: string;
  template?: string;
  className?: string;
}) {
  const t = useTranslations("appPreview");
  const [expanded, setExpanded] = useState(false);
  const [copied, setCopied] = useState(false);
  const inlineCopyTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    return () => {
      if (inlineCopyTimerRef.current) clearTimeout(inlineCopyTimerRef.current);
    };
  }, []);

  const handleCopyUrl = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(previewUrl);
      setCopied(true);
      if (inlineCopyTimerRef.current) clearTimeout(inlineCopyTimerRef.current);
      inlineCopyTimerRef.current = setTimeout(() => setCopied(false), 2000);
    } catch {
      setCopied(false);
    }
  }, [previewUrl]);

  const handleOpenExternal = useCallback(() => {
    if (isSafeUrl(previewUrl)) {
      window.open(previewUrl, "_blank", "noopener,noreferrer");
    }
  }, [previewUrl]);

  const safeUrl = isSafeUrl(previewUrl);

  // Use friendly display URL when available, falling back to preview URL
  const displayUrl = (displayUrlProp || previewUrl).replace(/^https?:\/\//, "");

  return (
    <div
      className={cn(
        "mt-4 rounded-lg border border-border/80 bg-card overflow-hidden",
        className
      )}
    >
      {/* Compact header - always visible */}
      <div className="flex items-center gap-3 px-4 py-3">
        {/* App icon */}
        <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-primary/20 to-primary/5 flex items-center justify-center flex-shrink-0">
          <Globe className="w-4 h-4 text-primary" />
        </div>

        {/* Info */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className="text-sm font-medium text-foreground">
              {t("appRunning")}
            </span>
            {template && (
              <span className="text-[10px] font-medium text-muted-foreground bg-muted px-1.5 py-0.5 rounded">
                {template}
              </span>
            )}
          </div>
          <span className="text-xs text-muted-foreground font-mono truncate block">
            {displayUrl}
          </span>
        </div>

        {/* Actions */}
        <div className="flex items-center gap-1">
          <button
            onClick={handleCopyUrl}
            className="p-1.5 rounded-md text-muted-foreground hover:text-foreground hover:bg-muted transition-colors"
            title={t("copyUrl")}
          >
            {copied ? (
              <Check className="w-4 h-4 text-emerald-500" />
            ) : (
              <Copy className="w-4 h-4" />
            )}
          </button>
          <button
            onClick={handleOpenExternal}
            className="p-1.5 rounded-md text-muted-foreground hover:text-foreground hover:bg-muted transition-colors"
            title={t("openExternal")}
          >
            <ExternalLink className="w-4 h-4" />
          </button>
          <button
            onClick={() => setExpanded(!expanded)}
            className="p-1.5 rounded-md text-muted-foreground hover:text-foreground hover:bg-muted transition-colors"
            title={expanded ? t("collapse") : t("expand")}
          >
            {expanded ? (
              <Minimize2 className="w-4 h-4" />
            ) : (
              <Maximize2 className="w-4 h-4" />
            )}
          </button>
        </div>
      </div>

      {/* Expandable preview */}
      {expanded && safeUrl && (
        <div className="border-t border-border/60">
          <div className="h-[300px] bg-background">
            <iframe
              src={previewUrl}
              className="w-full h-full border-0"
              sandbox="allow-scripts allow-forms allow-popups allow-modals"
              title={t("appPreview")}
            />
          </div>
        </div>
      )}
    </div>
  );
}

export default AppPreviewPanel;
