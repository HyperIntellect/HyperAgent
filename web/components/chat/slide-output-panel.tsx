"use client";

import React from "react";
import { Presentation, Download, Eye } from "lucide-react";
import { useTranslations } from "next-intl";
import { cn } from "@/lib/utils";
import { useComputerStore } from "@/lib/stores/computer-store";
import type { ExternalFileEntry } from "@/lib/stores/computer-store";

export interface SlideElement {
  type: string;
  content: string;
}

export interface SlideOutline {
  layout: string;
  title: string;
  subtitle?: string | null;
  elements: SlideElement[];
  notes?: string | null;
}

export interface SlideOutput {
  title: string;
  slide_count: number;
  download_url: string;
  storage_key?: string;
  style?: string;
  slide_outline?: SlideOutline[];
}

interface SlideOutputPanelProps {
  output: SlideOutput;
}

/**
 * Inline panel that displays slide generation results with download and open actions
 */
export function SlideOutputPanel({ output }: SlideOutputPanelProps) {
  const t = useTranslations("preview");
  const openFileInBrowser = useComputerStore((state) => state.openFileInBrowser);

  return (
    <div
      className={cn(
        "rounded-xl border border-border bg-secondary/30 p-4",
        "space-y-3"
      )}
    >
      {/* Header: icon + title */}
      <div className="flex items-center gap-3">
        <div
          className={cn(
            "w-8 h-8 rounded-lg flex items-center justify-center",
            "bg-primary/10"
          )}
        >
          <Presentation className="w-4 h-4 text-primary" />
        </div>
        <span className="text-sm font-semibold text-foreground truncate">
          {output.title}
        </span>
      </div>

      {/* Info badges */}
      <div className="flex items-center gap-2">
        <span
          className={cn(
            "inline-flex items-center px-2 py-0.5",
            "text-[11px] font-medium rounded-full",
            "bg-muted text-muted-foreground"
          )}
        >
          {t("slideCountBadge", { count: output.slide_count })}
        </span>
        {output.style && (
          <span
            className={cn(
              "inline-flex items-center px-2 py-0.5",
              "text-[11px] font-medium rounded-full",
              "bg-primary/10 text-primary/80"
            )}
          >
            {output.style}
          </span>
        )}
      </div>

      {/* Actions */}
      <div className="flex items-center gap-2">
        <a
          href={output.download_url}
          download
          className={cn(
            "inline-flex items-center gap-1.5",
            "px-3 py-1.5",
            "text-xs font-medium",
            "rounded-lg",
            "bg-primary text-primary-foreground",
            "hover:bg-primary/90 transition-colors"
          )}
        >
          <Download className="w-4 h-4" />
          {t("download")}
        </a>
        {output.slide_outline && output.slide_outline.length > 0 && (
          <button
            onClick={() => {
              const entry: ExternalFileEntry = {
                id: `slide-preview-${Date.now()}`,
                name: output.title || "Presentation.pptx",
                source: "generated-slide",
                contentType: "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                fileSize: 0,
                downloadUrl: output.download_url,
                slideOutput: output,
                timestamp: Date.now(),
              };
              openFileInBrowser(entry);
            }}
            className={cn(
              "inline-flex items-center gap-1.5",
              "px-3 py-1.5",
              "text-xs font-medium",
              "rounded-lg",
              "bg-secondary text-foreground",
              "hover:bg-secondary/80 border border-border",
              "transition-colors"
            )}
          >
            <Eye className="w-4 h-4" />
            {t("previewSlides")}
          </button>
        )}
      </div>
    </div>
  );
}
