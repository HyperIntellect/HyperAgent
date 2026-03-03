"use client";

import { useTranslations } from "next-intl";
import {
  ImageIcon,
  FileText,
  Presentation,
  Code,
  Sheet,
  File,
  Download,
  Trash2,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { usePreviewStore } from "@/lib/stores/preview-store";
import type { LibraryFile } from "@/lib/api/library";

function getFileIcon(contentType: string) {
  if (contentType.startsWith("image/")) return { Icon: ImageIcon, accent: "bg-pink-500/10 text-pink-500" };
  if (
    contentType === "application/pdf" ||
    contentType === "text/plain" ||
    contentType === "text/markdown" ||
    contentType.includes("wordprocessingml")
  )
    return { Icon: FileText, accent: "bg-blue-500/10 text-blue-500" };
  if (contentType.includes("presentationml"))
    return { Icon: Presentation, accent: "bg-orange-500/10 text-orange-500" };
  if (
    contentType === "text/x-python" ||
    contentType === "application/javascript" ||
    contentType === "application/typescript" ||
    contentType === "text/html" ||
    contentType === "text/css" ||
    contentType === "application/json"
  )
    return { Icon: Code, accent: "bg-green-500/10 text-green-500" };
  if (contentType === "text/csv" || contentType.includes("spreadsheetml"))
    return { Icon: Sheet, accent: "bg-emerald-500/10 text-emerald-500" };
  return { Icon: File, accent: "bg-muted-foreground/10 text-muted-foreground" };
}

function formatFileSize(bytes: number): string {
  if (bytes === 0) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(1024));
  return `${(bytes / Math.pow(1024, i)).toFixed(i > 0 ? 1 : 0)} ${units[i]}`;
}

function formatRelativeTime(dateStr: string): string {
  const date = new Date(dateStr);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffMin = Math.floor(diffMs / 60000);
  if (diffMin < 1) return "Just now";
  if (diffMin < 60) return `${diffMin}m ago`;
  const diffHrs = Math.floor(diffMin / 60);
  if (diffHrs < 24) return `${diffHrs}h ago`;
  const diffDays = Math.floor(diffHrs / 24);
  if (diffDays < 30) return `${diffDays}d ago`;
  return date.toLocaleDateString();
}

interface FileCardProps {
  file: LibraryFile;
  index?: number;
  onDelete?: (file: LibraryFile) => void;
}

export function FileCard({ file, index = 0, onDelete }: FileCardProps) {
  const t = useTranslations("library");
  const openPreview = usePreviewStore((s) => s.openPreview);
  const { Icon, accent } = getFileIcon(file.content_type);

  const handleClick = () => {
    openPreview({
      id: file.id,
      filename: file.filename,
      contentType: file.content_type,
      fileSize: file.file_size,
      previewUrl: file.preview_url,
      status: "uploaded",
    });
  };

  const handleDownload = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (file.preview_url) {
      const a = document.createElement("a");
      a.href = file.preview_url;
      a.download = file.filename;
      a.click();
    }
  };

  const handleDelete = (e: React.MouseEvent) => {
    e.stopPropagation();
    onDelete?.(file);
  };

  return (
    <div
      className={cn(
        "group relative border border-border/50 rounded-xl p-4",
        "hover:border-border hover:shadow-sm",
        "transition-all duration-200",
        "cursor-pointer",
        "animate-fade-in"
      )}
      style={{ animationDelay: `${index * 40}ms`, animationFillMode: "backwards" }}
      onClick={handleClick}
    >
      {/* Header: icon + filename */}
      <div className="flex items-start gap-3">
        <div className={cn("shrink-0 p-2 rounded-lg", accent)}>
          <Icon className="w-4 h-4" />
        </div>
        <div className="flex-1 min-w-0">
          <h4 className="font-semibold text-sm text-foreground truncate">
            {file.filename}
          </h4>
          <p className="text-xs text-muted-foreground mt-1">
            {formatFileSize(file.file_size)}
          </p>
        </div>
      </div>

      {/* Footer: date + download */}
      <div className="flex items-center justify-between mt-3 pt-3 border-t border-border/30">
        <span className="text-[10px] text-muted-foreground">
          {formatRelativeTime(file.created_at)}
        </span>
        <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity duration-150">
          <button
            onClick={handleDownload}
            className={cn(
              "flex items-center gap-1 px-2 py-1 rounded text-xs",
              "text-muted-foreground hover:text-foreground hover:bg-secondary",
              "transition-colors duration-150 cursor-pointer"
            )}
          >
            <Download className="w-3 h-3" />
            {t("download")}
          </button>
          <button
            onClick={handleDelete}
            className={cn(
              "flex items-center gap-1 px-2 py-1 rounded text-xs",
              "text-muted-foreground hover:text-destructive hover:bg-destructive/10",
              "transition-colors duration-150 cursor-pointer"
            )}
          >
            <Trash2 className="w-3 h-3" />
            {t("delete")}
          </button>
        </div>
      </div>
    </div>
  );
}
