"use client";

import { useEffect, useState, useDeferredValue } from "react";
import { useTranslations } from "next-intl";
import { listLibraryFiles, deleteLibraryFile, type LibraryFile } from "@/lib/api/library";
import { FileCard } from "./file-card";
import { Input } from "@/components/ui/input";
import {
  Search,
  Loader2,
  AlertCircle,
  ImageIcon,
  FileText,
  Presentation,
  Code,
  Sheet,
  LayoutGrid,
  FolderOpen,
} from "lucide-react";
import { cn } from "@/lib/utils";

const CATEGORIES = ["all", "images", "documents", "slides", "code", "data"] as const;
type CategoryFilter = (typeof CATEGORIES)[number];

const CATEGORY_ICONS: Record<CategoryFilter, React.ReactNode> = {
  all: <LayoutGrid className="w-3.5 h-3.5" />,
  images: <ImageIcon className="w-3.5 h-3.5" />,
  documents: <FileText className="w-3.5 h-3.5" />,
  slides: <Presentation className="w-3.5 h-3.5" />,
  code: <Code className="w-3.5 h-3.5" />,
  data: <Sheet className="w-3.5 h-3.5" />,
};

// Map categories to content type filters
const CATEGORY_CONTENT_TYPES: Record<string, string[]> = {
  images: ["image/"],
  documents: [
    "application/pdf",
    "text/plain",
    "text/markdown",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
  ],
  slides: ["application/vnd.openxmlformats-officedocument.presentationml.presentation"],
  code: [
    "text/x-python",
    "application/javascript",
    "application/typescript",
    "text/html",
    "text/css",
    "application/json",
  ],
  data: [
    "text/csv",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
  ],
};

function matchesCategory(contentType: string, category: CategoryFilter): boolean {
  if (category === "all") return true;
  const types = CATEGORY_CONTENT_TYPES[category];
  if (!types) return false;
  return types.some((t) =>
    t.endsWith("/") ? contentType.startsWith(t) : contentType === t
  );
}

export function LibraryBrowser() {
  const t = useTranslations("library");
  const [files, setFiles] = useState<LibraryFile[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [categoryFilter, setCategoryFilter] = useState<CategoryFilter>("all");

  const deferredSearch = useDeferredValue(searchQuery);

  useEffect(() => {
    loadFiles();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function loadFiles() {
    try {
      setLoading(true);
      setError(null);
      const data = await listLibraryFiles({ limit: 200 });
      setFiles(data.files);
    } catch (err) {
      console.error("Failed to load files:", err);
      setError(t("loadError"));
    } finally {
      setLoading(false);
    }
  }

  async function handleDelete(file: LibraryFile) {
    try {
      await deleteLibraryFile(file.id);
      setFiles((prev) => prev.filter((f) => f.id !== file.id));
    } catch (err) {
      console.error("Failed to delete file:", err);
    }
  }

  // Filter files client-side
  const filteredFiles = files.filter((file) => {
    const matchesSearch =
      deferredSearch === "" ||
      file.filename.toLowerCase().includes(deferredSearch.toLowerCase());

    const matchesCat = matchesCategory(file.content_type, categoryFilter);

    return matchesSearch && matchesCat;
  });

  // Count files per category (respecting search)
  const categoryCounts: Record<string, number> = {};
  for (const cat of CATEGORIES) {
    if (cat === "all") continue;
    categoryCounts[cat] = files.filter(
      (f) =>
        matchesCategory(f.content_type, cat) &&
        (deferredSearch === "" ||
          f.filename.toLowerCase().includes(deferredSearch.toLowerCase()))
    ).length;
  }
  // "all" count = total files matching search (some files may not match any specific category)
  const allCount = files.filter(
    (f) =>
      deferredSearch === "" ||
      f.filename.toLowerCase().includes(deferredSearch.toLowerCase())
  ).length;

  const categoryLabels: Record<CategoryFilter, string> = {
    all: t("allCategories"),
    images: t("images"),
    documents: t("documents"),
    slides: t("slides"),
    code: t("code"),
    data: t("data"),
  };

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center py-24 gap-3">
        <Loader2 className="w-5 h-5 animate-spin text-muted-foreground" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center py-24 gap-3">
        <div className="w-10 h-10 rounded-full bg-destructive/10 flex items-center justify-center">
          <AlertCircle className="w-5 h-5 text-destructive" />
        </div>
        <p className="text-sm text-muted-foreground">{error}</p>
        <button
          onClick={loadFiles}
          className="text-sm text-foreground font-medium underline underline-offset-4 hover:text-foreground/80 cursor-pointer"
        >
          {t("retry")}
        </button>
      </div>
    );
  }

  if (files.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-24 gap-3">
        <div className="w-10 h-10 rounded-full bg-secondary flex items-center justify-center">
          <FolderOpen className="w-5 h-5 text-muted-foreground" />
        </div>
        <p className="text-sm font-medium text-foreground">{t("noFiles")}</p>
        <p className="text-xs text-muted-foreground">{t("noFilesDescription")}</p>
      </div>
    );
  }

  return (
    <div className="space-y-5">
      {/* Search and filters row */}
      <div className="flex flex-col sm:flex-row gap-3">
        {/* Search input */}
        <div className="relative flex-1 max-w-md">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground pointer-events-none" />
          <Input
            placeholder={t("search")}
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="pl-9 h-9"
          />
        </div>

        {/* Category filter pills */}
        <div className="flex items-center gap-1.5 flex-wrap">
          {CATEGORIES.map((cat) => {
            const count = cat === "all" ? allCount : (categoryCounts[cat] || 0);
            return (
              <button
                key={cat}
                onClick={() => setCategoryFilter(cat)}
                className={cn(
                  "flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium cursor-pointer",
                  "transition-colors duration-150",
                  categoryFilter === cat
                    ? "bg-foreground text-background"
                    : "text-muted-foreground hover:text-foreground hover:bg-secondary"
                )}
              >
                {CATEGORY_ICONS[cat]}
                <span>{categoryLabels[cat]}</span>
                <span
                  className={cn(
                    "ml-0.5 text-[10px] tabular-nums",
                    categoryFilter === cat
                      ? "text-background/60"
                      : "text-muted-foreground/60"
                  )}
                >
                  {count}
                </span>
              </button>
            );
          })}
        </div>
      </div>

      {/* Results */}
      {filteredFiles.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-16 gap-2">
          <Search className="w-5 h-5 text-muted-foreground/50" />
          <p className="text-sm text-muted-foreground">{t("noResults")}</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
          {filteredFiles.map((file, idx) => (
            <FileCard key={file.id} file={file} index={idx} onDelete={handleDelete} />
          ))}
        </div>
      )}
    </div>
  );
}
