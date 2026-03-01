"use client";

import React, { useEffect, useCallback, useMemo, useState } from "react";
import {
    Folder,
    FolderOpen,
    ChevronRight,
    ChevronLeft,
    RefreshCw,
    Loader2,
    Search,
    FileText,
    FileCode,
    FileJson,
    FileImage,
    FileType,
    FileSpreadsheet,
    FileCog,
    FileArchive,
    FolderClosed,
    Home,
    X,
    Pencil,
    Save,
} from "lucide-react";
import { useTranslations } from "next-intl";
import { cn } from "@/lib/utils";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { useComputerStore, type FileEntry } from "@/lib/stores/computer-store";
import { Badge } from "@/components/ui/badge";
import { ComputerFileContent } from "./computer-file-content";
import { ComputerEmptyState } from "./computer-empty-state";

interface ComputerFileViewProps {
    className?: string;
}

function getFileIcon(filename: string, isDirectory: boolean, isSelected: boolean) {
    if (isDirectory) {
        const Icon = isSelected ? FolderOpen : Folder;
        return <Icon className="w-4 h-4 flex-shrink-0 text-primary" />;
    }

    const ext = filename.split(".").pop()?.toLowerCase() || "";
    const lowerName = filename.toLowerCase();

    // Code files
    if (["ts", "tsx", "js", "jsx", "mjs", "cjs"].includes(ext)) {
        return <FileCode className="w-4 h-4 flex-shrink-0 text-muted-foreground" />;
    }
    if (["py", "pyw", "pyi"].includes(ext)) {
        return <FileCode className="w-4 h-4 flex-shrink-0 text-muted-foreground" />;
    }
    if (["go", "rs", "c", "cpp", "h", "hpp", "java", "kt", "scala", "swift", "rb", "php", "lua", "r"].includes(ext)) {
        return <FileCode className="w-4 h-4 flex-shrink-0 text-muted-foreground" />;
    }
    if (["sh", "bash", "zsh", "fish"].includes(ext)) {
        return <FileCode className="w-4 h-4 flex-shrink-0 text-muted-foreground" />;
    }

    // Web files
    if (["html", "htm", "vue", "svelte"].includes(ext)) {
        return <FileCode className="w-4 h-4 flex-shrink-0 text-muted-foreground" />;
    }
    if (["css", "scss", "less", "sass"].includes(ext)) {
        return <FileCode className="w-4 h-4 flex-shrink-0 text-muted-foreground" />;
    }

    // Data / config files
    if (["json", "jsonl", "json5"].includes(ext)) {
        return <FileJson className="w-4 h-4 flex-shrink-0 text-muted-foreground" />;
    }
    if (["yml", "yaml", "toml", "ini", "env", "cfg"].includes(ext)) {
        return <FileCog className="w-4 h-4 flex-shrink-0 text-muted-foreground" />;
    }
    if (["xml", "svg"].includes(ext)) {
        return <FileCode className="w-4 h-4 flex-shrink-0 text-muted-foreground" />;
    }
    if (["csv", "tsv", "xls", "xlsx"].includes(ext)) {
        return <FileSpreadsheet className="w-4 h-4 flex-shrink-0 text-muted-foreground" />;
    }
    if (["sql", "db", "sqlite"].includes(ext)) {
        return <FileSpreadsheet className="w-4 h-4 flex-shrink-0 text-muted-foreground" />;
    }

    // Image files
    if (["png", "jpg", "jpeg", "gif", "webp", "ico", "bmp", "tiff"].includes(ext)) {
        return <FileImage className="w-4 h-4 flex-shrink-0 text-muted-foreground" />;
    }

    // Document files
    if (["md", "mdx", "txt", "rst", "adoc"].includes(ext)) {
        return <FileText className="w-4 h-4 flex-shrink-0 text-muted-foreground" />;
    }
    if (["pdf", "doc", "docx"].includes(ext)) {
        return <FileType className="w-4 h-4 flex-shrink-0 text-muted-foreground" />;
    }

    // Archives
    if (["zip", "tar", "gz", "bz2", "xz", "7z", "rar"].includes(ext)) {
        return <FileArchive className="w-4 h-4 flex-shrink-0 text-muted-foreground" />;
    }

    // Config-like files without extension
    if (["dockerfile", "makefile", "rakefile", "gemfile", "procfile"].includes(lowerName)) {
        return <FileCog className="w-4 h-4 flex-shrink-0 text-muted-foreground" />;
    }
    if (lowerName.startsWith(".") || lowerName === "license" || lowerName === "readme") {
        return <FileText className="w-4 h-4 flex-shrink-0 text-muted-foreground" />;
    }

    // Lock files
    if (ext === "lock" || lowerName.endsWith("-lock.json") || lowerName.endsWith(".lockb")) {
        return <FileCog className="w-4 h-4 flex-shrink-0 text-muted-foreground" />;
    }

    return <FileText className="w-4 h-4 flex-shrink-0 text-muted-foreground" />;
}

// Format relative time for modification dates
function formatModifiedTime(date: Date | undefined, t: (key: string, params?: Record<string, string | number | Date>) => string): string | null {
    if (!date) return null;
    const now = Date.now();
    const modified = new Date(date).getTime();
    const diffMs = now - modified;
    const diffMin = Math.floor(diffMs / 60000);
    const diffHr = Math.floor(diffMs / 3600000);
    const diffDay = Math.floor(diffMs / 86400000);

    if (diffMin < 1) return t("time.now");
    if (diffMin < 60) return t("time.minutes", { count: diffMin });
    if (diffHr < 24) return t("time.hours", { count: diffHr });
    if (diffDay < 30) return t("time.days", { count: diffDay });
    return new Date(date).toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

// Check if a file was recently modified (within last 5 minutes)
function isRecentlyModified(date?: Date): boolean {
    if (!date) return false;
    const now = Date.now();
    const modified = new Date(date).getTime();
    return now - modified < 5 * 60 * 1000;
}

function Breadcrumb({
    path,
    onNavigate,
}: {
    path: string;
    onNavigate: (path: string) => void;
}) {
    const parts = path.split("/").filter(Boolean);

    return (
        <div className="flex items-center gap-1 px-3 py-2 text-xs border-b border-border/30 bg-secondary/30 overflow-x-auto">
            <button
                onClick={() => onNavigate("/")}
                className="text-muted-foreground hover:text-foreground hover:bg-secondary/80 px-1 py-0.5 rounded transition-colors focus-visible:ring-2 focus-visible:ring-primary focus-visible:outline-none"
            >
                <Home className="w-3.5 h-3.5" />
            </button>
            {parts.map((part, index) => {
                const fullPath = "/" + parts.slice(0, index + 1).join("/");
                const isLast = index === parts.length - 1;
                return (
                    <React.Fragment key={fullPath}>
                        <ChevronRight className="w-3 h-3 text-muted-foreground/50" />
                        <button
                            onClick={() => !isLast && onNavigate(fullPath)}
                            className={cn(
                                "transition-colors truncate max-w-[160px] px-1 py-0.5 rounded focus-visible:ring-2 focus-visible:ring-primary focus-visible:outline-none",
                                isLast
                                    ? "text-foreground font-medium"
                                    : "text-muted-foreground hover:text-foreground hover:bg-secondary/80 cursor-pointer"
                            )}
                            disabled={isLast}
                        >
                            {part}
                        </button>
                    </React.Fragment>
                );
            })}
        </div>
    );
}

function FileItem({
    entry,
    isSelected,
    isChanged,
    onClick,
    onDoubleClick,
}: {
    entry: FileEntry;
    isSelected: boolean;
    isChanged: boolean;
    onClick: () => void;
    onDoubleClick: () => void;
}) {
    const t = useTranslations("computer");
    const isDir = entry.type === "directory";

    // Format file size
    const formatSize = (bytes?: number) => {
        if (bytes === undefined) return "";
        if (bytes < 1024) return `${bytes} B`;
        if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
        return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
    };

    const modifiedLabel = formatModifiedTime(entry.modifiedAt, t);
    const recentlyModified = isRecentlyModified(entry.modifiedAt);

    return (
        <button
            onClick={onClick}
            onDoubleClick={onDoubleClick}
            className={cn(
                "w-full flex items-center gap-2.5 px-3 py-2 text-left group",
                "hover:bg-secondary/80 transition-colors cursor-pointer",
                "focus-visible:ring-2 focus-visible:ring-primary focus-visible:outline-none",
                isSelected
                    ? "bg-primary/8 border-l-2 border-l-primary"
                    : "border-l-2 border-l-transparent",
                isChanged && "bg-primary/5"
            )}
            role="treeitem"
            aria-selected={isSelected}
            aria-expanded={isDir ? isSelected : undefined}
        >
            {getFileIcon(entry.name, isDir, isSelected)}
            <span className={cn(
                "flex-1 text-sm truncate",
                isSelected && "font-medium"
            )}>
                {entry.name}
            </span>
            {isChanged && (
                <Badge variant="subtle">{t("workspace.updated")}</Badge>
            )}
            {!isChanged && recentlyModified && (
                <Badge variant="subtle">{t("workspace.modified")}</Badge>
            )}
            {!isDir && modifiedLabel && !recentlyModified && !isChanged && (
                <span className="text-xs text-muted-foreground/70">
                    {modifiedLabel}
                </span>
            )}
            {!isDir && entry.size !== undefined && (
                <span className="text-xs text-muted-foreground/80 tabular-nums">{formatSize(entry.size)}</span>
            )}
            {isDir && (
                <ChevronRight className="w-3.5 h-3.5 text-muted-foreground/40 group-hover:text-muted-foreground transition-colors" />
            )}
        </button>
    );
}

export function ComputerFileView({ className }: ComputerFileViewProps) {
    const t = useTranslations("computer");
    const [searchQuery, setSearchQuery] = useState("");
    const [isSearchOpen, setIsSearchOpen] = useState(false);

    // Per-conversation state via a single consolidated selector
    const convState = useComputerStore((state) => {
        const id = state.activeConversationId;
        if (!id) return null;
        return state.conversationStates[id] ?? null;
    });

    const currentPath = convState?.currentPath ?? "/home/user";
    const files = convState?.files ?? [];
    const selectedFile = convState?.selectedFile ?? null;
    const workspaceSandboxType = convState?.workspaceSandboxType ?? null;
    const workspaceTaskId = convState?.workspaceTaskId ?? null;
    const fileContent = convState?.fileContent ?? null;
    const fileContentLoading = convState?.fileContentLoading ?? false;
    const fileContentError = convState?.fileContentError ?? null;
    const fileContentIsBinary = convState?.fileContentIsBinary ?? false;
    const changedFiles = convState?.changedFiles ?? [];

    // Set of recently changed file paths for quick lookup
    const changedFilesSet = useMemo(() => new Set(changedFiles), [changedFiles]);
    const isSelectedFileChanged = selectedFile ? changedFilesSet.has(selectedFile) : false;

    // Actions
    const setSelectedFile = useComputerStore((state) => state.setSelectedFile);
    const loadWorkspaceFiles = useComputerStore((state) => state.loadWorkspaceFiles);
    const loadFileContent = useComputerStore((state) => state.loadFileContent);

    const [isRefreshing, setIsRefreshing] = React.useState(false);

    // Load files when component mounts or path changes
    // Track a generation counter to discard stale responses
    const loadGenRef = React.useRef(0);
    useEffect(() => {
        if (workspaceSandboxType && workspaceTaskId) {
            const gen = ++loadGenRef.current;
            loadWorkspaceFiles(currentPath).then(() => {
                // If another navigation happened while this was in flight, discard
                if (gen !== loadGenRef.current) return;
            });
        }
        // Bump generation on cleanup to invalidate any in-flight request
        const ref = loadGenRef;
        return () => {
            ref.current++;
        };
    }, [workspaceSandboxType, workspaceTaskId, currentPath, loadWorkspaceFiles]);

    // Sort files: directories first, then alphabetically
    // Also filter by search query if present
    const sortedFiles = useMemo(() => {
        let filtered = [...files];
        if (searchQuery.trim()) {
            const query = searchQuery.toLowerCase();
            filtered = filtered.filter((f) => f.name.toLowerCase().includes(query));
        }
        return filtered.sort((a, b) => {
            if (a.type !== b.type) {
                return a.type === "directory" ? -1 : 1;
            }
            return a.name.localeCompare(b.name);
        });
    }, [files, searchQuery]);

    const handlePathChange = useCallback(
        (path: string) => {
            setSelectedFile(null);
            setSearchQuery("");
            loadWorkspaceFiles(path);
        },
        [setSelectedFile, loadWorkspaceFiles]
    );

    const handleFileSelect = useCallback(
        (entry: FileEntry) => {
            if (entry.type === "directory") {
                // Single click on directory - just select it
                setSelectedFile(entry.path);
            } else {
                // Single click on file - select and load content
                setSelectedFile(entry.path);
                loadFileContent(entry.path);
            }
        },
        [setSelectedFile, loadFileContent]
    );

    const handleDoubleClick = useCallback(
        (entry: FileEntry) => {
            if (entry.type === "directory") {
                handlePathChange(entry.path);
            }
        },
        [handlePathChange]
    );

    const handleParentClick = useCallback(() => {
        const parentPath = currentPath.split("/").slice(0, -1).join("/") || "/";
        handlePathChange(parentPath);
    }, [currentPath, handlePathChange]);

    const handleRefresh = useCallback(async () => {
        setIsRefreshing(true);
        try {
            await loadWorkspaceFiles(currentPath);
        } finally {
            setIsRefreshing(false);
        }
    }, [loadWorkspaceFiles, currentPath]);

    const toggleSearch = useCallback(() => {
        setIsSearchOpen((prev) => {
            if (prev) setSearchQuery("");
            return !prev;
        });
    }, []);

    // Get filename from selected path
    const selectedFileName = selectedFile ? selectedFile.split("/").pop() || "" : "";

    // Check if workspace is connected
    const isConnected = workspaceSandboxType && workspaceTaskId;

    // Determine if we're viewing file content (drill-down mode)
    const isViewingContent = selectedFile && !files.find(
        (f) => f.path === selectedFile && f.type === "directory"
    );

    // Recent files from store (persisted per conversation)
    const recentFiles = useComputerStore((s) => s.getRecentFiles());
    const addRecentFile = useComputerStore((s) => s.addRecentFile);

    // Track recently opened files in the store
    useEffect(() => {
        if (selectedFile && !files.find((f) => f.path === selectedFile && f.type === "directory")) {
            addRecentFile(selectedFile);
        }
    }, [selectedFile, files, addRecentFile]);

    // Editing state
    const [isEditing, setIsEditing] = useState(false);
    const [editContent, setEditContent] = useState("");
    const [isSaving, setIsSaving] = useState(false);

    // Reset editing state when switching files
    useEffect(() => {
        setIsEditing(false);
        setEditContent("");
    }, [selectedFile]);

    const handleStartEditing = useCallback(() => {
        if (fileContent !== null) {
            setEditContent(fileContent);
            setIsEditing(true);
        }
    }, [fileContent]);

    const handleCancelEditing = useCallback(() => {
        setIsEditing(false);
        setEditContent("");
    }, []);

    const handleSaveFile = useCallback(async () => {
        if (!workspaceTaskId || !selectedFile) return;

        setIsSaving(true);
        try {
            const response = await fetch(`/api/v1/sandbox/files/write`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    path: selectedFile,
                    content: editContent,
                    task_id: workspaceTaskId,
                    sandbox_type: workspaceSandboxType || "execution",
                }),
            });

            if (response.ok) {
                setIsEditing(false);
                // Reload file content
                loadFileContent(selectedFile);
            }
        } catch (error) {
            console.error("Failed to save file:", error);
        } finally {
            setIsSaving(false);
        }
    }, [workspaceTaskId, selectedFile, editContent, workspaceSandboxType, loadFileContent]);

    const handleBackToList = useCallback(() => {
        setSelectedFile(null);
        setIsEditing(false);
    }, [setSelectedFile]);

    return (
        <div className={cn("flex-1 flex flex-col overflow-hidden bg-background", className)}>
            {/* Header with refresh and search buttons */}
            <div className="flex items-center justify-between px-3 py-2 border-b border-border bg-secondary/20">
                {isViewingContent ? (
                    // Back button when viewing file content
                    <button
                        onClick={handleBackToList}
                        className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground hover:text-foreground transition-colors cursor-pointer focus-visible:ring-2 focus-visible:ring-primary focus-visible:outline-none rounded"
                    >
                        <ChevronLeft className="w-3.5 h-3.5" />
                        {t("workspace.title")}
                    </button>
                ) : (
                    <span className="text-xs font-medium text-muted-foreground">
                        {t("workspace.title")}
                    </span>
                )}
                <div className="flex items-center gap-1">
                    {!isViewingContent && (
                        <Button
                            variant="ghost"
                            size="icon"
                            className="h-8 w-8"
                            onClick={toggleSearch}
                            disabled={!isConnected}
                            aria-label={t("workspace.search")}
                            aria-expanded={isSearchOpen}
                        >
                            <Search className={cn("w-3.5 h-3.5", isSearchOpen && "text-primary")} />
                        </Button>
                    )}
                    <Button
                        variant="ghost"
                        size="icon"
                        className="h-8 w-8"
                        onClick={handleRefresh}
                        disabled={!isConnected || isRefreshing}
                        aria-label={t("workspace.refresh")}
                    >
                        {isRefreshing ? (
                            <Loader2 className="w-3.5 h-3.5 animate-spin" />
                        ) : (
                            <RefreshCw className="w-3.5 h-3.5" />
                        )}
                    </Button>
                </div>
            </div>

            {!isConnected ? (
                // No workspace connected
                <ComputerEmptyState
                    icon={FolderClosed}
                    title={t("workspace.empty")}
                />
            ) : isViewingContent ? (
                // Drill-down: full-width file content view with edit support
                isEditing ? (
                    <div className="flex-1 flex flex-col overflow-hidden">
                        {/* Editing header */}
                        <div className="flex items-center justify-between px-3 py-1.5 border-b border-border/50 bg-primary/5">
                            <div className="flex items-center gap-2 min-w-0">
                                <Pencil className="w-3.5 h-3.5 text-primary" />
                                <span className="text-xs font-mono text-foreground truncate max-w-[200px]">
                                    {selectedFileName}
                                </span>
                                <Badge variant="subtle">{t("editing")}</Badge>
                            </div>
                            <div className="flex items-center gap-1">
                                <Button
                                    variant="ghost"
                                    size="sm"
                                    className="h-8 px-2 text-xs text-muted-foreground hover:text-foreground"
                                    onClick={handleCancelEditing}
                                    disabled={isSaving}
                                >
                                    <X className="w-3.5 h-3.5 mr-1" />
                                    {t("cancelEdit")}
                                </Button>
                                <Button
                                    variant="default"
                                    size="sm"
                                    className="h-8 px-2 text-xs"
                                    onClick={handleSaveFile}
                                    disabled={isSaving}
                                >
                                    {isSaving ? (
                                        <Loader2 className="w-3.5 h-3.5 mr-1 animate-spin" />
                                    ) : (
                                        <Save className="w-3.5 h-3.5 mr-1" />
                                    )}
                                    {t("saveFile")}
                                </Button>
                            </div>
                        </div>
                        {/* Edit textarea */}
                        <textarea
                            value={editContent}
                            onChange={(e) => setEditContent(e.target.value)}
                            className="flex-1 w-full p-3 bg-background text-foreground font-mono text-xs leading-relaxed outline-none resize-none border-0"
                            spellCheck={false}
                            autoFocus
                        />
                    </div>
                ) : (
                    <div className="flex-1 flex flex-col overflow-hidden">
                        {/* Recent files tabs */}
                        {recentFiles.length > 1 && (
                            <div className="flex items-center gap-0.5 px-2 py-1 border-b border-border/30 bg-secondary/10 overflow-x-auto">
                                {recentFiles.map((filePath) => {
                                    const name = filePath.split("/").pop() || "";
                                    const isActive = filePath === selectedFile;
                                    return (
                                        <button
                                            key={filePath}
                                            onClick={() => {
                                                setSelectedFile(filePath);
                                                loadFileContent(filePath);
                                            }}
                                            className={cn(
                                                "px-2 py-1 text-xs rounded truncate max-w-[120px] transition-colors focus-visible:ring-2 focus-visible:ring-primary focus-visible:outline-none",
                                                isActive
                                                    ? "bg-secondary text-foreground font-medium"
                                                    : "text-muted-foreground hover:text-foreground hover:bg-secondary/60"
                                            )}
                                            title={filePath}
                                        >
                                            {name}
                                        </button>
                                    );
                                })}
                            </div>
                        )}
                        {/* Edit button overlay for text files */}
                        {fileContent !== null && !fileContentIsBinary && !fileContentLoading && (
                            <div className="flex items-center justify-end px-3 py-1 border-b border-border/30 bg-secondary/10 shrink-0">
                                <Button
                                    variant="ghost"
                                    size="sm"
                                    className="h-8 px-2 text-xs text-muted-foreground hover:text-foreground"
                                    onClick={handleStartEditing}
                                >
                                    <Pencil className="w-3.5 h-3.5 mr-1" />
                                    {t("editFile")}
                                </Button>
                            </div>
                        )}
                        {/* File updated banner */}
                        {isSelectedFileChanged && !fileContentLoading && (
                            <div className="flex items-center justify-between px-3 py-1.5 border-b border-primary/20 bg-primary/5 shrink-0">
                                <span className="text-xs text-primary">
                                    {t("workspace.fileUpdated")}
                                </span>
                                <Button
                                    variant="ghost"
                                    size="sm"
                                    className="h-8 px-2 text-xs text-primary hover:text-primary/80"
                                    onClick={() => selectedFile && loadFileContent(selectedFile)}
                                >
                                    <RefreshCw className="w-3 h-3 mr-1" />
                                    {t("workspace.reload")}
                                </Button>
                            </div>
                        )}
                        <ComputerFileContent
                            filename={selectedFileName}
                            content={fileContent}
                            isLoading={fileContentLoading}
                            error={fileContentError}
                            isBinary={fileContentIsBinary}
                            className="flex-1"
                        />
                    </div>
                )
            ) : (
                // Full-width file list
                <div className="flex-1 flex flex-col overflow-hidden">
                    {/* Breadcrumb navigation */}
                    <Breadcrumb path={currentPath} onNavigate={handlePathChange} />

                    {/* Search input */}
                    {isSearchOpen && (
                        <div className="px-2 py-1.5 border-b border-border/30 bg-secondary/10">
                            <div className="relative">
                                <Search className="absolute left-2 top-1/2 -translate-y-1/2 w-3 h-3 text-muted-foreground/50" />
                                <Input
                                    value={searchQuery}
                                    onChange={(e) => setSearchQuery(e.target.value)}
                                    placeholder={t("workspace.searchPlaceholder")}
                                    className="h-7 text-xs pl-7 pr-7 border-border/30 bg-background/50"
                                    autoFocus
                                />
                                {searchQuery && (
                                    <button
                                        onClick={() => setSearchQuery("")}
                                        className="absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground/50 hover:text-muted-foreground"
                                        aria-label={t("workspace.clearSearch")}
                                    >
                                        <X className="w-3 h-3" />
                                    </button>
                                )}
                            </div>
                        </div>
                    )}

                    {/* File list */}
                    <ScrollArea className="flex-1">
                        <div className="py-0.5" role="tree" aria-label={t("workspace.title")}>
                            {/* Parent directory */}
                            {currentPath !== "/" && !searchQuery && (
                                <button
                                    onClick={handleParentClick}
                                    className="w-full flex items-center gap-2.5 px-3 py-2 text-left hover:bg-secondary/80 transition-colors border-l-2 border-l-transparent cursor-pointer focus-visible:ring-2 focus-visible:ring-primary focus-visible:outline-none"
                                    role="treeitem"
                                    aria-selected={false}
                                    aria-label={t("workspace.parentDirectory")}
                                >
                                    <Folder className="w-4 h-4 text-primary" />
                                    <span className="text-sm text-muted-foreground">..</span>
                                </button>
                            )}

                            {/* Files */}
                            {sortedFiles.length === 0 ? (
                                <div className="px-3 py-8 flex flex-col items-center text-center">
                                    {searchQuery ? (
                                        <>
                                            <Search className="w-6 h-6 text-muted-foreground/30 mb-2" />
                                            <p className="text-sm text-muted-foreground/60">
                                                {t("workspace.noResults")}
                                            </p>
                                        </>
                                    ) : (
                                        <>
                                            <FolderOpen className="w-6 h-6 text-muted-foreground/30 mb-2" />
                                            <p className="text-sm text-muted-foreground/60">
                                                {t("emptyDirectory")}
                                            </p>
                                        </>
                                    )}
                                </div>
                            ) : (
                                sortedFiles.map((entry) => (
                                    <FileItem
                                        key={entry.path}
                                        entry={entry}
                                        isSelected={selectedFile === entry.path}
                                        isChanged={changedFilesSet.has(entry.path)}
                                        onClick={() => handleFileSelect(entry)}
                                        onDoubleClick={() => handleDoubleClick(entry)}
                                    />
                                ))
                            )}
                        </div>
                    </ScrollArea>

                    {/* File count footer */}
                    {files.length > 0 && (
                        <div className="px-3 py-1.5 border-t border-border/50 bg-secondary/10">
                            <span className="text-xs text-muted-foreground/70 tabular-nums">
                                {files.filter((f) => f.type === "file").length} {t("workspace.fileCount")}
                                {files.filter((f) => f.type === "directory").length > 0 &&
                                    ` / ${files.filter((f) => f.type === "directory").length} ${t("workspace.folderCount")}`}
                            </span>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}
