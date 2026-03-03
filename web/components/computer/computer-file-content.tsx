"use client";

import React, { useState, useCallback, useEffect } from "react";
import { useCopyToClipboard } from "@/lib/hooks/use-copy-to-clipboard";
import { downloadBase64, downloadText } from "@/lib/utils/download";
import { Badge } from "@/components/ui/badge";
import { ComputerEmptyState } from "./computer-empty-state";
import {
    FileText,
    Download,
    Loader2,
    AlertCircle,
    FileCode,
    Copy,
    Check,
    FileSearch,
    Maximize2,
    X,
    Eye,
    Code2,
} from "lucide-react";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneDark, oneLight } from "react-syntax-highlighter/dist/esm/styles/prism";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { useTranslations } from "next-intl";
import { cn } from "@/lib/utils";
import { useTheme } from "@/lib/hooks/use-theme";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Button } from "@/components/ui/button";
import {
    getLanguageFromFilename,
    getMimeType,
    isImageFile,
    isSvgFile,
    isVideoFile,
    isAudioFile,
    isPdfFile,
    isMarkdownFile,
} from "@/lib/utils/file-types";

interface ComputerFileContentProps {
    filename: string;
    content: string | null;
    isLoading: boolean;
    error: string | null;
    isBinary: boolean;
    onDownload?: () => void;
    className?: string;
}

function ContentHeader({
    filename,
    language,
    content,
    isBinary,
    onDownload,
}: {
    filename: string;
    language: string | null;
    content: string | null;
    isBinary?: boolean;
    onDownload?: () => void;
}) {
    const t = useTranslations("computer");
    const { copied, copy } = useCopyToClipboard(2000);

    const handleCopy = useCallback(() => {
        if (!content) return;
        copy(content);
    }, [content, copy]);

    const handleDownload = useCallback(() => {
        if (!content || !filename) return;
        if (onDownload) {
            onDownload();
            return;
        }
        if (isBinary) {
            downloadBase64(content, filename, getMimeType(filename));
        } else {
            downloadText(content, filename);
        }
    }, [content, filename, isBinary, onDownload]);

    const icon = language ? (
        <FileCode className="w-3.5 h-3.5 text-muted-foreground" />
    ) : (
        <FileText className="w-3.5 h-3.5 text-muted-foreground" />
    );

    return (
        <div className="flex items-center justify-between px-3 py-1.5 border-b border-border/30 bg-secondary/10">
            <div className="flex items-center gap-2 min-w-0">
                {icon}
                <span className="text-xs font-mono text-muted-foreground truncate max-w-[240px]">
                    {filename}
                </span>
                {language && (
                    <Badge variant="subtle">{language}</Badge>
                )}
            </div>
            <div className="flex items-center gap-0.5">
                {content && !isBinary && (
                    <Button
                        variant="ghost"
                        size="icon"
                        className="h-8 w-8"
                        onClick={handleCopy}
                        title={copied ? t("workspace.copied") : t("workspace.copyContent")}
                    >
                        {copied ? (
                            <Check className="w-3.5 h-3.5 text-primary" />
                        ) : (
                            <Copy className="w-3.5 h-3.5" />
                        )}
                    </Button>
                )}
                {content && (
                    <Button
                        variant="ghost"
                        size="icon"
                        className="h-8 w-8"
                        onClick={handleDownload}
                        title={t("workspace.download")}
                    >
                        <Download className="w-3.5 h-3.5" />
                    </Button>
                )}
            </div>
        </div>
    );
}

export function ComputerFileContent({
    filename,
    content,
    isLoading,
    error,
    isBinary,
    onDownload,
    className,
}: ComputerFileContentProps) {
    const t = useTranslations("computer");
    const { resolvedTheme } = useTheme();
    const isDark = resolvedTheme === "dark";
    const [isFullscreen, setIsFullscreen] = useState(false);
    const [markdownMode, setMarkdownMode] = useState<"preview" | "raw">("preview");

    const language = getLanguageFromFilename(filename);
    const isImage = isImageFile(filename);
    const isSvg = isSvgFile(filename);
    const isVideo = isVideoFile(filename);
    const isAudio = isAudioFile(filename);
    const isPdf = isPdfFile(filename);
    const isMarkdown = isMarkdownFile(filename);

    // Loading state
    if (isLoading) {
        return (
            <div className={cn("flex flex-col items-center justify-center h-full py-12", className)}>
                <Loader2 className="w-8 h-8 text-muted-foreground animate-spin mb-3" />
                <p className="text-sm text-muted-foreground">{t("workspace.loading")}</p>
            </div>
        );
    }

    // Error state
    if (error) {
        return (
            <div className={cn("flex flex-col items-center justify-center h-full py-12", className)}>
                <div className="w-12 h-12 rounded-xl bg-destructive/10 flex items-center justify-center mb-3">
                    <AlertCircle className="w-6 h-6 text-destructive" />
                </div>
                <p className="text-sm text-destructive text-center px-4">{error}</p>
            </div>
        );
    }

    // No content state
    if (content === null) {
        return (
            <ComputerEmptyState
                icon={FileSearch}
                title={t("workspace.selectFile")}
                className={cn("h-full", className)}
            />
        );
    }

    // --- SVG file (text content, not binary) ---
    if (isSvg && !isBinary) {
        const dataUri = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(content)}`;
        return (
            <div className={cn("flex flex-col h-full", className)}>
                <ContentHeader filename={filename} language={null} content={content} onDownload={onDownload} />
                <div className="flex-1 flex items-center justify-center p-4 relative group">
                    <img
                        src={dataUri}
                        alt={filename}
                        className="max-w-full max-h-full object-contain rounded-lg cursor-pointer"
                        onClick={() => setIsFullscreen(true)}
                    />
                    <button
                        onClick={() => setIsFullscreen(true)}
                        className="absolute top-6 right-6 opacity-0 group-hover:opacity-100 transition-opacity p-2 rounded-lg bg-background/80 border border-border/50 hover:bg-secondary focus-visible:ring-2 focus-visible:ring-primary focus-visible:outline-none"
                        title={t("fullscreen")}
                    >
                        <Maximize2 className="w-4 h-4" />
                    </button>
                </div>
                {isFullscreen && (
                    <FullscreenModal
                        src={dataUri}
                        filename={filename}
                        onClose={() => setIsFullscreen(false)}
                        onDownload={() => downloadText(content, filename, "image/svg+xml")}
                    />
                )}
            </div>
        );
    }

    // --- Video file (binary) ---
    if (isVideo && isBinary) {
        const MAX_VIDEO_SIZE = 10 * 1024 * 1024; // 10MB base64 limit
        if (content.length > MAX_VIDEO_SIZE) {
            return (
                <BinaryDownloadFallback
                    filename={filename}
                    content={content}
                    t={t}
                    className={className}
                    message={t("workspace.fileTooLarge", { size: (content.length / (1024 * 1024)).toFixed(1) })}
                />
            );
        }
        const mimeType = getMimeType(filename);
        return (
            <div className={cn("flex flex-col h-full", className)}>
                <ContentHeader filename={filename} language={null} content={content} isBinary onDownload={onDownload} />
                <div className="flex-1 flex items-center justify-center p-4">
                    <video
                        controls
                        className="max-w-full max-h-full rounded-lg"
                    >
                        <source src={`data:${mimeType};base64,${content}`} type={mimeType} />
                    </video>
                </div>
            </div>
        );
    }

    // --- Audio file (binary) ---
    if (isAudio && isBinary) {
        const mimeType = getMimeType(filename);
        return (
            <div className={cn("flex flex-col h-full", className)}>
                <ContentHeader filename={filename} language={null} content={content} isBinary onDownload={onDownload} />
                <div className="flex-1 flex items-center justify-center p-8">
                    <audio controls className="w-full max-w-md" src={`data:${mimeType};base64,${content}`} />
                </div>
            </div>
        );
    }

    // --- PDF file (binary) ---
    if (isPdf && isBinary) {
        return (
            <PdfPreview
                content={content}
                filename={filename}
                onDownload={onDownload}
                className={className}
            />
        );
    }

    // --- Binary file with download (generic binary that's not image/video/audio/pdf) ---
    if (isBinary && !isImage) {
        return (
            <BinaryDownloadFallback
                filename={filename}
                content={content}
                t={t}
                className={className}
            />
        );
    }

    // --- Image file (binary, base64) with fullscreen ---
    if (isImage && isBinary) {
        const MAX_IMAGE_SIZE = 5 * 1024 * 1024; // 5MB
        if (content.length > MAX_IMAGE_SIZE) {
            return (
                <div className={cn("flex flex-col items-center justify-center h-full py-12", className)}>
                    <FileText className="w-12 h-12 text-muted-foreground/50 mb-4" />
                    <p className="text-sm font-medium mb-2">{filename}</p>
                    <p className="text-xs text-muted-foreground">
                        {t("workspace.fileTooLarge", { size: (content.length / (1024 * 1024)).toFixed(1) })}
                    </p>
                </div>
            );
        }

        const mimeType = getMimeType(filename);
        const imageSrc = `data:${mimeType};base64,${content}`;

        return (
            <div className={cn("flex flex-col h-full", className)}>
                <ContentHeader filename={filename} language={null} content={content} isBinary onDownload={onDownload} />
                <div className="flex-1 flex items-center justify-center p-4 relative group">
                    <img
                        src={imageSrc}
                        alt={filename}
                        className="max-w-full max-h-full object-contain rounded-lg cursor-pointer"
                        onClick={() => setIsFullscreen(true)}
                    />
                    <button
                        onClick={() => setIsFullscreen(true)}
                        className="absolute top-6 right-6 opacity-0 group-hover:opacity-100 transition-opacity p-2 rounded-lg bg-background/80 border border-border/50 hover:bg-secondary focus-visible:ring-2 focus-visible:ring-primary focus-visible:outline-none"
                        title={t("fullscreen")}
                    >
                        <Maximize2 className="w-4 h-4" />
                    </button>
                </div>
                {isFullscreen && (
                    <FullscreenModal
                        src={imageSrc}
                        filename={filename}
                        onClose={() => setIsFullscreen(false)}
                        onDownload={() => downloadBase64(content, filename, mimeType)}
                    />
                )}
            </div>
        );
    }

    // --- Markdown file with preview/raw toggle ---
    if (isMarkdown && !isBinary) {
        return (
            <div className={cn("flex flex-col h-full", className)}>
                <div className="flex items-center justify-between px-3 py-1.5 border-b border-border/30 bg-secondary/10">
                    <div className="flex items-center gap-2 min-w-0">
                        <FileText className="w-3.5 h-3.5 text-muted-foreground" />
                        <span className="text-xs font-mono text-muted-foreground truncate max-w-[240px]">
                            {filename}
                        </span>
                        <Badge variant="subtle">markdown</Badge>
                    </div>
                    <div className="flex items-center gap-0.5">
                        <Button
                            variant="ghost"
                            size="icon"
                            className={cn("h-8 w-8", markdownMode === "preview" && "bg-primary/10 text-primary")}
                            onClick={() => setMarkdownMode("preview")}
                            title={t("workspace.preview")}
                        >
                            <Eye className="w-3.5 h-3.5" />
                        </Button>
                        <Button
                            variant="ghost"
                            size="icon"
                            className={cn("h-8 w-8", markdownMode === "raw" && "bg-primary/10 text-primary")}
                            onClick={() => setMarkdownMode("raw")}
                            title={t("workspace.raw")}
                        >
                            <Code2 className="w-3.5 h-3.5" />
                        </Button>
                        <Button
                            variant="ghost"
                            size="icon"
                            className="h-8 w-8"
                            onClick={() => downloadText(content, filename)}
                            title={t("workspace.download")}
                        >
                            <Download className="w-3.5 h-3.5" />
                        </Button>
                    </div>
                </div>
                <ScrollArea className="flex-1">
                    {markdownMode === "preview" ? (
                        <div className={cn(
                            "p-4 prose prose-sm prose-neutral dark:prose-invert max-w-none",
                            "prose-headings:font-bold prose-headings:tracking-tight",
                            "prose-a:text-primary prose-a:font-medium hover:prose-a:underline",
                            "prose-code:text-xs prose-code:bg-muted prose-code:px-1.5 prose-code:py-0.5 prose-code:rounded-md prose-code:before:content-none prose-code:after:content-none",
                            "prose-pre:bg-muted/50 prose-pre:border prose-pre:border-border/50 prose-pre:rounded-xl",
                            "prose-img:rounded-xl prose-img:border border-border/50",
                            "prose-blockquote:border-l-4 prose-blockquote:border-primary/20 prose-blockquote:bg-primary/5 prose-blockquote:py-1 prose-blockquote:px-5 prose-blockquote:rounded-r-lg"
                        )}>
                            <ReactMarkdown remarkPlugins={[remarkGfm]}>
                                {content}
                            </ReactMarkdown>
                        </div>
                    ) : (
                        <div className="p-0">
                            {content.length > 100 * 1024 ? (
                                <PlainTextWithLineNumbers content={content} />
                            ) : (
                                <SyntaxHighlighter
                                    language="markdown"
                                    style={isDark ? oneDark : oneLight}
                                    customStyle={{
                                        margin: 0,
                                        padding: "0.75rem 1rem",
                                        background: "transparent",
                                        fontSize: "12px",
                                        lineHeight: "1.5",
                                    }}
                                    showLineNumbers
                                    lineNumberStyle={{
                                        minWidth: "2.5em",
                                        paddingRight: "0.75em",
                                        opacity: 0.4,
                                        textAlign: "right",
                                    }}
                                >
                                    {content}
                                </SyntaxHighlighter>
                            )}
                        </div>
                    )}
                </ScrollArea>
            </div>
        );
    }

    // --- Code file with syntax highlighting ---
    const SYNTAX_HIGHLIGHT_SIZE_LIMIT = 100 * 1024; // 100KB
    if (language) {
        const usePlainText = content.length > SYNTAX_HIGHLIGHT_SIZE_LIMIT;
        return (
            <div className={cn("flex flex-col h-full", className)}>
                <ContentHeader
                    filename={filename}
                    language={language}
                    content={content}
                    onDownload={onDownload}
                />
                <ScrollArea className="flex-1">
                    <div className="p-0">
                        {usePlainText ? (
                            <PlainTextWithLineNumbers content={content} />
                        ) : (
                            <SyntaxHighlighter
                                language={language}
                                style={isDark ? oneDark : oneLight}
                                customStyle={{
                                    margin: 0,
                                    padding: "0.75rem 1rem",
                                    background: "transparent",
                                    fontSize: "12px",
                                    lineHeight: "1.5",
                                }}
                                showLineNumbers
                                lineNumberStyle={{
                                    minWidth: "2.5em",
                                    paddingRight: "0.75em",
                                    opacity: 0.4,
                                    textAlign: "right",
                                }}
                            >
                                {content}
                            </SyntaxHighlighter>
                        )}
                    </div>
                </ScrollArea>
            </div>
        );
    }

    // --- Plain text file with line numbers ---
    return (
        <div className={cn("flex flex-col h-full", className)}>
            <ContentHeader
                filename={filename}
                language={null}
                content={content}
                onDownload={onDownload}
            />
            <ScrollArea className="flex-1">
                <div className="p-0">
                    <PlainTextWithLineNumbers content={content} />
                </div>
            </ScrollArea>
        </div>
    );
}

// --- Sub-components ---

function PlainTextWithLineNumbers({ content }: { content: string }) {
    const lines = content.split("\n");
    const gutterWidth = String(lines.length).length;

    return (
        <pre className="text-xs font-mono leading-relaxed">
            <table className="border-collapse w-full">
                <tbody>
                    {lines.map((line, i) => (
                        <tr key={i} className="hover:bg-secondary/30">
                            <td
                                className="text-right select-none text-muted-foreground/40 pr-3 pl-3 align-top"
                                style={{ minWidth: `${gutterWidth + 2}ch` }}
                            >
                                {i + 1}
                            </td>
                            <td className="whitespace-pre-wrap break-words text-foreground/90 pr-4">
                                {line || "\n"}
                            </td>
                        </tr>
                    ))}
                </tbody>
            </table>
        </pre>
    );
}

/** Binary fallback with client-side download from base64 content */
function BinaryDownloadFallback({
    filename,
    content,
    t,
    className,
    message,
}: {
    filename: string;
    content: string;
    t: ReturnType<typeof useTranslations>;
    className?: string;
    message?: string;
}) {
    const handleDownload = useCallback(() => {
        downloadBase64(content, filename, getMimeType(filename));
    }, [content, filename]);

    return (
        <ComputerEmptyState
            icon={FileText}
            title={filename}
            subtitle={message || t("workspace.noPreview")}
            className={cn("h-full", className)}
        >
            <Button variant="outline" size="sm" onClick={handleDownload} className="mt-4">
                <Download className="w-4 h-4 mr-2" />
                {t("workspace.download")}
            </Button>
        </ComputerEmptyState>
    );
}

/** PDF preview using iframe with blob URL */
function PdfPreview({
    content,
    filename,
    onDownload,
    className,
}: {
    content: string;
    filename: string;
    onDownload?: () => void;
    className?: string;
}) {
    const [blobUrl, setBlobUrl] = useState("");

    useEffect(() => {
        const byteChars = atob(content);
        const byteArray = new Uint8Array(byteChars.length);
        for (let i = 0; i < byteChars.length; i++) {
            byteArray[i] = byteChars.charCodeAt(i);
        }
        const blob = new Blob([byteArray], { type: "application/pdf" });
        const url = URL.createObjectURL(blob);
        // Use a microtask to avoid synchronous setState warning
        queueMicrotask(() => setBlobUrl(url));
        return () => URL.revokeObjectURL(url);
    }, [content]);

    if (!blobUrl) {
        return (
            <div className={cn("flex items-center justify-center h-full", className)}>
                <Loader2 className="w-6 h-6 animate-spin text-muted-foreground" />
            </div>
        );
    }

    return (
        <div className={cn("flex flex-col h-full", className)}>
            <ContentHeader
                filename={filename}
                language={null}
                content={content}
                isBinary
                onDownload={onDownload}
            />
            <iframe
                src={`${blobUrl}#view=FitH&toolbar=0`}
                className="flex-1 w-full border-0"
                title={filename}
            />
        </div>
    );
}

/** Fullscreen modal for images */
function FullscreenModal({
    src,
    filename,
    onClose,
    onDownload,
}: {
    src: string;
    filename: string;
    onClose: () => void;
    onDownload: () => void;
}) {
    return (
        <div
            className="fixed inset-0 z-50 bg-background/98 animate-in fade-in duration-200"
            onClick={onClose}
        >
            <div className="h-full flex flex-col p-4">
                <div className="flex items-center justify-between mb-4">
                    <span className="text-sm font-medium truncate">{filename}</span>
                    <div className="flex items-center gap-2">
                        <button
                            onClick={(e) => {
                                e.stopPropagation();
                                onDownload();
                            }}
                            className="flex items-center gap-2 px-3 py-1.5 text-sm rounded-lg bg-secondary/60 hover:bg-secondary transition-colors focus-visible:ring-2 focus-visible:ring-primary focus-visible:outline-none"
                        >
                            <Download className="w-4 h-4" />
                        </button>
                        <button
                            onClick={onClose}
                            className="flex items-center gap-2 px-3 py-1.5 text-sm rounded-lg bg-muted hover:bg-muted/80 transition-colors focus-visible:ring-2 focus-visible:ring-primary focus-visible:outline-none"
                        >
                            <X className="w-4 h-4" />
                        </button>
                    </div>
                </div>
                <div
                    className="flex-1 overflow-auto flex items-center justify-center"
                    onClick={(e) => e.stopPropagation()}
                >
                    <img
                        src={src}
                        alt={filename}
                        className="max-w-none h-auto"
                        style={{ maxHeight: "calc(100vh - 100px)" }}
                    />
                </div>
            </div>
        </div>
    );
}
