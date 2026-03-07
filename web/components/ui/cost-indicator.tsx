"use client";

import { useState, useMemo, useRef, useEffect } from "react";
import { Coins, ChevronDown, ChevronUp, Zap } from "lucide-react";
import { useTranslations } from "next-intl";
import { getModelDisplayName } from "@/lib/utils/model-display";
import type { AgentEvent } from "@/lib/types";

interface CostIndicatorProps {
    events: AgentEvent[];
    variant?: "compact" | "badge";
}

export function CostIndicator({ events, variant = "compact" }: CostIndicatorProps) {
    const [expanded, setExpanded] = useState(false);
    const containerRef = useRef<HTMLDivElement>(null);
    const t = useTranslations("chat");

    const usage = useMemo(() => {
        const usageEvents = events.filter((e) => e.type === "usage");

        let totalInput = 0;
        let totalOutput = 0;
        let totalCached = 0;
        let totalCost = 0;
        let lastModel: string | undefined;
        const byTier: Record<string, { input: number; output: number; cost: number }> = {};

        for (const ev of usageEvents) {
            totalInput += ev.input_tokens || 0;
            totalOutput += ev.output_tokens || 0;
            totalCached += ev.cached_tokens || 0;
            totalCost += ev.cost_usd || 0;
            if (ev.model) lastModel = ev.model;

            const tier = ev.tier || "unknown";
            if (!byTier[tier]) byTier[tier] = { input: 0, output: 0, cost: 0 };
            byTier[tier].input += ev.input_tokens || 0;
            byTier[tier].output += ev.output_tokens || 0;
            byTier[tier].cost += ev.cost_usd || 0;
        }

        return { totalInput, totalOutput, totalCached, totalCost, byTier, count: usageEvents.length, lastModel };
    }, [events]);

    // Close expanded panel when clicking outside
    useEffect(() => {
        if (!expanded) return;
        const handleClickOutside = (event: MouseEvent) => {
            if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
                setExpanded(false);
            }
        };
        document.addEventListener("mousedown", handleClickOutside);
        return () => document.removeEventListener("mousedown", handleClickOutside);
    }, [expanded]);

    if (usage.count === 0) return null;

    const formatTokens = (n: number) => {
        if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
        if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
        return String(n);
    };

    const formatCost = (n: number) => {
        if (n < 0.001) return "<$0.001";
        if (n < 0.01) return `$${n.toFixed(3)}`;
        return `$${n.toFixed(2)}`;
    };

    const modelName = getModelDisplayName(usage.lastModel);
    const totalTokens = formatTokens(usage.totalInput + usage.totalOutput);

    const expandedPopup = expanded && (
        <div className="absolute bottom-full left-0 mb-1 bg-popover border rounded-lg shadow-lg p-3 text-xs min-w-[240px] z-50">
            <div className="space-y-2">
                <div className="flex justify-between text-muted-foreground">
                    <span>{t("inputTokens")}</span>
                    <span>{formatTokens(usage.totalInput)}</span>
                </div>
                <div className="flex justify-between text-muted-foreground">
                    <span>{t("outputTokens")}</span>
                    <span>{formatTokens(usage.totalOutput)}</span>
                </div>
                {usage.totalCached > 0 && (
                    <div className="flex justify-between text-muted-foreground">
                        <span>{t("cachedTokens")}</span>
                        <span>{formatTokens(usage.totalCached)}</span>
                    </div>
                )}
                <div className="border-t pt-2 flex justify-between font-medium">
                    <span>{t("totalCost")}</span>
                    <span>{formatCost(usage.totalCost)}</span>
                </div>

                {Object.keys(usage.byTier).length > 1 && (
                    <>
                        <div className="border-t pt-2 text-muted-foreground font-medium">
                            {t("byTier")}
                        </div>
                        {Object.entries(usage.byTier).map(([tier, data]) => (
                            <div key={tier} className="flex justify-between text-muted-foreground">
                                <span className="capitalize">{tier}</span>
                                <span>{formatCost(data.cost)}</span>
                            </div>
                        ))}
                    </>
                )}
            </div>
        </div>
    );

    if (variant === "badge") {
        return (
            <div ref={containerRef} className="relative inline-flex items-center">
                <button
                    onClick={() => setExpanded(!expanded)}
                    className="inline-flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition-colors rounded-full bg-secondary/60 hover:bg-secondary px-2.5 py-1"
                >
                    <Zap className="w-3 h-3" />
                    {modelName && (
                        <>
                            <span className="font-medium">{modelName}</span>
                            <span className="text-muted-foreground/50">&middot;</span>
                        </>
                    )}
                    <span>{totalTokens} tokens</span>
                    <span className="text-muted-foreground/50">&middot;</span>
                    <span>{formatCost(usage.totalCost)}</span>
                    {expanded ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
                </button>

                {expandedPopup}
            </div>
        );
    }

    return (
        <div ref={containerRef} className="relative inline-flex items-center">
            <button
                onClick={() => setExpanded(!expanded)}
                className="inline-flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors rounded-md px-1.5 py-0.5"
            >
                <Coins className="w-3 h-3" />
                <span>{totalTokens} tokens</span>
                <span className="text-muted-foreground/60">&middot;</span>
                <span>{formatCost(usage.totalCost)}</span>
                {expanded ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
            </button>

            {expandedPopup}
        </div>
    );
}
