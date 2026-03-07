"use client";

import { useEffect } from "react";
import { useTranslations } from "next-intl";
import { Zap, Sparkles, Crown } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { useSettingsStore } from "@/lib/stores/settings-store";

const TIER_CONFIG = [
  { id: "max" as const, icon: Crown },
  { id: "pro" as const, icon: Sparkles },
  { id: "lite" as const, icon: Zap },
] as const;

export function ModelSection() {
  const t = useTranslations("settings");
  const {
    provider,
    setProvider,
    tier,
    setTier,
    availableProviders,
    providersLoaded,
    loadProviders,
  } = useSettingsStore();

  useEffect(() => {
    if (!providersLoaded) {
      loadProviders();
    }
  }, [providersLoaded, loadProviders]);

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-base font-semibold text-foreground">
          {t("model.title")}
        </h2>
        <p className="text-sm text-muted-foreground mt-1">
          {t("model.description")}
        </p>
      </div>

      {/* Provider */}
      {availableProviders.length > 1 && (
        <div className="space-y-3">
          <label className="text-sm font-medium text-foreground">
            {t("model.provider")}
          </label>
          <div className="flex flex-wrap gap-2">
            {availableProviders.map((p) => (
              <Button
                key={p.id}
                variant="ghost"
                onClick={() => setProvider(p.id)}
                className={cn(
                  "flex-1 min-w-0 h-10",
                  provider === p.id
                    ? "bg-secondary text-foreground border border-foreground/15 font-medium"
                    : "bg-secondary/50 text-muted-foreground hover:text-foreground hover:bg-secondary"
                )}
              >
                <span className="truncate">{p.name}</span>
              </Button>
            ))}
          </div>
        </div>
      )}

      {/* Quality Tier */}
      <div className="space-y-3">
        <label className="text-sm font-medium text-foreground">
          {t("model.tier")}
        </label>
        <div className="space-y-2">
          {TIER_CONFIG.map(({ id: t_tier, icon: TierIcon }) => (
            <Button
              key={t_tier}
              variant="ghost"
              onClick={() => setTier(t_tier)}
              className={cn(
                "w-full h-auto px-4 py-3 justify-start gap-3 text-left",
                tier === t_tier
                  ? "bg-primary text-primary-foreground"
                  : "bg-secondary/50 text-muted-foreground hover:text-foreground hover:bg-secondary"
              )}
            >
              <TierIcon className="w-4 h-4 shrink-0 mt-0.5" />
              <div className="min-w-0">
                <div className="text-sm font-medium">
                  {t_tier.charAt(0).toUpperCase() + t_tier.slice(1)}
                </div>
                <div
                  className={cn(
                    "text-xs mt-0.5 font-normal",
                    tier === t_tier
                      ? "text-primary-foreground/70"
                      : "text-muted-foreground"
                  )}
                >
                  {t(`model.tier${t_tier.charAt(0).toUpperCase() + t_tier.slice(1)}` as
                    "model.tierMax" | "model.tierPro" | "model.tierLite")}
                </div>
              </div>
            </Button>
          ))}
        </div>
      </div>
    </div>
  );
}
