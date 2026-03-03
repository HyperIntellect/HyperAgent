"use client";

import { useState, useRef, useEffect } from "react";
import { useRouter } from "next/navigation";
import { useTranslations, useLocale } from "next-intl";
import { Sun, Moon, Globe, Monitor, ChevronUp, Cpu, Sparkles } from "lucide-react";
import { cn } from "@/lib/utils";
import { useSettingsStore } from "@/lib/stores/settings-store";

interface PreferencesPanelProps {
  theme: "light" | "dark" | "auto";
  mounted: boolean;
  onThemeChange: (theme: "light" | "dark" | "auto") => void;
  className?: string;
}

const LOCALES = [
  { code: "en", label: "English", shortLabel: "EN" },
  { code: "zh-CN", label: "简体中文", shortLabel: "中文" },
] as const;

export function PreferencesPanel({
  theme,
  mounted,
  onThemeChange,
  className,
}: PreferencesPanelProps) {
  const t = useTranslations("sidebar");
  const locale = useLocale();
  const router = useRouter();
  const [isOpen, setIsOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  const {
    provider,
    setProvider,
    tier,
    setTier,
    availableProviders,
    providersLoaded,
    loadProviders,
  } = useSettingsStore();
  const tSettings = useTranslations("settings");

  // Load providers on mount
  useEffect(() => {
    if (!providersLoaded) {
      loadProviders();
    }
  }, [providersLoaded, loadProviders]);

  const handleLocaleChange = (newLocale: string) => {
    document.cookie = `locale=${newLocale};path=/;max-age=31536000`;
    setIsOpen(false);
    router.refresh();
  };

  const handleThemeChange = (newTheme: "light" | "dark" | "auto") => {
    onThemeChange(newTheme);
  };

  const currentLocale = LOCALES.find((l) => l.code === locale) || LOCALES[0];

  // Get icon for current theme
  const ThemeIcon = theme === "auto" ? Monitor : theme === "dark" ? Moon : Sun;

  // Close menu when clicking outside
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    }

    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  return (
    <div ref={menuRef} className={cn("relative", className)}>
      {/* Trigger */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className={cn(
          "w-full h-9 px-3 rounded-lg",
          "flex items-center justify-between",
          "text-base",
          "text-muted-foreground hover:text-foreground",
          "transition-colors",
          "cursor-pointer",
          "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
        )}
      >
        <div className="flex items-center gap-2">
          {mounted && <ThemeIcon className="w-4 h-4" />}
          <span>{t("preferences")}</span>
        </div>
        <ChevronUp className={cn(
          "w-4 h-4",
          isOpen ? "rotate-0" : "rotate-180"
        )} />
      </button>

      {/* Menu */}
      {isOpen && (
        <div
          className={cn(
            "absolute bottom-full left-0 right-0 mb-1 z-50",
            "bg-card border border-border rounded-lg",
            "overflow-hidden"
          )}
        >
          {/* Provider */}
          {availableProviders.length > 1 && (
            <div className="p-2.5 border-b border-border">
              <div className="px-1 py-1 mb-1">
                <span className="text-xs font-medium text-muted-foreground">
                  {t("provider")}
                </span>
              </div>
              <div className="flex flex-wrap gap-1">
                {availableProviders.map((p) => (
                  <button
                    key={p.id}
                    onClick={() => setProvider(p.id)}
                    className={cn(
                      "flex-1 min-w-0 h-9 px-2 rounded-lg flex items-center justify-center gap-1.5",
                      "text-xs font-medium transition-colors",
                      "cursor-pointer",
                      "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary",
                      provider === p.id
                        ? "bg-primary text-primary-foreground"
                        : "bg-secondary text-muted-foreground hover:text-foreground"
                    )}
                  >
                    <Cpu className="w-3.5 h-3.5 shrink-0" />
                    <span className="truncate">{p.name}</span>
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Model Tier */}
          <div className="p-2.5 border-b border-border">
            <div className="px-1 py-1 mb-1">
              <span className="text-xs font-medium text-muted-foreground">
                {tSettings("modelTier")}
              </span>
            </div>
            <div className="flex flex-wrap gap-1">
              {(["max", "pro", "lite"] as const).map((t) => (
                <button
                  key={t}
                  onClick={() => setTier(t)}
                  className={cn(
                    "flex-1 min-w-0 h-9 px-2 rounded-lg flex items-center justify-center gap-1.5",
                    "text-xs font-medium transition-colors",
                    "cursor-pointer",
                    "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary",
                    tier === t
                      ? "bg-primary text-primary-foreground"
                      : "bg-secondary text-muted-foreground hover:text-foreground"
                  )}
                >
                  <Sparkles className="w-3.5 h-3.5 shrink-0" />
                  <span className="truncate">
                    {t.charAt(0).toUpperCase() + t.slice(1)}
                  </span>
                </button>
              ))}
            </div>
          </div>

          {/* Theme */}
          <div className="p-2.5 border-b border-border">
            <div className="px-1 py-1 mb-1">
              <span className="text-xs font-medium text-muted-foreground">
                {t("theme")}
              </span>
            </div>
            <div className="flex gap-1">
              {[
                { value: "auto", icon: Monitor, label: t("auto") },
                { value: "light", icon: Sun, label: t("light") },
                { value: "dark", icon: Moon, label: t("dark") },
              ].map(({ value, icon: Icon, label }) => (
                <button
                  key={value}
                  onClick={() => handleThemeChange(value as "light" | "dark" | "auto")}
                  className={cn(
                    "flex-1 h-9 rounded-lg flex items-center justify-center gap-1.5",
                    "text-xs font-medium transition-colors",
                    "cursor-pointer",
                    "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary",
                    mounted && theme === value
                      ? "bg-primary text-primary-foreground"
                      : "bg-secondary text-muted-foreground hover:text-foreground"
                  )}
                >
                  <Icon className="w-3.5 h-3.5" />
                  <span>{label}</span>
                </button>
              ))}
            </div>
          </div>

          {/* Language */}
          <div className="p-2.5">
            <div className="px-1 py-1 mb-1">
              <span className="text-xs font-medium text-muted-foreground">
                {t("language")}
              </span>
            </div>
            <div className="flex gap-1">
              {LOCALES.map((loc) => (
                <button
                  key={loc.code}
                  onClick={() => handleLocaleChange(loc.code)}
                  className={cn(
                    "flex-1 h-9 rounded-lg flex items-center justify-center gap-1.5",
                    "text-xs font-medium transition-colors",
                    "cursor-pointer",
                    "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary",
                    currentLocale.code === loc.code
                      ? "bg-primary text-primary-foreground"
                      : "bg-secondary text-muted-foreground hover:text-foreground"
                  )}
                >
                  <Globe className="w-3.5 h-3.5" />
                  <span>{loc.shortLabel}</span>
                </button>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
