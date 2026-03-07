"use client";

import { useRouter } from "next/navigation";
import { useTranslations, useLocale } from "next-intl";
import { Sun, Moon, Monitor, Globe } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { useTheme } from "@/lib/hooks/use-theme";

const LOCALES = [
  { code: "en", label: "English", shortLabel: "EN" },
  { code: "zh-CN", label: "简体中文", shortLabel: "中文" },
] as const;

const THEME_OPTIONS = [
  { value: "auto" as const, icon: Monitor },
  { value: "light" as const, icon: Sun },
  { value: "dark" as const, icon: Moon },
] as const;

const THEME_PREVIEW_COLORS: Record<string, { bg: string; fg: string; accent: string }> = {
  auto: { bg: "bg-gradient-to-r from-stone-100 to-stone-800", fg: "bg-stone-500", accent: "bg-indigo-500" },
  light: { bg: "bg-stone-100", fg: "bg-stone-800", accent: "bg-indigo-500" },
  dark: { bg: "bg-stone-800", fg: "bg-stone-200", accent: "bg-indigo-400" },
};

export function GeneralSection() {
  const t = useTranslations("settings");
  const { theme, setTheme, mounted } = useTheme();
  const locale = useLocale();
  const router = useRouter();

  const currentLocale = LOCALES.find((l) => l.code === locale) || LOCALES[0];

  const handleLocaleChange = (newLocale: string) => {
    // eslint-disable-next-line react-hooks/immutability
    document.cookie = `locale=${newLocale};path=/;max-age=31536000`;
    router.refresh();
  };

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-base font-semibold text-foreground">
          {t("general.title")}
        </h2>
        <p className="text-sm text-muted-foreground mt-1">
          {t("general.description")}
        </p>
      </div>

      {/* Theme */}
      <div className="space-y-3">
        <label className="text-sm font-medium text-foreground">
          {t("general.theme")}
        </label>
        <div className="grid grid-cols-3 gap-2">
          {THEME_OPTIONS.map(({ value, icon: Icon }) => {
            const isActive = mounted && theme === value;
            const preview = THEME_PREVIEW_COLORS[value];
            return (
              <Button
                key={value}
                variant="ghost"
                onClick={() => setTheme(value)}
                className={cn(
                  "h-auto flex-col gap-2 py-3 rounded-lg",
                  isActive
                    ? "bg-secondary text-foreground border border-foreground/15"
                    : "bg-secondary/50 text-muted-foreground hover:text-foreground hover:bg-secondary"
                )}
              >
                {/* Mini theme preview */}
                <div className={cn("w-full h-8 rounded-md overflow-hidden relative", preview.bg)}>
                  <div className={cn("absolute bottom-1 left-1.5 h-1 w-4 rounded-full", preview.fg)} />
                  <div className={cn("absolute bottom-1 right-1.5 h-1 w-2 rounded-full", preview.accent)} />
                </div>
                <div className="flex items-center gap-1.5">
                  <Icon className="w-3.5 h-3.5" />
                  <span className="text-xs font-medium">
                    {t(`general.theme${value.charAt(0).toUpperCase() + value.slice(1)}` as
                      "general.themeAuto" | "general.themeLight" | "general.themeDark")}
                  </span>
                </div>
              </Button>
            );
          })}
        </div>
      </div>

      {/* Language */}
      <div className="space-y-3">
        <label className="text-sm font-medium text-foreground">
          {t("general.language")}
        </label>
        <div className="grid grid-cols-2 gap-2">
          {LOCALES.map((loc) => (
            <Button
              key={loc.code}
              variant="ghost"
              onClick={() => handleLocaleChange(loc.code)}
              className={cn(
                "h-10 gap-2",
                currentLocale.code === loc.code
                  ? "bg-secondary text-foreground border border-foreground/15 font-medium"
                  : "bg-secondary/50 text-muted-foreground hover:text-foreground hover:bg-secondary"
              )}
            >
              <Globe className="w-4 h-4" />
              <span>{loc.label}</span>
            </Button>
          ))}
        </div>
      </div>
    </div>
  );
}
