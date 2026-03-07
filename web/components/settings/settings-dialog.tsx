"use client";

import { useEffect, useCallback, type ComponentType } from "react";
import { useTranslations } from "next-intl";
import { X, Settings, Cpu, Brain } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import {
  useSettingsDialogStore,
  type SettingsSection,
} from "@/lib/stores/settings-dialog-store";
import { GeneralSection } from "./general-section";
import { ModelSection } from "./model-section";
import { MemorySection } from "./memory-section";

interface SectionDef {
  id: SettingsSection;
  icon: ComponentType<{ className?: string }>;
  component: ComponentType;
}

const SECTIONS: SectionDef[] = [
  { id: "general", icon: Settings, component: GeneralSection },
  { id: "model", icon: Cpu, component: ModelSection },
  { id: "memory", icon: Brain, component: MemorySection },
];

export function SettingsDialog() {
  const { isOpen, activeSection, closeSettings, setActiveSection } =
    useSettingsDialogStore();
  const t = useTranslations("settings");

  const handleEscape = useCallback(
    (e: KeyboardEvent) => {
      if (e.key === "Escape" && isOpen) {
        closeSettings();
      }
    },
    [isOpen, closeSettings]
  );

  useEffect(() => {
    document.addEventListener("keydown", handleEscape);
    return () => document.removeEventListener("keydown", handleEscape);
  }, [handleEscape]);

  // Lock body scroll when dialog is open
  useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = "hidden";
    } else {
      document.body.style.overflow = "";
    }
    return () => {
      document.body.style.overflow = "";
    };
  }, [isOpen]);

  if (!isOpen) return null;

  const activeDef = SECTIONS.find((s) => s.id === activeSection) || SECTIONS[0];
  const ActiveComponent = activeDef.component;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/40"
        onClick={closeSettings}
      />

      {/* Dialog */}
      <div
        className={cn(
          "relative w-full max-w-2xl max-h-[85vh]",
          "bg-card border border-border rounded-xl",
          "overflow-hidden animate-scale-in",
          "flex flex-col",
          "mx-4"
        )}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-border">
          <h1 className="text-lg font-semibold text-foreground">
            {t("title")}
          </h1>
          <Button
            variant="ghost"
            size="icon"
            onClick={closeSettings}
            className="w-8 h-8 text-muted-foreground hover:text-foreground"
          >
            <X className="w-4 h-4" />
          </Button>
        </div>

        {/* Body: nav + content */}
        <div className="flex flex-1 min-h-0">
          {/* Left nav - hidden on mobile, replaced by tabs */}
          <nav className="hidden md:flex flex-col w-44 shrink-0 border-r border-border py-3 px-2 gap-0.5">
            {SECTIONS.map(({ id, icon: Icon }) => (
              <Button
                key={id}
                variant="ghost"
                onClick={() => setActiveSection(id)}
                className={cn(
                  "w-full h-9 px-3 justify-start gap-2.5",
                  activeSection === id
                    ? "bg-secondary text-foreground font-medium"
                    : "text-muted-foreground hover:text-foreground hover:bg-secondary/50"
                )}
              >
                <Icon className="w-4 h-4 shrink-0" />
                <span>{t(`sections.${id}`)}</span>
              </Button>
            ))}
          </nav>

          {/* Mobile tabs */}
          <div className="md:hidden flex border-b border-border px-4 pt-2 gap-1 w-full">
            {SECTIONS.map(({ id, icon: Icon }) => (
              <Button
                key={id}
                variant="ghost"
                onClick={() => setActiveSection(id)}
                className={cn(
                  "flex-1 h-9 text-xs gap-1.5",
                  activeSection === id
                    ? "bg-secondary text-foreground font-medium"
                    : "text-muted-foreground hover:text-foreground hover:bg-secondary/50"
                )}
              >
                <Icon className="w-3.5 h-3.5 shrink-0" />
                <span>{t(`sections.${id}`)}</span>
              </Button>
            ))}
          </div>

          {/* Content */}
          <div className="flex-1 overflow-y-auto p-6 md:p-8">
            <ActiveComponent />
          </div>
        </div>
      </div>
    </div>
  );
}
