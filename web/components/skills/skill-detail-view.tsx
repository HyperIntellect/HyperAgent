"use client";

import { useEffect, useState, useCallback } from "react";
import { useTranslations } from "next-intl";
import Link from "next/link";
import { getSkill, updateSkill } from "@/lib/api/skills";
import type { SkillDetailMetadata } from "@/lib/types/skills";
import { SkillCodeEditor } from "@/components/editor";
import { Button } from "@/components/ui/button";
import {
  ArrowLeft,
  Code,
  Loader2,
  AlertCircle,
  CheckCircle2,
  XCircle,
  Pencil,
  X,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { SKILL_ICONS, SKILL_ACCENT, CATEGORY_ICONS, CATEGORY_ACCENT, DEFAULT_ACCENT } from "@/lib/utils/skill-categories";
import { getTranslatedSkillName, getTranslatedSkillDescription, getTranslatedCategory } from "@/lib/utils/skill-i18n";

interface SkillDetailViewProps {
  skillId: string;
}

export function SkillDetailView({ skillId }: SkillDetailViewProps) {
  const t = useTranslations("skills");
  const [skill, setSkill] = useState<SkillDetailMetadata | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Edit mode state
  const [isEditing, setIsEditing] = useState(false);
  const [editedCode, setEditedCode] = useState("");
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);

  const loadSkill = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await getSkill(skillId);
      setSkill(data);
    } catch (err) {
      console.error("Failed to load skill:", err);
      setError(t("loadError"));
    } finally {
      setLoading(false);
    }
  }, [skillId, t]);

  useEffect(() => {
    loadSkill();
  }, [loadSkill]);

  function startEditing() {
    if (skill?.source_code) {
      setEditedCode(skill.source_code);
    }
    setSaveError(null);
    setIsEditing(true);
  }

  function cancelEditing() {
    setIsEditing(false);
    setEditedCode("");
    setSaveError(null);
  }

  async function handleSave() {
    if (!skill) return;

    try {
      setSaving(true);
      setSaveError(null);
      const updated = await updateSkill(skill.id, {
        source_code: editedCode,
      });
      setSkill(updated);
      setIsEditing(false);
      setEditedCode("");
    } catch (err) {
      console.error("Failed to save skill:", err);
      setSaveError(
        err instanceof Error ? err.message : t("saveError")
      );
    } finally {
      setSaving(false);
    }
  }

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center py-24 gap-3">
        <Loader2 className="w-5 h-5 animate-spin text-muted-foreground" />
        <p className="text-sm text-muted-foreground">{t("loading")}</p>
      </div>
    );
  }

  if (error || !skill) {
    return (
      <div className="flex flex-col items-center justify-center py-24 gap-3">
        <div className="w-10 h-10 rounded-full bg-destructive/10 flex items-center justify-center">
          <AlertCircle className="w-5 h-5 text-destructive" />
        </div>
        <p className="text-sm text-muted-foreground">{error || t("notFound")}</p>
        <Link
          href="/skills"
          className="text-sm text-foreground font-medium underline underline-offset-4 hover:text-foreground/80"
        >
          {t("backToSkills")}
        </Link>
      </div>
    );
  }

  const Icon =
    SKILL_ICONS[skill.id] || CATEGORY_ICONS[skill.category as keyof typeof CATEGORY_ICONS] || Code;
  const accent =
    SKILL_ACCENT[skill.id] || CATEGORY_ACCENT[skill.category as keyof typeof CATEGORY_ACCENT] || DEFAULT_ACCENT;

  return (
    <div className="space-y-8">
      {/* Back link */}
      <Link
        href="/skills"
        className="inline-flex items-center gap-1.5 text-sm text-muted-foreground hover:text-foreground transition-colors"
      >
        <ArrowLeft className="w-3.5 h-3.5" />
        {t("backToSkills")}
      </Link>

      {/* Header */}
      <div className="flex items-start gap-4">
        <div className={cn("shrink-0 p-3 rounded-xl", accent.bg)}>
          <Icon className={cn("w-6 h-6", accent.icon)} />
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-3 flex-wrap">
            <h1 className="text-2xl font-bold tracking-tight">
              {getTranslatedSkillName(skill.id, skill.name, t)}
            </h1>
            <span className="text-xs text-muted-foreground/60 font-mono bg-secondary px-2 py-0.5 rounded">
              v{skill.version}
            </span>
            <span
              className={cn(
                "inline-flex items-center gap-1 text-xs font-medium px-2 py-0.5 rounded",
                skill.enabled
                  ? "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400"
                  : "bg-red-500/10 text-red-600 dark:text-red-400"
              )}
            >
              {skill.enabled ? (
                <CheckCircle2 className="w-3 h-3" />
              ) : (
                <XCircle className="w-3 h-3" />
              )}
              {skill.enabled ? t("enabled") : t("disabled")}
            </span>
            {skill.is_builtin && (
              <span className="text-xs font-medium px-2 py-0.5 rounded bg-secondary text-muted-foreground">
                {t("builtinSkill")}
              </span>
            )}
          </div>
          <p className="text-muted-foreground mt-2 leading-relaxed">
            {getTranslatedSkillDescription(skill.id, skill.description, t)}
          </p>
        </div>
      </div>

      {/* Metadata row */}
      <div className="flex items-center gap-4 flex-wrap text-sm">
        <div className="flex items-center gap-1.5">
          <span className="text-muted-foreground">{t("category")}:</span>
          <span className="font-medium uppercase tracking-wider text-xs">
            {getTranslatedCategory(skill.category, t)}
          </span>
        </div>
        {skill.tags && skill.tags.length > 0 && (
          <div className="flex items-center gap-1.5">
            <span className="text-muted-foreground">{t("tags")}:</span>
            <div className="flex gap-1">
              {skill.tags.map((tag) => (
                <span
                  key={tag}
                  className="text-xs font-medium px-1.5 py-0.5 rounded bg-secondary text-muted-foreground"
                >
                  {tag}
                </span>
              ))}
            </div>
          </div>
        )}
        {skill.author && (
          <div className="flex items-center gap-1.5">
            <span className="text-muted-foreground">{t("author")}:</span>
            <span className="text-xs font-medium">{skill.author}</span>
          </div>
        )}
      </div>

      {/* Source Code section */}
      {!skill.is_builtin && skill.source_code && (
        <section>
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-sm font-semibold text-foreground uppercase tracking-wide">
              {t("sourceCode")}
            </h2>
            {!isEditing ? (
              <Button
                variant="ghost"
                size="sm"
                onClick={startEditing}
                className="h-7 text-xs gap-1.5 cursor-pointer"
              >
                <Pencil className="w-3 h-3" />
                {t("edit")}
              </Button>
            ) : (
              <Button
                variant="ghost"
                size="sm"
                onClick={cancelEditing}
                className="h-7 text-xs gap-1.5 cursor-pointer"
              >
                <X className="w-3 h-3" />
                {t("cancel")}
              </Button>
            )}
          </div>

          <SkillCodeEditor
            value={isEditing ? editedCode : skill.source_code}
            onChange={isEditing ? setEditedCode : undefined}
            readOnly={!isEditing}
            height="400px"
          />

          {/* Save controls */}
          {isEditing && (
            <div className="mt-4 flex items-center gap-3">
              <Button
                onClick={handleSave}
                disabled={saving}
                className="gap-2 cursor-pointer"
                size="sm"
              >
                {saving && <Loader2 className="w-3.5 h-3.5 animate-spin" />}
                {saving ? t("saving") : t("saveChanges")}
              </Button>
              {saveError && (
                <p className="text-sm text-destructive">{saveError}</p>
              )}
            </div>
          )}
        </section>
      )}

      {/* Parameters section */}
      <section>
        <h2 className="text-sm font-semibold text-foreground uppercase tracking-wide mb-4">
          {t("parameters", { count: skill.parameters?.length ?? 0 })}
        </h2>
        {skill.parameters && skill.parameters.length > 0 ? (
          <div className="border border-border/50 rounded-xl overflow-hidden">
            <table className="w-full text-sm">
              <thead>
                <tr className="bg-secondary/50">
                  <th className="text-left px-4 py-2.5 font-medium text-muted-foreground">
                    {t("paramName")}
                  </th>
                  <th className="text-left px-4 py-2.5 font-medium text-muted-foreground">
                    {t("paramType")}
                  </th>
                  <th className="text-left px-4 py-2.5 font-medium text-muted-foreground">
                    {t("paramRequired")}
                  </th>
                  <th className="text-left px-4 py-2.5 font-medium text-muted-foreground">
                    {t("paramDefault")}
                  </th>
                  <th className="text-left px-4 py-2.5 font-medium text-muted-foreground">
                    {t("paramDescription")}
                  </th>
                </tr>
              </thead>
              <tbody>
                {skill.parameters.map((param, i) => (
                  <tr
                    key={param.name}
                    className={cn(
                      "border-t border-border/30",
                      i % 2 === 0 ? "" : "bg-secondary/20"
                    )}
                  >
                    <td className="px-4 py-2.5 font-mono text-xs">{param.name}</td>
                    <td className="px-4 py-2.5 font-mono text-xs text-muted-foreground">
                      {param.type}
                    </td>
                    <td className="px-4 py-2.5">
                      {param.required ? (
                        <CheckCircle2 className="w-3.5 h-3.5 text-emerald-500" />
                      ) : (
                        <span className="text-xs text-muted-foreground/50">-</span>
                      )}
                    </td>
                    <td className="px-4 py-2.5 font-mono text-xs text-muted-foreground">
                      {param.default !== undefined && param.default !== null
                        ? String(param.default)
                        : "-"}
                    </td>
                    <td className="px-4 py-2.5 text-xs text-muted-foreground">
                      {param.description}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="text-sm text-muted-foreground/60">{t("noParameters")}</p>
        )}
      </section>

      {/* Output Schema section */}
      {skill.output_schema && Object.keys(skill.output_schema).length > 0 && (
        <section>
          <h2 className="text-sm font-semibold text-foreground uppercase tracking-wide mb-4">
            {t("outputSchema")}
          </h2>
          <pre className="bg-secondary/50 border border-border/50 rounded-xl p-4 text-xs font-mono overflow-x-auto">
            {JSON.stringify(skill.output_schema, null, 2)}
          </pre>
        </section>
      )}
    </div>
  );
}
