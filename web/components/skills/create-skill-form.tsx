"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { useTranslations } from "next-intl";
import { createSkill } from "@/lib/api/skills";
import type { SkillParameter } from "@/lib/types/skills";
import { SKILL_TEMPLATES, type SkillTemplate } from "./skill-templates";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import {
  ArrowLeft,
  Plus,
  X,
  Loader2,
  Code,
  Sparkles,
  Search,
  BarChart3,
  FileText,
  CheckCircle2,
} from "lucide-react";
import { cn } from "@/lib/utils";
import Link from "next/link";
import { SkillCodeEditor } from "@/components/editor";

const CATEGORIES = [
  { id: "research", icon: Search },
  { id: "code", icon: Code },
  { id: "data", icon: BarChart3 },
  { id: "creative", icon: Sparkles },
  { id: "automation", icon: FileText },
] as const;

const PARAM_TYPES = ["string", "number", "boolean", "object", "array"] as const;

interface ParameterEntry {
  name: string;
  type: string;
  description: string;
  required: boolean;
}

export function CreateSkillForm() {
  const t = useTranslations("skills");
  const router = useRouter();

  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [category, setCategory] = useState("automation");
  const [version, setVersion] = useState("1.0.0");
  const [tagsInput, setTagsInput] = useState("");
  const [parameters, setParameters] = useState<ParameterEntry[]>([]);
  const [selectedTemplate, setSelectedTemplate] = useState<SkillTemplate | null>(null);
  const [sourceCode, setSourceCode] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function addParameter() {
    setParameters([
      ...parameters,
      { name: "", type: "string", description: "", required: false },
    ]);
  }

  function removeParameter(index: number) {
    setParameters(parameters.filter((_, i) => i !== index));
  }

  function updateParameter(index: number, field: keyof ParameterEntry, value: string | boolean) {
    const updated = [...parameters];
    updated[index] = { ...updated[index], [field]: value };
    setParameters(updated);
  }

  function selectTemplate(template: SkillTemplate) {
    setSelectedTemplate(template);
    setSourceCode(template.source_code);
    if (!category || category === "automation") {
      setCategory(template.category);
    }
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);

    if (!name.trim() || !description.trim() || !selectedTemplate) {
      setError(
        !selectedTemplate
          ? t("selectTemplate")
          : t("createError")
      );
      return;
    }

    const tags = tagsInput
      .split(",")
      .map((t) => t.trim())
      .filter(Boolean);

    const skillParams: SkillParameter[] = parameters
      .filter((p) => p.name.trim())
      .map((p) => ({
        name: p.name.trim(),
        type: p.type as SkillParameter["type"],
        description: p.description,
        required: p.required,
      }));

    try {
      setSubmitting(true);
      const result = await createSkill({
        name: name.trim(),
        description: description.trim(),
        category,
        version,
        tags,
        parameters: skillParams,
        source_code: sourceCode,
      });
      router.push(`/skills/${result.id}`);
    } catch (err) {
      console.error("Failed to create skill:", err);
      setError(err instanceof Error ? err.message : t("createError"));
    } finally {
      setSubmitting(false);
    }
  }

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

      <h1 className="text-2xl font-bold tracking-tight">{t("createSkill")}</h1>

      <form onSubmit={handleSubmit}>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Left: Metadata form */}
          <div className="space-y-5">
            {/* Name */}
            <div className="space-y-1.5">
              <label htmlFor="skill-name" className="text-sm font-medium">{t("skillName")}</label>
              <Input
                id="skill-name"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder={t("skillNamePlaceholder")}
                required
              />
            </div>

            {/* Description */}
            <div className="space-y-1.5">
              <label htmlFor="skill-description" className="text-sm font-medium">{t("skillDescription")}</label>
              <Textarea
                id="skill-description"
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder={t("skillDescriptionPlaceholder")}
                rows={3}
                required
              />
            </div>

            {/* Category */}
            <div className="space-y-1.5">
              <label className="text-sm font-medium">{t("category")}</label>
              <div className="flex gap-1.5 flex-wrap">
                {CATEGORIES.map((cat) => {
                  const CatIcon = cat.icon;
                  return (
                    <button
                      key={cat.id}
                      type="button"
                      onClick={() => setCategory(cat.id)}
                      className={cn(
                        "flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium cursor-pointer",
                        "transition-colors duration-150",
                        category === cat.id
                          ? "bg-foreground text-background border border-transparent"
                          : "text-muted-foreground hover:text-foreground hover:bg-secondary border border-border/50"
                      )}
                    >
                      <CatIcon className="w-3.5 h-3.5" />
                      {t(`categories.${cat.id}`)}
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Version */}
            <div className="space-y-1.5">
              <label htmlFor="skill-version" className="text-sm font-medium">{t("version")}</label>
              <Input
                id="skill-version"
                value={version}
                onChange={(e) => setVersion(e.target.value)}
                className="max-w-32"
              />
            </div>

            {/* Tags */}
            <div className="space-y-1.5">
              <label htmlFor="skill-tags" className="text-sm font-medium">{t("tags")}</label>
              <Input
                id="skill-tags"
                value={tagsInput}
                onChange={(e) => setTagsInput(e.target.value)}
                placeholder={t("tagsPlaceholder")}
              />
            </div>

            {/* Parameters */}
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <label className="text-sm font-medium">
                  {t("parameters", { count: parameters.length })}
                </label>
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  onClick={addParameter}
                  className="h-7 text-xs gap-1 cursor-pointer"
                >
                  <Plus className="w-3 h-3" />
                  {t("addParameter")}
                </Button>
              </div>
              {parameters.map((param, i) => (
                <div
                  key={i}
                  className="border border-border/50 rounded-lg p-3 space-y-2"
                >
                  <div className="flex items-center gap-2">
                    <Input
                      value={param.name}
                      onChange={(e) => updateParameter(i, "name", e.target.value)}
                      placeholder={t("paramName")}
                      className="flex-1 h-8 text-sm"
                    />
                    <select
                      value={param.type}
                      onChange={(e) => updateParameter(i, "type", e.target.value)}
                      className="h-8 px-2 rounded-lg border border-border bg-transparent text-sm cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
                    >
                      {PARAM_TYPES.map((type) => (
                        <option key={type} value={type}>
                          {type}
                        </option>
                      ))}
                    </select>
                    <button
                      type="button"
                      onClick={() => removeParameter(i)}
                      aria-label={t("removeParameter")}
                      className="p-1.5 text-muted-foreground hover:text-destructive cursor-pointer"
                    >
                      <X className="w-3.5 h-3.5" />
                    </button>
                  </div>
                  <Input
                    value={param.description}
                    onChange={(e) =>
                      updateParameter(i, "description", e.target.value)
                    }
                    placeholder={t("paramDescription")}
                    className="h-8 text-sm"
                  />
                  <label className="flex items-center gap-2 text-xs text-muted-foreground cursor-pointer">
                    <input
                      type="checkbox"
                      checked={param.required}
                      onChange={(e) =>
                        updateParameter(i, "required", e.target.checked)
                      }
                      className="rounded cursor-pointer h-3.5 w-3.5 border-border accent-foreground"
                    />
                    {t("paramRequired")}
                  </label>
                </div>
              ))}
            </div>
          </div>

          {/* Right: Template picker + code preview */}
          <div className="space-y-5">
            <div className="space-y-1.5">
              <label className="text-sm font-medium">{t("selectTemplate")}</label>
              <p className="text-xs text-muted-foreground">
                {t("templateDescription")}
              </p>
            </div>

            <div className="grid grid-cols-2 gap-2">
              {SKILL_TEMPLATES.map((template) => {
                const isSelected = selectedTemplate?.id === template.id;
                return (
                  <button
                    key={template.id}
                    type="button"
                    onClick={() => selectTemplate(template)}
                    className={cn(
                      "text-left p-3 rounded-lg border transition-all duration-150 cursor-pointer",
                      isSelected
                        ? "border-foreground bg-foreground/5"
                        : "border-border/50 hover:border-border hover:bg-secondary"
                    )}
                  >
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-medium">
                        {t(`templates.${template.id}.name`)}
                      </span>
                      {isSelected && (
                        <CheckCircle2 className="w-3.5 h-3.5 text-emerald-500" />
                      )}
                    </div>
                    <p className="text-xs text-muted-foreground mt-1">
                      {t(`templates.${template.id}.description`)}
                    </p>
                  </button>
                );
              })}
            </div>

            {/* Code editor */}
            {selectedTemplate && (
              <div className="space-y-1.5">
                <label className="text-sm font-medium">{t("codePreview")}</label>
                <SkillCodeEditor
                  value={sourceCode}
                  onChange={setSourceCode}
                  height="384px"
                />
              </div>
            )}
          </div>
        </div>

        {/* Error */}
        {error && (
          <div role="alert" className="mt-6 p-3 bg-destructive/10 border border-destructive/20 rounded-lg text-sm text-destructive">
            {error}
          </div>
        )}

        {/* Actions */}
        <div className="flex items-center gap-3 mt-8 pt-6 border-t border-border/50">
          <Button
            type="submit"
            disabled={submitting}
            className="gap-2 cursor-pointer"
          >
            {submitting && <Loader2 className="w-3.5 h-3.5 animate-spin" />}
            {submitting ? t("creating") : t("createSkill")}
          </Button>
          <Link href="/skills">
            <Button type="button" variant="ghost" className="cursor-pointer">
              {t("cancel")}
            </Button>
          </Link>
        </div>
      </form>
    </div>
  );
}
