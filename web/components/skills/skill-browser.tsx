"use client";

import { useEffect, useState, useDeferredValue } from "react";
import { useTranslations } from "next-intl";
import Link from "next/link";
import { listSkills } from "@/lib/api/skills";
import { SkillMetadata } from "@/lib/types/skills";
import { SkillCard } from "./skill-card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import {
  Search,
  Loader2,
  AlertCircle,
  Code,
  Sparkles,
  BarChart3,
  FileText,
  LayoutGrid,
  Plus,
} from "lucide-react";
import { cn } from "@/lib/utils";

const CATEGORIES = ["all", "research", "code", "data", "creative", "automation"] as const;
type CategoryFilter = (typeof CATEGORIES)[number];

const CATEGORY_ICONS: Record<CategoryFilter, React.ReactNode> = {
  all: <LayoutGrid className="w-3.5 h-3.5" />,
  research: <Search className="w-3.5 h-3.5" />,
  code: <Code className="w-3.5 h-3.5" />,
  data: <BarChart3 className="w-3.5 h-3.5" />,
  creative: <Sparkles className="w-3.5 h-3.5" />,
  automation: <FileText className="w-3.5 h-3.5" />,
};

interface SkillBrowserProps {
  onSkillSelect?: (skillId: string) => void;
}

export function SkillBrowser({ onSkillSelect }: SkillBrowserProps) {
  const t = useTranslations("skills");
  const [skills, setSkills] = useState<SkillMetadata[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [categoryFilter, setCategoryFilter] = useState<CategoryFilter>("all");

  const deferredSearch = useDeferredValue(searchQuery);

  useEffect(() => {
    loadSkills();
  }, []);

  async function loadSkills() {
    try {
      setLoading(true);
      setError(null);
      const data = await listSkills();
      setSkills(data);
    } catch (err) {
      console.error("Failed to load skills:", err);
      setError(t("loadError"));
    } finally {
      setLoading(false);
    }
  }

  // Get unique categories from actual data
  const categories = Array.from(new Set(skills.map((s) => s.category)));

  // Filter skills using deferred search value
  const filteredSkills = skills.filter((skill) => {
    const matchesSearch =
      deferredSearch === "" ||
      skill.name.toLowerCase().includes(deferredSearch.toLowerCase()) ||
      skill.description.toLowerCase().includes(deferredSearch.toLowerCase()) ||
      skill.tags?.some((tag) =>
        tag.toLowerCase().includes(deferredSearch.toLowerCase())
      );

    const matchesCategory =
      categoryFilter === "all" || skill.category === categoryFilter;

    return matchesSearch && matchesCategory;
  });

  // Count skills per category
  const categoryCounts: Record<string, number> = {};
  skills.forEach((s) => {
    const searchMatch =
      deferredSearch === "" ||
      s.name.toLowerCase().includes(deferredSearch.toLowerCase()) ||
      s.description.toLowerCase().includes(deferredSearch.toLowerCase()) ||
      s.tags?.some((tag) =>
        tag.toLowerCase().includes(deferredSearch.toLowerCase())
      );
    if (searchMatch) {
      categoryCounts[s.category] = (categoryCounts[s.category] || 0) + 1;
    }
  });
  const totalFilteredBySearch = Object.values(categoryCounts).reduce((a, b) => a + b, 0);

  // Group by category
  const groupedSkills = categories.reduce(
    (acc, category) => {
      acc[category] = filteredSkills.filter((s) => s.category === category);
      return acc;
    },
    {} as Record<string, SkillMetadata[]>
  );

  const categoryLabels: Record<CategoryFilter, string> = {
    all: t("allCategories"),
    research: t("categories.research"),
    code: t("categories.code"),
    data: t("categories.data"),
    creative: t("categories.creative"),
    automation: t("categories.automation"),
  };

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center py-24 gap-3">
        <Loader2 className="w-5 h-5 animate-spin text-muted-foreground" />
        <p className="text-sm text-muted-foreground">{t("loadingSkills")}</p>
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
          onClick={loadSkills}
          className="text-sm text-foreground font-medium underline underline-offset-4 hover:text-foreground/80 cursor-pointer"
        >
          {t("retry")}
        </button>
      </div>
    );
  }

  if (skills.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-24 gap-3">
        <div className="w-10 h-10 rounded-full bg-secondary flex items-center justify-center">
          <Sparkles className="w-5 h-5 text-muted-foreground" />
        </div>
        <p className="text-sm text-muted-foreground">{t("noSkills")}</p>
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

        {/* Create Skill button */}
        <Link href="/skills/new">
          <Button size="sm" className="h-9 gap-1.5 cursor-pointer">
            <Plus className="w-3.5 h-3.5" />
            {t("createSkill")}
          </Button>
        </Link>

        {/* Category filter pills */}
        <div className="flex items-center gap-1.5 flex-wrap">
          {CATEGORIES.map((cat) => {
            const count = cat === "all"
              ? totalFilteredBySearch
              : (categoryCounts[cat] || 0);

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
      {filteredSkills.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-16 gap-2">
          <Search className="w-5 h-5 text-muted-foreground/50" />
          <p className="text-sm text-muted-foreground">{t("noResults")}</p>
        </div>
      ) : categoryFilter === "all" ? (
        // Grouped view
        <div className="space-y-8">
          {categories.map((category) => {
            const categorySkills = groupedSkills[category];
            if (!categorySkills || categorySkills.length === 0) return null;

            return (
              <section key={category}>
                <div className="flex items-center gap-2 mb-4">
                  <h3 className="text-sm font-semibold text-foreground uppercase tracking-wide">
                    {categoryLabels[category as CategoryFilter] || category}
                  </h3>
                  <span className="text-xs text-muted-foreground tabular-nums">
                    ({categorySkills.length})
                  </span>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                  {categorySkills.map((skill, idx) => (
                    <SkillCard
                      key={skill.id}
                      skill={skill}
                      onExecute={onSkillSelect}
                      index={idx}
                    />
                  ))}
                </div>
              </section>
            );
          })}
        </div>
      ) : (
        // Flat view for single category
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
          {filteredSkills.map((skill, idx) => (
            <SkillCard
              key={skill.id}
              skill={skill}
              onExecute={onSkillSelect}
              index={idx}
            />
          ))}
        </div>
      )}
    </div>
  );
}
