"use client";

import { useTranslations } from "next-intl";
import { MainLayout } from "@/components/layout/main-layout";
import { LibraryBrowser } from "@/components/library";

export default function LibraryPage() {
  const t = useTranslations("library");

  return (
    <MainLayout>
      <div className="flex-1 overflow-y-auto">
        {/* Hero section */}
        <div className="border-b border-border/50">
          <div className="container mx-auto px-6 pt-12 pb-8 max-w-6xl">
            <h1 className="text-3xl font-bold tracking-tight">
              {t("title")}
            </h1>
            <p className="text-muted-foreground mt-2 text-base max-w-xl">
              {t("subtitle")}
            </p>
          </div>
        </div>

        {/* Library content */}
        <div className="container mx-auto px-6 py-6 max-w-6xl">
          <LibraryBrowser />
        </div>
      </div>
    </MainLayout>
  );
}
