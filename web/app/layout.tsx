import type { Metadata, Viewport } from "next";
import { DM_Sans, JetBrains_Mono } from "next/font/google";
import { NextIntlClientProvider } from "next-intl";
import { getLocale, getMessages, getTranslations } from "next-intl/server";
import { SessionProvider } from "@/components/providers/session-provider";
import { ErrorHandler } from "@/components/providers/error-handler";
import { AuthGuard } from "@/components/auth/auth-guard";
import "./globals.css";

// Load fonts with next/font for better performance (eliminates render-blocking request)
const dmSans = DM_Sans({
    subsets: ["latin"],
    weight: ["400", "500", "600", "700"],
    display: "swap",
    variable: "--font-dm-sans",
});


const jetbrainsMono = JetBrains_Mono({
    subsets: ["latin"],
    weight: ["400", "500", "600"],
    display: "swap",
    variable: "--font-jetbrains-mono",
});

export const viewport: Viewport = {
    width: 'device-width',
    initialScale: 1,
    maximumScale: 5,
    viewportFit: 'cover',
};

export async function generateMetadata(): Promise<Metadata> {
    const t = await getTranslations("metadata");
    return {
        title: t("title"),
        description: t("description"),
        icons: {
            icon: "/images/logo-dark.svg",
            apple: "/images/logo-dark.svg",
        },
    };
}

export default async function RootLayout({
    children,
}: Readonly<{
    children: React.ReactNode;
}>) {
    const locale = await getLocale();
    const messages = await getMessages();

    return (
        <html lang={locale} suppressHydrationWarning className={`${dmSans.variable} ${jetbrainsMono.variable}`}>
            <head>
                <script
                    dangerouslySetInnerHTML={{
                        __html: `
              (function() {
                const stored = localStorage.getItem('theme-preference');
                const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
                const theme = stored || 'auto';
                const resolved = theme === 'auto' ? (prefersDark ? 'dark' : 'light') : theme;
                if (resolved === 'dark') document.documentElement.classList.add('dark');
              })();
            `,
                    }}
                />
            </head>
            <body className="antialiased">
                <ErrorHandler />
                <SessionProvider>
                    <NextIntlClientProvider messages={messages}>
                        <AuthGuard>{children}</AuthGuard>
                    </NextIntlClientProvider>
                </SessionProvider>
            </body>
        </html>
    );
}
