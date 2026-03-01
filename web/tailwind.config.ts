import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: ["class"],
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      fontSize: {
        // UI Scale - optimized for clarity
        'xs': ['0.75rem', { lineHeight: '1rem', letterSpacing: '0' }],             // 12px - labels, badges
        'sm': ['0.875rem', { lineHeight: '1.25rem', letterSpacing: '0' }],         // 14px - primary UI, body text
        'base': ['0.9375rem', { lineHeight: '1.5rem', letterSpacing: '0' }],       // 15px - emphasized UI
        // Reading Scale - comfortable for prose
        'lg': ['1rem', { lineHeight: '1.625rem', letterSpacing: '-0.01em' }],      // 16px - prose body
        'xl': ['1.125rem', { lineHeight: '1.75rem', letterSpacing: '-0.01em' }],   // 18px - lead text
        // Heading Scale - clean hierarchy
        '2xl': ['1.25rem', { lineHeight: '1.75rem', letterSpacing: '-0.015em' }],  // 20px - h4
        '3xl': ['1.5rem', { lineHeight: '2rem', letterSpacing: '-0.02em' }],       // 24px - h3
        '4xl': ['2rem', { lineHeight: '2.5rem', letterSpacing: '-0.025em' }],      // 32px - h2
        '5xl': ['2.5rem', { lineHeight: '3rem', letterSpacing: '-0.03em' }],       // 40px - h1 hero
        '6xl': ['3rem', { lineHeight: '3.5rem', letterSpacing: '-0.03em' }],       // 48px - display
      },
      colors: {
        // Theme colors (OKLCH with alpha-value for opacity modifiers)
        background: "oklch(var(--background) / <alpha-value>)",
        foreground: "oklch(var(--foreground) / <alpha-value>)",
        card: {
          DEFAULT: "oklch(var(--card) / <alpha-value>)",
          foreground: "oklch(var(--card-foreground) / <alpha-value>)",
        },
        popover: {
          DEFAULT: "oklch(var(--popover) / <alpha-value>)",
          foreground: "oklch(var(--popover-foreground) / <alpha-value>)",
        },
        primary: {
          DEFAULT: "oklch(var(--primary) / <alpha-value>)",
          foreground: "oklch(var(--primary-foreground) / <alpha-value>)",
        },
        secondary: {
          DEFAULT: "oklch(var(--secondary) / <alpha-value>)",
          foreground: "oklch(var(--secondary-foreground) / <alpha-value>)",
        },
        muted: {
          DEFAULT: "oklch(var(--muted) / <alpha-value>)",
          foreground: "oklch(var(--muted-foreground) / <alpha-value>)",
        },
        accent: {
          DEFAULT: "oklch(var(--accent) / <alpha-value>)",
          foreground: "oklch(var(--accent-foreground) / <alpha-value>)",
        },
        destructive: {
          DEFAULT: "oklch(var(--destructive) / <alpha-value>)",
          foreground: "oklch(var(--destructive-foreground) / <alpha-value>)",
        },
        border: "oklch(var(--border) / <alpha-value>)",
        input: "oklch(var(--input) / <alpha-value>)",
        ring: "oklch(var(--ring) / <alpha-value>)",
        // Sidebar colors (OKLCH)
        sidebar: {
          DEFAULT: "oklch(var(--sidebar) / <alpha-value>)",
          foreground: "oklch(var(--sidebar-foreground) / <alpha-value>)",
          primary: "oklch(var(--sidebar-primary) / <alpha-value>)",
          "primary-foreground": "oklch(var(--sidebar-primary-foreground) / <alpha-value>)",
          accent: "oklch(var(--sidebar-accent) / <alpha-value>)",
          "accent-foreground": "oklch(var(--sidebar-accent-foreground) / <alpha-value>)",
          border: "oklch(var(--sidebar-border) / <alpha-value>)",
          ring: "oklch(var(--sidebar-ring) / <alpha-value>)",
        },
        // Chart colors (OKLCH)
        chart: {
          "1": "oklch(var(--chart-1) / <alpha-value>)",
          "2": "oklch(var(--chart-2) / <alpha-value>)",
          "3": "oklch(var(--chart-3) / <alpha-value>)",
          "4": "oklch(var(--chart-4) / <alpha-value>)",
          "5": "oklch(var(--chart-5) / <alpha-value>)",
        },
        // Custom colors (HSL)
        "accent-cyan": "hsl(var(--accent-cyan) / <alpha-value>)",
        success: "hsl(var(--success) / <alpha-value>)",
        warning: "hsl(var(--warning) / <alpha-value>)",
        info: "hsl(var(--info) / <alpha-value>)",
      },
      borderRadius: {
        lg: "var(--radius)",
        md: "calc(var(--radius) - 2px)",
        sm: "calc(var(--radius) - 4px)",
        xl: "calc(var(--radius) + 0.25rem)",
        "2xl": "calc(var(--radius) + 0.5rem)",
      },
      fontFamily: {
        sans: [
          "var(--font-dm-sans)",
          "system-ui",
          "-apple-system",
          "BlinkMacSystemFont",
          "Segoe UI",
          "sans-serif",
        ],
        display: [
          "var(--font-dm-sans)",
          "system-ui",
          "sans-serif",
        ],
        mono: [
          "var(--font-jetbrains-mono)",
          "ui-monospace",
          "SFMono-Regular",
          "Menlo",
          "Consolas",
          "monospace",
        ],
      },
      boxShadow: {
        // Minimal shadows only for elevation when absolutely needed
        "subtle": "0 1px 2px 0 rgb(0 0 0 / 0.03)",
      },
    },
  },
  plugins: [],
};

export default config;
