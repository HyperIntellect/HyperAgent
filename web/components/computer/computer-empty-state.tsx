import { cn } from "@/lib/utils";

interface ComputerEmptyStateProps {
    icon: React.ElementType;
    title: string;
    subtitle?: string;
    className?: string;
    children?: React.ReactNode;
}

export function ComputerEmptyState({ icon: Icon, title, subtitle, className, children }: ComputerEmptyStateProps) {
    return (
        <div className={cn("flex-1 flex flex-col items-center justify-center py-12 px-6", className)}>
            <div className="w-16 h-16 rounded-2xl bg-secondary/50 flex items-center justify-center mb-4">
                <Icon className="w-8 h-8 text-muted-foreground/40" />
            </div>
            <p className="text-sm font-medium text-muted-foreground">{title}</p>
            {subtitle && <p className="text-xs text-muted-foreground/60 mt-1">{subtitle}</p>}
            {children}
        </div>
    );
}
