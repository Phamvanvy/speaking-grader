import { Toaster as Sonner, toast } from 'sonner';
import { useUiStore } from '@/store/ui';

type ToasterProps = React.ComponentProps<typeof Sonner>;

// Wrapper shadcn cho sonner — theme lấy từ store legacy (body.dark) thay next-themes.
const Toaster = ({ ...props }: ToasterProps) => {
  const theme = useUiStore((s) => s.theme);

  return (
    <Sonner
      theme={theme}
      className="toaster group"
      toastOptions={{
        classNames: {
          toast:
            'group toast group-[.toaster]:bg-background group-[.toaster]:text-foreground group-[.toaster]:border-border group-[.toaster]:shadow-lg',
          description: 'group-[.toast]:text-muted-foreground',
          actionButton: 'group-[.toast]:bg-primary group-[.toast]:text-primary-foreground',
          cancelButton: 'group-[.toast]:bg-muted group-[.toast]:text-muted-foreground',
        },
      }}
      {...props}
    />
  );
};

export { Toaster, toast };
