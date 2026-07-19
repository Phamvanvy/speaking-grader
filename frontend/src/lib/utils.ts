import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

// Helper chuẩn shadcn: gộp class có điều kiện + dedupe conflict Tailwind.
export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}
