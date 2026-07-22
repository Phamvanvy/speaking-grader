// Cửa hàng cosmetic (Phase 4 game hóa): lưới vật phẩm, mua bằng xu, trang bị/tháo.
// Item thuần trang trí (theme thanh XP, màu ngọn lửa) — KHÔNG ảnh hưởng chấm điểm.
// Server là nguồn sự thật (xu/giá/sở hữu); store useShop lo gọi API + đồng bộ ví.

import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Check, Loader2, Lock } from 'lucide-react';
import { cn } from '@/lib/utils';
import { useShop, type ShopItem } from '@/store/shop';

const SLOT_LABEL: Record<string, string> = {
  xp_theme: 'Thanh XP',
  streak_flame: 'Ngọn lửa streak',
};

export default function ShopDialog() {
  const open = useShop((s) => s.open);
  const close = useShop((s) => s.close);
  const data = useShop((s) => s.data);
  const loading = useShop((s) => s.loading);

  // Nhóm item theo slot để người dùng thấy rõ "1 slot chỉ trang bị 1".
  const groups = groupBySlot(data?.items ?? []);

  return (
    <Dialog open={open} onOpenChange={(o) => !o && close()}>
      <DialogContent className="max-h-[92vh] max-w-lg gap-4 overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            🛍️ Cửa hàng
            <span className="ml-auto flex items-center gap-1.5 rounded-full border border-amber-300 bg-amber-50 px-3 py-1 text-sm font-bold tabular-nums dark:border-amber-500/40 dark:bg-amber-950/40">
              🪙 {data?.coins ?? 0}
            </span>
          </DialogTitle>
        </DialogHeader>

        <p className="text-sm text-muted-foreground">
          Dùng xu (kiếm từ nhiệm vụ hằng ngày) đổi vật phẩm trang trí. Chỉ để cho vui — không
          ảnh hưởng điểm luyện tập.
        </p>

        {loading && !data ? (
          <div className="flex justify-center py-10 text-muted-foreground">
            <Loader2 className="h-6 w-6 animate-spin" />
          </div>
        ) : (
          <div className="flex flex-col gap-5">
            {groups.map(([slot, items]) => (
              <section key={slot} className="flex flex-col gap-2">
                <h4 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                  {SLOT_LABEL[slot] ?? slot}
                </h4>
                <div className="grid grid-cols-2 gap-2">
                  {items.map((it) => (
                    <ItemCard key={it.id} item={it} />
                  ))}
                </div>
              </section>
            ))}
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}

function ItemCard({ item }: { item: ShopItem }) {
  const buy = useShop((s) => s.buy);
  const equip = useShop((s) => s.equip);
  const busyId = useShop((s) => s.busyId);
  const busy = busyId === item.id;

  return (
    <div
      className={cn(
        'flex flex-col gap-2 rounded-xl border p-3 transition-colors',
        item.equipped ? 'border-primary bg-primary/5' : 'border-border',
      )}
    >
      <div className="flex items-center gap-2">
        <span className="text-2xl" aria-hidden>
          {item.icon}
        </span>
        <div className="min-w-0">
          <div className="truncate text-sm font-semibold">{item.label}</div>
          <div className="truncate text-xs text-muted-foreground">{item.desc}</div>
        </div>
      </div>

      {item.owned ? (
        item.equipped ? (
          <Button size="sm" variant="secondary" disabled={busy} onClick={() => equip(item.id, false)}>
            {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Check className="h-4 w-4" />}
            Đang dùng
          </Button>
        ) : (
          <Button size="sm" variant="outline" disabled={busy} onClick={() => equip(item.id, true)}>
            {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : null}
            Trang bị
          </Button>
        )
      ) : (
        <Button
          size="sm"
          disabled={busy || !item.affordable}
          onClick={() => buy(item.id)}
          title={item.affordable ? undefined : 'Chưa đủ xu'}
        >
          {busy ? (
            <Loader2 className="h-4 w-4 animate-spin" />
          ) : item.affordable ? null : (
            <Lock className="h-4 w-4" />
          )}
          🪙 {item.price}
        </Button>
      )}
    </div>
  );
}

/** Gom item theo slot, giữ thứ tự xuất hiện đầu tiên của mỗi slot. */
function groupBySlot(items: ShopItem[]): [string, ShopItem[]][] {
  const order: string[] = [];
  const map = new Map<string, ShopItem[]>();
  for (const it of items) {
    if (!map.has(it.slot)) {
      map.set(it.slot, []);
      order.push(it.slot);
    }
    map.get(it.slot)!.push(it);
  }
  return order.map((s) => [s, map.get(s)!]);
}
