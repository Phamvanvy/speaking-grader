// Popup thêm nhanh từ vựng (nút 📚 góc phải trên) — port addwords popup của saved.js
// sang shadcn Dialog. Nhiều dòng, mỗi dòng 1 từ/cụm; "Lưu tất cả" → POST /words từng
// từ (server tự tra IPA). Dòng lưu xong/trùng bỏ đi, dòng sai giữ lại + viền đỏ.

import { useState } from 'react';
import { create } from 'zustand';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Plus, X } from 'lucide-react';
import { useSavedWords } from '@/store/savedWords';

// Store mở/đóng — nút 📚 ở App gọi open(); Dialog tự quản nội dung.
interface AddWordsUi {
  open: boolean;
  setOpen: (open: boolean) => void;
}
export const useAddWords = create<AddWordsUi>((set) => ({
  open: false,
  setOpen: (open) => set({ open }),
}));

// Validate client mirror _WORD_RE của src/words.py (lỗi hiện ngay, không tốn round-trip).
const ADD_WORD_RE = /^[A-Za-z][A-Za-z' -]{0,39}$/;
const norm = (v: string) => (v || '').trim().toLowerCase().replace(/\s+/g, ' ');

interface Row {
  value: string;
  err: boolean;
}

export default function AddWordsDialog() {
  const open = useAddWords((s) => s.open);
  const setOpen = useAddWords((s) => s.setOpen);
  const swAdd = useSavedWords((s) => s.add);
  const swHas = useSavedWords((s) => s.has);

  const [rows, setRows] = useState<Row[]>([{ value: '', err: false }]);
  const [msg, setMsg] = useState<{ text: string; err: boolean }>({ text: '', err: false });
  const [saving, setSaving] = useState(false);

  function reset() {
    setRows([{ value: '', err: false }]);
    setMsg({ text: '', err: false });
  }

  function setRow(i: number, value: string) {
    setRows((rs) => rs.map((r, j) => (j === i ? { value, err: false } : r)));
  }
  function addRow() {
    setRows((rs) => [...rs, { value: '', err: false }]);
  }
  function delRow(i: number) {
    setRows((rs) => (rs.length > 1 ? rs.filter((_, j) => j !== i) : [{ value: '', err: false }]));
  }

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setSaving(true);
    setMsg({ text: 'Đang lưu…', err: false });
    let saved = 0,
      dup = 0,
      bad = 0,
      failed = 0;
    const remaining: Row[] = [];
    for (const row of rows) {
      const word = norm(row.value);
      if (!word) continue; // dòng trống bỏ qua
      if (!ADD_WORD_RE.test(word) || word.split(' ').length > 4) {
        bad++;
        remaining.push({ value: row.value, err: true });
        continue;
      }
      if (swHas(word)) {
        dup++;
        continue;
      }
      try {
        await swAdd({ word });
        saved++;
      } catch {
        failed++;
        remaining.push({ value: row.value, err: true });
      }
    }
    setRows(remaining.length ? remaining : [{ value: '', err: false }]);
    const parts: string[] = [];
    if (saved) parts.push(`đã lưu ${saved} từ`);
    if (dup) parts.push(`${dup} từ đã có sẵn`);
    if (bad) parts.push(`${bad} dòng không hợp lệ (chỉ chữ cái tiếng Anh, nháy đơn, gạch nối; cụm ≤4 từ)`);
    if (failed) parts.push(`${failed} dòng lưu lỗi — thử lại`);
    const text = parts.length ? parts.join(' · ') : 'Chưa có từ nào để lưu — nhập từ vào ô trên.';
    setMsg({ text: text.charAt(0).toUpperCase() + text.slice(1), err: !!(bad || failed) });
    setSaving(false);
  }

  return (
    <Dialog
      open={open}
      onOpenChange={(o) => {
        setOpen(o);
        if (!o) reset();
      }}
    >
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle>📚 Thêm từ vựng</DialogTitle>
        </DialogHeader>
        <p className="text-sm text-muted-foreground">
          Nhập từ hoặc cụm tiếng Anh muốn lưu để luyện tập (cụm tối đa 4 từ). Bấm ＋ để thêm nhiều từ cùng lúc.
        </p>
        <form onSubmit={submit} className="flex flex-col gap-2">
          <div className="flex flex-col gap-2">
            {rows.map((row, i) => (
              <div key={i} className="flex items-center gap-2">
                <Input
                  value={row.value}
                  onChange={(e) => setRow(i, e.target.value)}
                  placeholder="ví dụ: bookstore / gain valuable insights"
                  maxLength={40}
                  autoComplete="off"
                  spellCheck={false}
                  className={row.err ? 'border-destructive' : ''}
                />
                {/* shrink-0: không cho flex bóp nút ✕ nhỏ lại — nếu bóp thì mép phải
                    nút "Lưu tất cả" (mr-11) sẽ lệch vài px so với ô nhập. */}
                <Button type="button" variant="ghost" size="icon" className="shrink-0" onClick={() => delRow(i)} title="Xoá dòng" aria-label="Xoá dòng">
                  <X className="h-4 w-4" />
                </Button>
              </div>
            ))}
          </div>
          <Button type="button" variant="secondary" size="sm" className="self-start" onClick={addRow}>
            <Plus className="h-4 w-4" /> Thêm dòng
          </Button>
          {msg.text && <div className={`text-sm ${msg.err ? 'text-destructive' : 'text-muted-foreground'}`}>{msg.text}</div>}
          {/* mr-11 = bề ngang nút ✕ (h-9 = 36px) + gap-2 (8px): mép phải nút Lưu
              thẳng hàng đúng mép phải ô nhập thay vì nhô ra thêm 44px. */}
          <Button type="submit" className="mr-11" disabled={saving}>
            Lưu tất cả
          </Button>
        </form>
      </DialogContent>
    </Dialog>
  );
}
