// Dựng file .zip ngay trên trình duyệt (không phụ thuộc thư viện) để tải TẤT CẢ
// audio trong MỘT lượt tải duy nhất — thay cho kiểu tải rời từng file.
//
// Dùng ZIP_STORED (không nén): audio webm/mp3/... đã nén sẵn nên deflate chỉ tốn
// CPU mà gần như không nhỏ thêm — khớp với backend /history/{id}/audio.zip.

const _CRC_TABLE: Uint32Array = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c >>> 0;
  }
  return t;
})();

function crc32(bytes: Uint8Array): number {
  let c = 0xffffffff;
  for (let i = 0; i < bytes.length; i++) c = _CRC_TABLE[(c ^ bytes[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

export interface ZipEntry {
  filename: string;
  blob: Blob;
}

// Trả về 1 Blob application/zip chứa tất cả entry (STORED). Tên trùng được thêm
// hậu tố -2, -3… để không đè nhau trong archive.
export async function buildZip(entries: ZipEntry[]): Promise<Blob> {
  const enc = new TextEncoder();
  const seen = new Map<string, number>();

  const files = await Promise.all(
    entries.map(async (e) => {
      let name = e.filename || 'audio';
      const n = (seen.get(name) || 0) + 1;
      seen.set(name, n);
      if (n > 1) {
        const dot = name.lastIndexOf('.');
        name = dot > 0 ? `${name.slice(0, dot)}-${n}${name.slice(dot)}` : `${name}-${n}`;
      }
      const data = new Uint8Array(await e.blob.arrayBuffer());
      return { nameBytes: enc.encode(name), data, crc: crc32(data) };
    }),
  );

  const parts: Uint8Array<ArrayBuffer>[] = [];
  const central: Uint8Array<ArrayBuffer>[] = [];
  let offset = 0;

  const u16 = (v: number) => Uint8Array.from([v & 0xff, (v >>> 8) & 0xff]) as Uint8Array<ArrayBuffer>;
  const u32 = (v: number) =>
    Uint8Array.from([v & 0xff, (v >>> 8) & 0xff, (v >>> 16) & 0xff, (v >>> 24) & 0xff]) as Uint8Array<ArrayBuffer>;

  for (const f of files) {
    const local = concat([
      u32(0x04034b50), // local file header signature
      u16(20), // version needed
      u16(0), // flags
      u16(0), // method: STORED
      u16(0), // mod time
      u16(0), // mod date
      u32(f.crc),
      u32(f.data.length), // compressed size
      u32(f.data.length), // uncompressed size
      u16(f.nameBytes.length),
      u16(0), // extra len
      f.nameBytes,
      f.data,
    ]);
    parts.push(local);

    central.push(
      concat([
        u32(0x02014b50), // central dir header signature
        u16(20), // version made by
        u16(20), // version needed
        u16(0), // flags
        u16(0), // method
        u16(0), // mod time
        u16(0), // mod date
        u32(f.crc),
        u32(f.data.length),
        u32(f.data.length),
        u16(f.nameBytes.length),
        u16(0), // extra len
        u16(0), // comment len
        u16(0), // disk number
        u16(0), // internal attrs
        u32(0), // external attrs
        u32(offset), // local header offset
        f.nameBytes,
      ]),
    );
    offset += local.length;
  }

  const centralBytes = concat(central);
  const end = concat([
    u32(0x06054b50), // end of central dir signature
    u16(0), // disk number
    u16(0), // disk with central dir
    u16(files.length),
    u16(files.length),
    u32(centralBytes.length),
    u32(offset), // central dir offset
    u16(0), // comment len
  ]);

  return new Blob([...parts, centralBytes, end], { type: 'application/zip' });
}

function concat(chunks: Uint8Array[]): Uint8Array<ArrayBuffer> {
  let len = 0;
  for (const c of chunks) len += c.length;
  const out = new Uint8Array(new ArrayBuffer(len));
  let p = 0;
  for (const c of chunks) {
    out.set(c, p);
    p += c.length;
  }
  return out;
}

// Dựng zip rồi tải xuống 1 lần. Trả về số entry đã đóng gói.
export async function downloadZipFromBlobs(entries: ZipEntry[], zipName: string): Promise<number> {
  const zip = await buildZip(entries);
  const url = URL.createObjectURL(zip);
  const a = document.createElement('a');
  a.href = url;
  a.download = zipName.endsWith('.zip') ? zipName : `${zipName}.zip`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
  return entries.length;
}
