// @ts-nocheck
// Port phần IndexedDB của web/js/recording.js — bản ghi lưu TRÊN THIẾT BỊ (sống qua
// reload; localStorage không giữ Blob được). Mỗi row: {id, name, blob, type, size, createdAt}.

const REC_DB_NAME = 'speaking-grader';
const REC_STORE = 'recordings';
let recDbPromise = null;

function recDb() {
  if (recDbPromise) return recDbPromise;
  recDbPromise = new Promise((resolve, reject) => {
    const req = indexedDB.open(REC_DB_NAME, 1);
    req.onupgradeneeded = () => {
      req.result.createObjectStore(REC_STORE, { keyPath: 'id', autoIncrement: true });
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
  return recDbPromise;
}
function recStore(mode) {
  return recDb().then((db) => db.transaction(REC_STORE, mode).objectStore(REC_STORE));
}
function reqDone(req) {
  return new Promise((resolve, reject) => {
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

export async function saveRecording(rec) {
  const store = await recStore('readwrite');
  return reqDone(store.add(rec));
}
export async function listRecordings() {
  const store = await recStore('readonly');
  const all = await reqDone(store.getAll());
  return all.sort((a, b) => b.createdAt - a.createdAt); // newest first
}
export async function getRecording(id) {
  const store = await recStore('readonly');
  return reqDone(store.get(id));
}
export async function deleteRecordingDb(id) {
  const store = await recStore('readwrite');
  return reqDone(store.delete(id));
}
export async function clearRecordingsDb() {
  const store = await recStore('readwrite');
  return reqDone(store.clear());
}

export function recordingExtension(mimeType) {
  if (mimeType.includes('ogg')) return '.ogg';
  if (mimeType.includes('mp4') || mimeType.includes('mpeg')) return '.mp4';
  return '.webm';
}
export function formatBytes(n) {
  if (!n) return '';
  const kb = n / 1024;
  return kb < 1024 ? `${kb.toFixed(0)} KB` : `${(kb / 1024).toFixed(1)} MB`;
}
