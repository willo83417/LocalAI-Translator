const OCR_CACHE_NAME = 'PaddleOCR-Cache-v2';

// Check if OPFS is supported
const isOPFSSupported = typeof navigator !== 'undefined' && 'storage' in navigator && 'getDirectory' in navigator.storage;

const getCacheFileKey = (key: string) => `ocr_${key}`;

async function getCacheAPI() {
    return caches.open(OCR_CACHE_NAME);
}

// Retrieves a value from the storage by its key.
export const getFromDB = async <T>(key: string): Promise<T | undefined> => {
    try {
        if (isOPFSSupported) {
             const root = await navigator.storage.getDirectory();
             try {
                 const cacheDir = await root.getDirectoryHandle('ocr_cache', { create: true });
                 const fileHandle = await cacheDir.getFileHandle(getCacheFileKey(key));
                 const file = await fileHandle.getFile();
                 if (key.startsWith('dict')) {
                      return await file.text() as unknown as T;
                 } else {
                      return await file.arrayBuffer() as unknown as T;
                 }
             } catch (e: any) {
                 if (e.name !== 'NotFoundError' && e.name !== 'NotAllowedError') {
                     console.warn(`OPFS read error for ${key}:`, e);
                 }
             }
        } else {
             const cache = await getCacheAPI();
             const response = await cache.match(new Request(`https://local-ocr-cache/${key}`));
             if (response) {
                  if (key.startsWith('dict')) return await response.text() as unknown as T;
                  return await response.arrayBuffer() as unknown as T;
             }
        }
    } catch(e) {
         console.warn(`Error getting ${key} from storage:`, e);
    }
    return undefined;
};

// Sets a value in the storage with a given key.
export const setInDB = async (key: string, value: any): Promise<void> => {
     try {
         if (isOPFSSupported) {
              const root = await navigator.storage.getDirectory();
              const cacheDir = await root.getDirectoryHandle('ocr_cache', { create: true });
              const fileHandle = await cacheDir.getFileHandle(getCacheFileKey(key), { create: true });
              
              if (typeof self !== 'undefined' && typeof (fileHandle as any).createSyncAccessHandle === 'function') {
                  const accessHandle = await (fileHandle as any).createSyncAccessHandle();
                  let bufferData: Uint8Array;
                  if (typeof value === 'string') {
                       bufferData = new TextEncoder().encode(value);
                  } else if (value instanceof ArrayBuffer) {
                       bufferData = new Uint8Array(value);
                  } else {
                       bufferData = new Uint8Array(value);
                  }
                  accessHandle.truncate(0);
                  accessHandle.write(bufferData, { at: 0 });
                  accessHandle.flush();
                  accessHandle.close();
              } else {
                  const writable = await (fileHandle as any).createWritable();
                  if (typeof value === 'string') {
                       await writable.write(new Blob([value], { type: 'text/plain' }));
                  } else {
                       await writable.write(value);
                  }
                  await writable.close();
              }
         } else {
             const cache = await getCacheAPI();
             const headers = new Headers();
             let body: Blob | string;
             if (typeof value === 'string') {
                 body = value;
                 headers.set('Content-Type', 'text/plain');
             } else {
                 body = new Blob([value], { type: 'application/octet-stream' });
                 headers.set('Content-Type', 'application/octet-stream');
             }
             await cache.put(new Request(`https://local-ocr-cache/${key}`), new Response(body, { headers }));
         }
     } catch (e) {
          console.error(`Error saving ${key} to storage:`, e);
          throw e; // Rethrow to let the caller handle it
     }
};

export const deleteOcrModelCache = async (modelKey: string): Promise<void> => {
    // We target `rec-model-${modelKey}`, `dict-${modelKey}`, and `det-model-${modelKey}`
    const keysToDelete = [`rec-model-${modelKey}`, `dict-${modelKey}`, `det-model-${modelKey}`];
    
    try {
        if (isOPFSSupported) {
            const root = await navigator.storage.getDirectory();
            try {
                const cacheDir = await root.getDirectoryHandle('ocr_cache');
                for (const k of keysToDelete) {
                    try {
                        await cacheDir.removeEntry(getCacheFileKey(k));
                    } catch (e) {} // ignore not found
                }
            } catch (e) {}
        } else {
            const cache = await getCacheAPI();
            for (const k of keysToDelete) {
                await cache.delete(new Request(`https://local-ocr-cache/${k}`));
            }
        }
    } catch (e) {
         console.error(`Error deleting OCR cache for ${modelKey}:`, e);
    }
};
