// services/downloadManager.ts
import { openDB, IDBPDatabase } from 'idb';
import { ModelRegistry } from '@huggingface/transformers';

const DB_NAME = 'offline-model-db';
const CHUNKS_STORE = 'model-chunks';
const META_STORE = 'model-meta';
const CONSOLIDATED_STORE = 'consolidated-models';
const DB_VERSION = 3;

export type DownloadStatus = 'not_started' | 'downloading' | 'paused' | 'completed' | 'error' | 'consolidating';

export interface DownloadProgress {
    downloaded: number;
    total: number;
    percent: number;
    status: DownloadStatus;
    error?: string;
}

let cachedShaderF16Support: boolean | null = null;

/**
 * 檢測當前瀏覽器與設備 WebGPU 是否支援 shader-f16 (半精度 16-bit 浮點數著色器運算)
 */
export async function checkShaderF16Support(): Promise<boolean> {
    if (cachedShaderF16Support !== null) {
        return cachedShaderF16Support;
    }
    if (typeof navigator === 'undefined' || !('gpu' in navigator) || !navigator.gpu) {
        cachedShaderF16Support = false;
        return false;
    }
    try {
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) {
            cachedShaderF16Support = false;
            return false;
        }
        cachedShaderF16Support = adapter.features.has('shader-f16');
        return cachedShaderF16Support;
    } catch (e) {
        console.warn('[DownloadManager] Error checking WebGPU shader-f16 support:', e);
        cachedShaderF16Support = false;
        return false;
    }
}

class DownloadManager {
    private dbPromise: Promise<IDBPDatabase>;
    private controllers: Map<string, AbortController> = new Map();
    private readonly isOpfsSupported: boolean;
    private opfsRootPromise: Promise<FileSystemDirectoryHandle | null>;
    
    constructor() {
        this.dbPromise = openDB(DB_NAME, DB_VERSION, {
            upgrade(db) {
                if (!db.objectStoreNames.contains(CHUNKS_STORE)) {
                     const store = db.createObjectStore(CHUNKS_STORE, { keyPath: ['modelName', 'chunkIndex'] });
                     store.createIndex('modelName', 'modelName', { unique: false });
                }
                 if (!db.objectStoreNames.contains(META_STORE)) {
                    db.createObjectStore(META_STORE, { keyPath: 'modelName' });
                }
                if (!db.objectStoreNames.contains(CONSOLIDATED_STORE)) {
                    db.createObjectStore(CONSOLIDATED_STORE, { keyPath: 'modelName' });
                }
            },
        });

        this.isOpfsSupported = 'getDirectory' in navigator.storage;
        this.opfsRootPromise = this.isOpfsSupported ? navigator.storage.getDirectory().catch(err => {
            console.error("Failed to get OPFS root directory:", err);
            return null;
        }) : Promise.resolve(null);

        if (this.isOpfsSupported) {
            console.log('Origin Private File System is supported and will be used for model storage.');
        } else {
            console.log('Origin Private File System not supported, falling back to IndexedDB.');
        }
    }

    private async getOpfsRoot(): Promise<FileSystemDirectoryHandle | null> {
        return this.opfsRootPromise;
    }

    /**
     * 依據設備 WebGPU shader-f16 支援能力，解析實際可用的 dtype：
     * 1. 若支援 shader-f16：保留原設定 (如 q4f16, q2f16, fp16 等)
     * 2. 若不支援：
     *    - Gemma QAT 模型 (Gemma-4-E2B-it-qat, Gemma-4-E4B-it-qat) 僅支援 f16，彈出警告並拒絕下載
     *    - 其他模型 (如 Gemma-4-E2B-it, Qwen3.5-4B 等) 若指定了含有 f16 的量化，自動降級為 q4 確保正常執行
     */
    async resolveDtypeForModel(
        modelName: string, 
        dtype: string | Record<string, string>
    ): Promise<string | Record<string, string>> {
        const supportsF16 = await checkShaderF16Support();
        const isQatModel = modelName.toLowerCase().includes('qat');

        if (!supportsF16) {
            if (isQatModel) {
                const warningMsg = `此裝置的 WebGPU 不支援「shader-f16」(半精度浮點運算)。模型「${modelName}」僅支援 f16 量化，無法在此裝置上下載與運行。`;
                try {
                    if (typeof window !== 'undefined' && typeof window.alert === 'function') {
                        window.alert(warningMsg);
                    }
                } catch (_) {}
                throw new Error(warningMsg);
            }

            // 一般模型降級為基礎 q4，以保證可以在無 shader-f16 設備上執行
            if (typeof dtype === 'string') {
                if (dtype.includes('f16')) {
                    console.info(`[DownloadManager] WebGPU shader-f16 not supported. Falling back dtype from '${dtype}' to 'q4' for ${modelName}.`);
                    return 'q4';
                }
            } else if (dtype && typeof dtype === 'object') {
                console.info(`[DownloadManager] WebGPU shader-f16 not supported. Falling back composite dtype to 'q4' for ${modelName}.`);
                return 'q4';
            }
        }

        return dtype;
    }

    async startTSModelDownload(
        modelName: string, 
        dtype: string | Record<string, string>, 
        hfApiKey: string, 
        onProgress: (progress: DownloadProgress) => void
    ): Promise<void> {
        if (this.controllers.has(modelName)) return;

        // 下載前檢測 WebGPU shader-f16 支援度與解析實際使用的 dtype
        let resolvedDtype: string | Record<string, string>;
        try {
            resolvedDtype = await this.resolveDtypeForModel(modelName, dtype);
        } catch (checkErr: any) {
            onProgress({ downloaded: 0, total: 0, percent: 0, status: 'error', error: checkErr.message });
            throw checkErr;
        }

        const controller = new AbortController();
        this.controllers.set(modelName, controller);
        
        try {
            onProgress({ downloaded: 0, total: 0, percent: 0, status: 'downloading' });

            const targetFiles = new Set<string>();

            // 1. 使用 Transformers.js ModelRegistry.get_files 解析核心檔案與 ONNX 外部權重分片 (.onnx_data, .onnx_data_1 等)
            try {
                const registryFiles = await ModelRegistry.get_files(modelName, { dtype: resolvedDtype as any });
                if (Array.isArray(registryFiles)) {
                    for (const f of registryFiles) {
                        targetFiles.add(f);
                    }
                }
            } catch (regErr) {
                console.warn(`[DownloadManager] ModelRegistry.get_files failed for ${modelName}:`, regErr);
            }

            // 2. 透過 Hugging Face Tree API 獲取遠端結構，確保所有配置與模板 (chat_template.jinja, processor_config.json 等)
            const authHeaders: Record<string, string> = hfApiKey ? { Authorization: `Bearer ${hfApiKey}` } : {};

            try {
                const rootRes = await fetch(`https://huggingface.co/api/models/${modelName}/tree/main`, { 
                    headers: authHeaders, 
                    signal: controller.signal 
                });
                if (rootRes.ok) {
                    const rootFiles = await rootRes.json();
                    for (const f of rootFiles) {
                        // 包含 json, jinja, txt 等元資料檔案
                        if (f.path.endsWith('.json') || f.path.endsWith('.jinja') || f.path.endsWith('.txt')) {
                            targetFiles.add(f.path);
                        }
                    }
                }

                // 檢查 onnx 目錄，補足當前 dtype 的所有 ONNX 模型與 .onnx_data 權重分片
                const onnxRes = await fetch(`https://huggingface.co/api/models/${modelName}/tree/main/onnx`, { 
                    headers: authHeaders, 
                    signal: controller.signal 
                });
                if (onnxRes.ok) {
                    const onnxFiles = await onnxRes.json();
                    const dtypes = typeof resolvedDtype === 'string' ? [resolvedDtype] : Object.values(resolvedDtype);
                    for (const f of onnxFiles) {
                        const matchesDtype = dtypes.some(dt => f.path.includes(`_${dt}.onnx`) || f.path.includes(`_${dt}.onnx_data`));
                        if (matchesDtype) {
                            targetFiles.add(`onnx/${f.path}`);
                        }
                    }
                }
            } catch (treeErr) {
                console.warn(`[DownloadManager] HF Tree API fetch failed, proceeding with known registry files:`, treeErr);
            }

            // 3. 確保 Gemma / Qwen 等關鍵配置檔案至少進入候選測試清單
            const essentialCandidates = [
                'config.json',
                'generation_config.json',
                'tokenizer.json',
                'tokenizer_config.json',
                'chat_template.jinja',
                'processor_config.json',
                'preprocessor_config.json'
            ];
            for (const item of essentialCandidates) {
                targetFiles.add(item);
            }

            const cache = await caches.open('transformers-cache');
            const fileList = Array.from(targetFiles);
            const fileEntries: { url: string; size: number; cached: boolean }[] = [];

            // 4. 解析各檔案大小與快取狀態
            for (const filePath of fileList) {
                const url = `https://huggingface.co/${modelName}/resolve/main/${filePath}`;

                // 檢查快取
                const cachedRes = await cache.match(url);
                if (cachedRes) {
                    const cachedLength = parseInt(cachedRes.headers.get('content-length') || '0', 10);
                    if (cachedLength > 0) {
                        fileEntries.push({ url, size: cachedLength, cached: true });
                        continue;
                    }
                }

                // 檢查遠端是否存在以及大小 (HEAD request)
                try {
                    const headRes = await fetch(url, { 
                        method: 'HEAD', 
                        headers: authHeaders, 
                        signal: controller.signal 
                    });
                    if (headRes.ok) {
                        const size = parseInt(headRes.headers.get('content-length') || '0', 10);
                        fileEntries.push({ url, size, cached: false });
                    }
                } catch {
                    // 若 HEAD 失敗或不存在，略過此候選檔案
                }
            }

            let grandTotal = fileEntries.reduce((sum, item) => sum + item.size, 0);
            let grandDownloaded = fileEntries.filter(i => i.cached).reduce((sum, item) => sum + item.size, 0);

            if (grandTotal === 0 && fileEntries.length > 0) {
                grandTotal = fileEntries.length * 1024 * 1024;
            }

            onProgress({ 
                downloaded: grandDownloaded, 
                total: grandTotal, 
                percent: grandTotal > 0 ? (grandDownloaded / grandTotal) * 100 : 0, 
                status: 'downloading' 
            });

            // 5. 進行下載並存入 Cache API (包含 content-length header)
            for (const fileEntry of fileEntries) {
                if (fileEntry.cached) continue;

                const response = await fetch(fileEntry.url, { 
                    headers: authHeaders, 
                    signal: controller.signal 
                });
                if (!response.ok || !response.body) {
                    console.warn(`[DownloadManager] Failed to fetch ${fileEntry.url}, status: ${response.status}`);
                    continue;
                }

                const contentLengthHeader = response.headers.get('content-length');
                const realSize = contentLengthHeader ? parseInt(contentLengthHeader, 10) : fileEntry.size;
                const reader = response.body.getReader();
                const headers = new Headers(response.headers);
                if (!headers.has('content-length') && realSize > 0) {
                    headers.set('content-length', realSize.toString());
                }

                const stream = new ReadableStream({
                    async start(ctrl) {
                        while (true) {
                            const { done, value } = await reader.read();
                            if (done) {
                                ctrl.close();
                                break;
                            }
                            grandDownloaded += value.length;
                            onProgress({ 
                                downloaded: grandDownloaded, 
                                total: grandTotal, 
                                percent: grandTotal > 0 ? Math.min(99.9, (grandDownloaded / grandTotal) * 100) : 0, 
                                status: 'downloading' 
                            });
                            ctrl.enqueue(value);
                        }
                    }
                });

                const streamResponse = new Response(stream, { headers });
                await cache.put(fileEntry.url, streamResponse);
            }
            
            const db = await this.getDb();
            await db.put(META_STORE, { modelName, total: grandTotal, status: 'completed' });
            onProgress({ downloaded: grandTotal, total: grandTotal, percent: 100, status: 'completed' });
            //console.log(`TS Model ${modelName} pre-download completed successfully using ModelRegistry & Cache API.`);
            
        } catch (error: any) {
            console.error('Download error:', error);
            if (error.name === 'AbortError') {
                onProgress({ downloaded: 0, total: 0, percent: 0, status: 'paused' });
            } else {
                onProgress({ downloaded: 0, total: 0, percent: 0, status: 'error', error: error.message });
            }
        } finally {
            this.controllers.delete(modelName);
        }
    }

    async startDownload(modelName: string, url: string, hfApiKey: string, onProgress: (progress: DownloadProgress) => void): Promise<void> {
        if (this.controllers.has(modelName)) {
            console.warn(`Download for ${modelName} is already in progress.`);
            return;
        }
        if (this.isOpfsSupported) {
            await this.runOpfsDownload(modelName, url, hfApiKey, onProgress, false);
        } else {
            await this.runIdbDownload(modelName, url, hfApiKey, onProgress, false);
        }
    }
    
    async resumeDownload(modelName: string, url: string, hfApiKey: string, onProgress: (progress: DownloadProgress) => void): Promise<void> {
        if (this.controllers.has(modelName)) return;
        if (this.isOpfsSupported) {
            await this.runOpfsDownload(modelName, url, hfApiKey, onProgress, true);
        } else {
            await this.runIdbDownload(modelName, url, hfApiKey, onProgress, true);
        }
    }

    private async runOpfsDownload(modelName: string, url: string, hfApiKey: string, onProgress: (progress: DownloadProgress) => void, isResume: boolean): Promise<void> {
        /*
        if (!hfApiKey) {
            onProgress({ downloaded: 0, total: 0, percent: 0, status: 'error', error: 'Hugging Face API Key is required.' });
            return;
        }
        */

        const controller = new AbortController();
        this.controllers.set(modelName, controller);
        
        let downloaded = 0;
        let total = 0;
    
        try {
            const root = await this.getOpfsRoot();
            if (!root) throw new Error("Could not access OPFS.");
    
            const fileHandle = await root.getFileHandle(modelName, { create: true });
            const db = await this.getDb();
            const meta = await db.get(META_STORE, modelName);
    
            if (isResume && meta && meta.total > 0) {
                total = meta.total;
                const file = await fileHandle.getFile();
                downloaded = file.size;
                 // If file is larger than expected (corrupt), restart download
                if (downloaded >= total) {
                    console.warn(`Resuming download for ${modelName}, but existing file is complete or corrupt. Restarting.`);
                    downloaded = 0;
                }
            }
            
            // If total is still unknown, fetch it
            if (total === 0) {
                /*
                const headResponse = await fetch(url, { 
                    method: 'HEAD',
                    headers: { 'Authorization': `Bearer ${hfApiKey}` },
                    signal: controller.signal 
                });
                */
                const headResponse = await fetch(url, { 
                    method: 'HEAD',
                    signal: controller.signal 
                });
                if (!headResponse.ok) throw new Error(`Failed to get model info. Status: ${headResponse.status}`);
                total = Number(headResponse.headers.get('Content-Length'));
                if (isNaN(total) || total === 0) throw new Error('Could not determine file size.');
                downloaded = 0;
            }
            
            await db.put(META_STORE, { modelName, total, status: 'downloading' });
            onProgress({ downloaded, total, percent: total > 0 ? (downloaded / total) * 100 : 0, status: 'downloading' });
    
            if (downloaded < total) {
                /*
                const response = await fetch(url, {
                    headers: { 'Range': `bytes=${downloaded}-`, 'Authorization': `Bearer ${hfApiKey}` },
                    signal: controller.signal,
                });
                */
                const response = await fetch(url, {
                    headers: { 'Range': `bytes=${downloaded}-` },
                    signal: controller.signal,
                });
    
                if (!response.ok || !response.body) throw new Error(`Download request failed. Status: ${response.status}`);
                
                const writable = await fileHandle.createWritable({ keepExistingData: true });
                await writable.seek(downloaded);
    
                const reader = response.body.getReader();
                while(true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    await writable.write(value);
                    downloaded += value.length;
                    onProgress({ downloaded, total, percent: (downloaded / total) * 100, status: 'downloading' });
                }
                await writable.close();
            }
            
            const finalFile = await fileHandle.getFile();
            if (finalFile.size !== total) {
                throw new Error(`Final file size ${finalFile.size} does not match expected size ${total}`);
            }
            
            await db.put(META_STORE, { modelName, total, status: 'completed' });
            onProgress({ downloaded: total, total, percent: 100, status: 'completed' });
            console.log(`Model ${modelName} downloaded to OPFS successfully.`);
    
        } catch (error) {
            this.handleDownloadError(error, modelName, downloaded, total, onProgress);
        } finally {
            this.controllers.delete(modelName);
        }
    }
    
    private async runIdbDownload(modelName: string, url: string, hfApiKey: string, onProgress: (progress: DownloadProgress) => void, isResume: boolean): Promise<void> {
        /*
        if (!hfApiKey) {
            onProgress({ downloaded: 0, total: 0, percent: 0, status: 'error', error: 'Hugging Face API Key is required.' });
            return;
        }
        */

        const controller = new AbortController();
        this.controllers.set(modelName, controller);
        
        let downloaded = 0;
        let total = 0;
        let chunkIndex = 0;
        const CHUNK_SIZE = 5 * 1024 * 1024; // 5MB

        try {
            const db = await this.getDb();
            if (isResume) {
                const meta = await db.get(META_STORE, modelName);
                if (!meta || !meta.total) {
                    // If no metadata, treat as a new download
                    return this.runIdbDownload(modelName, url, hfApiKey, onProgress, false);
                }
                total = meta.total;
                const chunks = await db.getAllFromIndex(CHUNKS_STORE, 'modelName', modelName);
                downloaded = chunks.reduce((acc, chunk) => acc + chunk.data.byteLength, 0);
                chunkIndex = chunks.length;
            } else {
                 await this.deleteModel(modelName); // Clear any old data before a fresh start
                 /*
                 const headResponse = await fetch(url, { 
                    method: 'HEAD',
                    headers: { 'Authorization': `Bearer ${hfApiKey}` },
                    signal: controller.signal 
                });
                */
                const headResponse = await fetch(url, { 
                    method: 'HEAD',
                    signal: controller.signal 
                });
                if (!headResponse.ok) throw new Error(`Failed to get model info. Status: ${headResponse.status}`);
                total = Number(headResponse.headers.get('Content-Length'));
                if (isNaN(total) || total === 0) throw new Error('Could not determine file size.');
            }

            await db.put(META_STORE, { modelName, total, status: 'downloading' });
            
            while (downloaded < total) {
                const start = downloaded;
                const end = Math.min(downloaded + CHUNK_SIZE - 1, total - 1);

                /*
                const chunkResponse = await fetch(url, {
                    headers: { 'Range': `bytes=${start}-${end}`, 'Authorization': `Bearer ${hfApiKey}` },
                    signal: controller.signal,
                });
                */
                const chunkResponse = await fetch(url, {
                    headers: { 'Range': `bytes=${start}-${end}` },
                    signal: controller.signal,
                });

                if (!chunkResponse.ok) throw new Error(`Chunk download failed. Status: ${chunkResponse.status}`);
                
                const chunk = await chunkResponse.arrayBuffer();
                await db.put(CHUNKS_STORE, { modelName, chunkIndex, data: chunk });
                
                downloaded += chunk.byteLength;
                chunkIndex++;
                
                onProgress({ downloaded, total, percent: (downloaded / total) * 100, status: 'downloading' });
            }
            
            await this._consolidateModel(modelName, onProgress);

        } catch (error) {
            this.handleDownloadError(error, modelName, downloaded, total, onProgress);
        } finally {
            this.controllers.delete(modelName);
        }
    }
    
    private handleDownloadError(error: unknown, modelName: string, downloaded: number, total: number, onProgress: (progress: DownloadProgress) => void) {
        if (error instanceof DOMException && error.name === 'AbortError') {
            console.log(`Download for ${modelName} was paused.`);
            this.getDb().then(db => db.put(META_STORE, { modelName, total, status: 'paused', downloaded }));
            onProgress({ downloaded, total, percent: total > 0 ? (downloaded / total) * 100 : 0, status: 'paused' });
        } else {
            const errorMessage = error instanceof Error ? error.message : 'Unknown download error';
            console.error(`Download error for ${modelName}:`, error);
            this.getDb().then(db => db.put(META_STORE, { modelName, status: 'error', error: errorMessage, total }));
            onProgress({ downloaded, total, percent: total > 0 ? (downloaded / total) * 100 : 0, status: 'error', error: errorMessage });
        }
    }
    
    private async _consolidateModel(modelName: string, onProgress: (progress: DownloadProgress) => void): Promise<void> {
        const db = await this.getDb();
        const meta = await db.get(META_STORE, modelName);
        if (!meta) return;
        const total = meta.total || 0;
        const downloaded = meta.downloaded || total;

        try {
            onProgress({ downloaded, total, percent: 100, status: 'consolidating' });
            
            const consolidatedBlob = await this.getModelFromChunksAsStream(modelName);
            if (!consolidatedBlob) throw new Error("Failed to reconstruct model from chunks.");

            await db.put(CONSOLIDATED_STORE, { modelName, blob: consolidatedBlob });
            await this._deleteChunks(modelName);

            await db.put(META_STORE, { modelName, total, status: 'completed', downloaded: total });
            onProgress({ downloaded: total, total, percent: 100, status: 'completed' });
            console.log(`Model ${modelName} consolidated successfully.`);
        } catch(error) {
             const errorMessage = error instanceof Error ? error.message : 'Unknown consolidation error';
             console.error(`Consolidation error for ${modelName}:`, error);
             await db.put(META_STORE, { modelName, status: 'error', error: errorMessage, total });
             onProgress({ downloaded, total, percent: (downloaded/total)*100, status: 'error', error: `Consolidation failed: ${errorMessage}` });
        }
    }

    private async _deleteChunks(modelName: string): Promise<void> {
        const db = await this.getDb();
        const tx = db.transaction(CHUNKS_STORE, 'readwrite');
        const index = tx.objectStore(CHUNKS_STORE).index('modelName');
        let cursor = await index.openCursor(IDBKeyRange.only(modelName));
        while(cursor) {
            cursor.delete();
            cursor = await cursor.continue();
        }
        await tx.done;
    }
    
    pauseDownload(modelName: string): void {
        const controller = this.controllers.get(modelName);
        if (controller) {
            controller.abort();
            this.controllers.delete(modelName);
        }
    }

    async deleteModel(modelName: string): Promise<void> {
        this.pauseDownload(modelName);

        // Delete from Cache API (Transformers-cache)
        try {
            const cache = await caches.open('transformers-cache');
            const keys = await cache.keys();
            for (const key of keys) {
                if (key.url.includes(modelName)) {
                    await cache.delete(key);
                    console.log(`[Cache API] Deleted cached file: ${key.url}`);
                }
            }
        } catch (e) {
            console.error(`Error deleting ${modelName} from Cache API:`, e);
        }

        if (this.isOpfsSupported) {
            try {
                const root = await this.getOpfsRoot();
                if (root) {
                    try {
                        await root.removeEntry(modelName);
                    } catch (e) { /* ignore if not found */ }

                    try {
                        const cacheRoot = await root.getDirectoryHandle('transformers-cache', { create: false }).catch(() => null);
                        if (cacheRoot) {
                            const prefix = modelName.replace(/\//g, '_');
                            const filesToDelete: string[] = [];
                            for await (const name of (cacheRoot as any).keys()) {
                                if (name.startsWith(prefix)) {
                                    filesToDelete.push(name);
                                }
                            }
                            for (const fileName of filesToDelete) {
                                await cacheRoot.removeEntry(fileName);
                                console.log(`[OPFS] Deleted cached file: ${fileName}`);
                            }
                        }
                    } catch (e) { /* ignore */ }
                }
            } catch (error) {
                if (!(error instanceof DOMException && error.name === 'NotFoundError')) {
                    console.error(`Error deleting ${modelName} from OPFS:`, error);
                }
            }
        }
        
        const db = await this.getDb();
        await db.delete(META_STORE, modelName);
        await db.delete(CONSOLIDATED_STORE, modelName);
        await this._deleteChunks(modelName);
    }

    async checkTSModelCached(modelName: string, dtype: string | Record<string, string> = 'q4'): Promise<boolean> {
        try {
            const resolvedDtype = await this.resolveDtypeForModel(modelName, dtype).catch(() => dtype);

            // 1. 優先使用 Transformers.js 官方 ModelRegistry.is_cached 判定
            const cachedByRegistry = await ModelRegistry.is_cached(modelName, { dtype: resolvedDtype as any }).catch(() => null);
            if (cachedByRegistry === true) {
                return true;
            }

            // 2. 次選檢查 Cache API ('transformers-cache') 中的關鍵設定檔
            const cache = await caches.open('transformers-cache');
            const keys = await cache.keys();
            const urls = keys.map(k => k.url);

            const hasConfig = urls.some(u => u.includes(modelName) && u.endsWith('config.json'));
            const hasTokenizer = urls.some(u => u.includes(modelName) && (u.endsWith('tokenizer.json') || u.endsWith('tokenizer_config.json')));
            const hasOnnx = urls.some(u => u.includes(modelName) && u.includes('.onnx'));

            return hasConfig && hasTokenizer && hasOnnx;
        } catch (err) {
            console.error(`Error checking TS model cache for ${modelName}:`, err);
            return false;
        }
    }

    async getStatus(modelName: string, dtype?: string | Record<string, string>): Promise<DownloadProgress> {
        const db = await this.getDb();
        const meta = await db.get(META_STORE, modelName);
    
        if (!meta) {
            // 對於 TS 模型，若 META_STORE 無記錄，檢查 Cache API 是否早已下載過
            if (modelName.includes('/')) {
                const isCached = await this.checkTSModelCached(modelName, dtype);
                if (isCached) {
                    return { downloaded: 1, total: 1, percent: 100, status: 'completed' };
                }
            }
            return { downloaded: 0, total: 0, percent: 0, status: 'not_started' };
        }
        
        const total = meta.total || 0;
    
        if (meta.status === 'completed') {
            // 若為 TS 模型，防禦性檢查快取是否被外部清理
            if (modelName.includes('/')) {
                try {
                    const cache = await caches.open('transformers-cache');
                    const keys = await cache.keys();
                    const hasModelFiles = keys.some(k => k.url.includes(modelName));
                    if (!hasModelFiles) {
                        await db.delete(META_STORE, modelName);
                        return { downloaded: 0, total: 0, percent: 0, status: 'not_started' };
                    }
                } catch { /* ignore */ }
            }
            return { downloaded: total, total, percent: 100, status: 'completed' };
        }

        let downloaded = 0;
        if (this.isOpfsSupported) {
            try {
                const root = await this.getOpfsRoot();
                if (root) {
                    const fileHandle = await root.getFileHandle(modelName);
                    const file = await fileHandle.getFile();
                    downloaded = file.size;
                }
            } catch (e) { /* File not found is expected */ }
        } else {
             const chunks = await db.getAllFromIndex(CHUNKS_STORE, 'modelName', modelName);
             downloaded = chunks.reduce((acc, chunk) => acc + chunk.data.byteLength, 0);
        }
    
        return { downloaded, total, percent: total > 0 ? (downloaded / total) * 100 : 0, status: meta.status, error: meta.error };
    }

    async getModelAsBlob(modelName: string): Promise<Blob | null> {
        const db = await this.getDb();
        const meta = await db.get(META_STORE, modelName);
        if (!meta || meta.status !== 'completed') return null;
    
        if (this.isOpfsSupported) {
            try {
                const root = await this.getOpfsRoot();
                if(root) {
                    const fileHandle = await root.getFileHandle(modelName);
                    const file = await fileHandle.getFile();
                    if (file.size === meta.total) {
                        return file;
                    } else {
                         console.error(`OPFS file for ${modelName} is incomplete. Expected ${meta.total}, found ${file.size}.`);
                         return null;
                    }
                }
            } catch (e) {
                console.warn(`Could not get model ${modelName} from OPFS, will check IndexedDB.`, e);
            }
        }
    
        const consolidatedModel = await db.get(CONSOLIDATED_STORE, modelName);
        if (consolidatedModel && consolidatedModel.blob) {
            return consolidatedModel.blob;
        }
    
        console.warn(`Consolidated model for ${modelName} not found in store. Reconstructing from chunks as a fallback.`);
        return this.getModelFromChunksAsStream(modelName);
    }

    private async getModelFromChunksAsStream(modelName: string): Promise<Blob | null> {
        const db = await this.getDb();
        // Check if chunks even exist to avoid creating an empty blob
        const firstChunk = await db.getFromIndex(CHUNKS_STORE, 'modelName', modelName);
        if (!firstChunk) return null;

        const stream = new ReadableStream({
            async start(controller) {
                const tx = db.transaction(CHUNKS_STORE, 'readonly');
                const index = tx.objectStore(CHUNKS_STORE).index('modelName');
                let cursor = await index.openCursor(IDBKeyRange.only(modelName));
                while (cursor) {
                    controller.enqueue(new Uint8Array(cursor.value.data));
                    cursor = await cursor.continue();
                }
                controller.close();
            }
        });

        try {
            return await new Response(stream).blob();
        } catch {
            return null;
        }
    }
    
    private getDb(): Promise<IDBPDatabase> {
        return this.dbPromise;
    }
}

export const downloadManager = new DownloadManager();