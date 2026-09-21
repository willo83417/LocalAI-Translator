
// Polyfill SharedArrayBuffer if not defined in worker scope
// to prevent esearch-ocr from crashing on `n instanceof SharedArrayBuffer`
if (typeof (globalThis as any).SharedArrayBuffer === 'undefined') {
    (globalThis as any).SharedArrayBuffer = ArrayBuffer;
}

import * as ocr from 'esearch-ocr';
import * as ort from 'onnxruntime-web/webgpu';
import { getFromDB, setInDB } from '../utils/db';

let ocrInstance: any = null;

/**
 * Extract embedded character dictionary from ONNX model metadata (key: 'character')
 * Used by PP-OCRv6 models where characters are embedded directly in the ONNX metadata_props.
 */
function extractDictFromOnnx(buffer: ArrayBuffer): string {
    try {
        const u8 = new Uint8Array(buffer);
        const target = new TextEncoder().encode('character');
        let found = -1;
        // Search backwards from the end of the buffer where metadata_props are stored
        for (let i = u8.length - target.length; i >= 0; i--) {
            let match = true;
            for (let j = 0; j < target.length; j++) {
                if (u8[i + j] !== target[j]) {
                    match = false;
                    break;
                }
            }
            if (match) {
                // Protobuf StringStringEntryProto: tag 1 (0x0a), length 9 (0x09) before 'character'
                if (i >= 2 && u8[i - 2] === 0x0a && u8[i - 1] === 0x09) {
                    found = i;
                    break;
                }
            }
        }
        if (found === -1) return '';

        // In StringStringEntryProto:
        // tag 2 (0x12) is value (string)
        let pos = found + target.length;
        if (u8[pos] === 0x12) {
            pos++;
            let len = 0;
            let shift = 0;
            while (pos < u8.length) {
                const b = u8[pos++];
                len |= (b & 0x7f) << shift;
                if ((b & 0x80) === 0) break;
                shift += 7;
            }
            if (len > 0 && pos + len <= u8.length) {
                const valBytes = u8.subarray(pos, pos + len);
                return new TextDecoder('utf-8').decode(valBytes);
            }
        }
    } catch (e) {
        console.warn('Failed to extract embedded character dictionary from ONNX metadata:', e);
    }
    return '';
}

const post = (type: string, payload: any) => self.postMessage({ type, payload });

self.onmessage = async (e: MessageEvent<any>) => {
    const { type, payload } = e.data;

    switch (type) {
        case 'load':
            try {
                const { key, detPath, recPath, dictPath } = payload;
                post('status', 'initializing');

                // Differentiate V6 from V5: V6 models (such as PP_v6_small) do not require a separate dictionary file
                const isV6 = Boolean(key && (key.toLowerCase().includes('v6') || !dictPath));

                const DET_MODEL_KEY = isV6 ? `det-model-${key}` : 'det-model';
                const REC_MODEL_KEY = `rec-model-${key}`;
                const DICT_KEY = `dict-${key}`;

                let detBuffer = await getFromDB<ArrayBuffer>(DET_MODEL_KEY);
                let recBuffer = await getFromDB<ArrayBuffer>(REC_MODEL_KEY);
                let dictText: string | undefined = undefined;

                const fetchPromises: Promise<void>[] = [];

                if (!detBuffer) {
                    fetchPromises.push(
                        fetch(detPath).then(async res => {
                            if (!res.ok) throw new Error(`HTTP ${res.status} fetching det model: ${detPath}`);
                            const b = await res.arrayBuffer();
                            detBuffer = b;
                            await setInDB(DET_MODEL_KEY, b);
                        })
                    );
                }
                if (!recBuffer) {
                    fetchPromises.push(
                        fetch(recPath).then(async res => {
                            if (!res.ok) throw new Error(`HTTP ${res.status} fetching rec model: ${recPath}`);
                            const b = await res.arrayBuffer();
                            recBuffer = b;
                            await setInDB(REC_MODEL_KEY, b);
                        })
                    );
                }

                // V5 requires fetching dictPath; V6 does not use an external dict file
                if (!isV6) {
                    dictText = await getFromDB<string>(DICT_KEY);
                    // Discard corrupted HTML cache
                    if (dictText && (dictText.includes('<!DOCTYPE') || dictText.includes('<html') || dictText.includes('id="root"'))) {
                        dictText = undefined;
                    }
                    if (!dictText && dictPath) {
                        fetchPromises.push(
                            fetch(dictPath).then(async res => {
                                if (!res.ok) throw new Error(`HTTP ${res.status} fetching dictionary: ${dictPath}`);
                                const b = await res.arrayBuffer();
                                const decoder = new TextDecoder('utf-8');
                                const text = decoder.decode(b);
                                if (text.includes('<!DOCTYPE') || text.includes('<html')) {
                                    throw new Error(`Received HTML page instead of valid dictionary from ${dictPath}`);
                                }
                                dictText = text;
                                await setInDB(DICT_KEY, dictText);
                            })
                        );
                    }
                }

                if (fetchPromises.length > 0) await Promise.all(fetchPromises);

                if (isV6) {
                    // For V6, character dictionary is embedded in the ONNX model metadata (key: 'character')
                    // Always extract directly from recBuffer to ensure it is not corrupted by stale HTML fallbacks
                    if (recBuffer) {
                        dictText = extractDictFromOnnx(recBuffer);
                    }
                    if (dictText) {
                        await setInDB(DICT_KEY, dictText);
                    } else {
                        // Fallback: check DB if valid, or empty string
                        const cached = await getFromDB<string>(DICT_KEY);
                        if (cached && !cached.includes('<!DOCTYPE') && !cached.includes('<html')) {
                            dictText = cached;
                        } else {
                            dictText = '';
                        }
                    }
                }

                const ortInstance: any = (ort as any).default || ort;
                
                // Configure ONNX Runtime inside worker
                ortInstance.env.wasm.numThreads = 0;
                ortInstance.env.wasm.simd = true;
                ortInstance.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/';
                
                // Set OCR environment for Worker (OffscreenCanvas)
                (ocr as any).setOCREnv({
                    canvas: (w: number, h: number) => new OffscreenCanvas(w, h),
                    imageData: (data: Uint8ClampedArray, w: number, h: number) => new ImageData(data, w, h)
                });

                ocrInstance = await (ocr as any).init({
                    det: { input: new Uint8Array(detBuffer!) },
                    rec: {
                        input: new Uint8Array(recBuffer!),
                        decodeDic: dictText || '',
                        optimize: { space: false },
                        on: (index: number, result: any, total: number) => {
                            const text = result.map((c: any) => c[0]?.t || '').join('');
                            if (text.trim()) {
                                post('recognize_chunk', text);
                            }
                        }
                    },
                    ort: ortInstance,
                    ortOption: {
						//executionProviders: ['webgpu']
                        executionProviders: ['webgpu', 'wasm']
                    }
                });

                post('status', 'ready');
            } catch (err: any) {
                console.error('OCR Worker Load Error:', err);
                post('error', err.message || 'Initialization failed');
            }
            break;

        case 'recognize':
            if (!ocrInstance) {
                post('error', 'OCR Instance not initialized');
                return;
            }
            try {
                const { imageData, imageBitmap } = payload;
                let finalInput = imageData;

                if (imageBitmap) {
                    const canvas = new OffscreenCanvas(imageBitmap.width, imageBitmap.height);
                    const ctx = canvas.getContext('2d');
                    if (!ctx) throw new Error('Worker canvas context error');
                    ctx.drawImage(imageBitmap, 0, 0);
                    finalInput = ctx.getImageData(0, 0, canvas.width, canvas.height);
                    imageBitmap.close(); // Clean up
                }

                if (!finalInput) {
                    throw new Error('No input image provided to worker');
                }

                const startTime = performance.now();
                const result = await ocrInstance.ocr(finalInput);
                const endTime = performance.now();
                post('result', { result, time: endTime - startTime });
            } catch (err: any) {
                console.error('OCR Worker Recognition Error:', err);
                post('error', err.message || 'Recognition failed');
            }
            break;
            
        case 'unload':
            ocrInstance = null;
            post('status', 'uninitialized');
            break;
    }
};
