
// Polyfill SharedArrayBuffer if not defined in worker scope
// to prevent esearch-ocr from crashing on `n instanceof SharedArrayBuffer`
if (typeof (globalThis as any).SharedArrayBuffer === 'undefined') {
    (globalThis as any).SharedArrayBuffer = ArrayBuffer;
}

import * as ocr from 'esearch-ocr';
import * as ort from 'onnxruntime-web/webgpu';
import { getFromDB, setInDB } from '../utils/db';

let ocrInstance: any = null;

const post = (type: string, payload: any) => self.postMessage({ type, payload });

self.onmessage = async (e: MessageEvent<any>) => {
    const { type, payload } = e.data;

    switch (type) {
        case 'load':
            try {
                const { key, detPath, recPath, dictPath } = payload;
                post('status', 'initializing');

                const DET_MODEL_KEY = 'det-model';
                const REC_MODEL_KEY = `rec-model-${key}`;
                const DICT_KEY = `dict-${key}`;

                let detBuffer = await getFromDB<ArrayBuffer>(DET_MODEL_KEY);
                let recBuffer = await getFromDB<ArrayBuffer>(REC_MODEL_KEY);
                let dictText = await getFromDB<string>(DICT_KEY);

                const fetchPromises: Promise<void>[] = [];

                if (!detBuffer) {
                    fetchPromises.push(
                        fetch(detPath).then(res => res.arrayBuffer()).then(async b => { detBuffer = b; await setInDB(DET_MODEL_KEY, b); })
                    );
                }
                if (!recBuffer) {
                    fetchPromises.push(
                        fetch(recPath).then(res => res.arrayBuffer()).then(async b => { recBuffer = b; await setInDB(REC_MODEL_KEY, b); })
                    );
                }
                if (!dictText) {
                    fetchPromises.push(
                        fetch(dictPath).then(res => res.arrayBuffer()).then(async b => {
                            const decoder = new TextDecoder('utf-8');
                            dictText = decoder.decode(b);
                            await setInDB(DICT_KEY, dictText);
                        })
                    );
                }

                if (fetchPromises.length > 0) await Promise.all(fetchPromises);

                const ortInstance: any = (ort as any).default || ort;
                
                // Configure ONNX Runtime inside worker
                ortInstance.env.wasm.numThreads = 0;
                ortInstance.env.wasm.simd = true;
                ortInstance.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.29.0/dist/';
                
                // Set OCR environment for Worker (OffscreenCanvas)
                (ocr as any).setOCREnv({
                    canvas: (w: number, h: number) => new OffscreenCanvas(w, h),
                    imageData: (data: Uint8ClampedArray, w: number, h: number) => new ImageData(data, w, h)
                });

                ocrInstance = await (ocr as any).init({
                    det: { input: new Uint8Array(detBuffer!) },
                    rec: {
                        input: new Uint8Array(recBuffer!),
                        decodeDic: dictText!,
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
