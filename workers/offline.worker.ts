
// workers/offline.worker.ts
// This worker acts as a controller, managing an inference worker
// and handling complex logic like two-step translations.

let inferenceWorker: Worker | null = null;
let currentEngine: 'mediapipe' | 'transformers' | 'litert-lm' = 'mediapipe';
let twoStepState: {
    resolve: (value: unknown) => void;
    reject: (reason?: any) => void;
} | null = null;

// This function attaches the necessary event listeners to the inference worker instance.
const setupInferenceWorkerListeners = () => {
    if (!inferenceWorker) return;

    inferenceWorker.onmessage = (e: MessageEvent) => {
        const { type: msgType, payload: msgPayload } = e.data;
        if (twoStepState && msgType === 'translation_full_done') {
            twoStepState.resolve(msgPayload.result);
            twoStepState = null;
        } else if (twoStepState && (msgType === 'translation_error' || msgType === 'error')) {
            twoStepState.reject(new Error(msgPayload.error));
            twoStepState = null;
        } else {
            // Forward all other messages to the main thread
            self.postMessage(e.data);
        }
    };
    
    inferenceWorker.onerror = (error) => {
        const errorMessage = `Inference worker error: ${error.message}`;
        console.error("Controller received an error from inference worker:", error);
        if (twoStepState) {
            twoStepState.reject(new Error(errorMessage));
            twoStepState = null;
        }
        self.postMessage({ type: 'init_error', payload: { error: errorMessage } });
    };
};

// This function now handles both creation and message forwarding logic
const forwardToInferenceWorker = (event: MessageEvent) => {
    if (!inferenceWorker) {
        console.error("Inference worker does not exist. Cannot process message:", event.data.type);
        self.postMessage({ type: 'error', payload: { error: 'Inference worker is not initialized.' } });
        return;
    }
    
    if (event.data.type === 'extractText' && event.data.payload.imageBitmap) {
        inferenceWorker.postMessage(event.data, [event.data.payload.imageBitmap]);
    } else {
        inferenceWorker.postMessage(event.data);
    }
};

const handleTranslate = async (payload: any) => {
    const { text, sourceLang, targetLang, sourceLangCode, targetLangCode, isTwoStepEnabled } = payload;
    const isJpToCn = sourceLangCode === 'ja' && targetLangCode.startsWith('zh');

    if (isTwoStepEnabled && isJpToCn) {
        try {
            const intermediateEnglishPromise = new Promise((resolve, reject) => {
                twoStepState = { resolve, reject };
                if (!inferenceWorker) throw new Error("Inference worker not available for two-step translation.");
                inferenceWorker.postMessage({
                    type: 'translate_full',
                    payload: { text, sourceLang: 'Japanese', targetLang: 'English' }
                });
            });
            const intermediateEnglish = await intermediateEnglishPromise as string;
            if (!intermediateEnglish?.trim()) throw new Error('Intermediate English translation failed.');
            if (!inferenceWorker) throw new Error("Inference worker not available for second step of translation.");
            inferenceWorker.postMessage({
                type: 'translate_stream',
                payload: { text: intermediateEnglish, sourceLang: 'English', targetLang }
            });
        } catch (error) {
            const isAbort = error instanceof DOMException && error.name === 'AbortError';
            if (isAbort) {
                self.postMessage({ type: 'translation_cancelled' });
            } else {
                const errorMessage = error instanceof Error ? error.message : 'Two-step translation failed.';
                self.postMessage({ type: 'translation_error', payload: { error: errorMessage } });
            }
            twoStepState = null;
        }
    } else {
        if (!inferenceWorker) {
             self.postMessage({ type: 'translation_error', payload: { error: "Inference worker not ready." } });
             return;
        }
        inferenceWorker.postMessage({ type: 'translate_stream', payload });
    }
};

const handleCancel = () => {
    if (twoStepState) {
        twoStepState.reject(new DOMException('Translation cancelled.', 'AbortError'));
        twoStepState = null;
    }
    if (inferenceWorker) {
        inferenceWorker.postMessage({ type: 'cancel_task' });
    }
};

self.onmessage = async (event: MessageEvent) => {
    const { type, payload } = event.data;

    switch (type) {
        case 'init':
            const engine = payload.engine || 'mediapipe';
            if (inferenceWorker && currentEngine !== engine) {
                 inferenceWorker.terminate();
                 inferenceWorker = null;
            }
            if (!inferenceWorker) {
                console.log(`Controller: Creating inference worker for ${engine}...`);
                if (engine === 'transformers') {
                    inferenceWorker = new Worker(new URL('./transformers.worker.ts', import.meta.url), { type: 'module' });
                } else if (engine === 'litert-lm') {
                    inferenceWorker = new Worker(new URL('./litert-lm.worker.ts', import.meta.url), { type: 'module' });
                } else {
                    inferenceWorker = new Worker(new URL('./mediapipe.worker.ts?url', import.meta.url), { type: 'classic' });
                }
                currentEngine = engine;
                setupInferenceWorkerListeners();
            }
            forwardToInferenceWorker(event);
            break;

        case 'unload':
            if (inferenceWorker) {
                console.log('Controller: Forwarding unload command to inference worker for graceful shutdown.');
                // Forward the unload command to the inference worker to allow it to clean up GPU resources.
                inferenceWorker.postMessage({ type: 'unload' });
            } else {
                // If there's no worker, we can consider it already unloaded.
                console.log('Controller: Unload called but no inference worker exists.');
                self.postMessage({ type: 'unload_done' });
            }
            break;

        case 'transcribe':
        case 'extractText':
        case 'translate_full':
            forwardToInferenceWorker(event);
            break;
            
        case 'translate':
            await handleTranslate(payload);
            break;
            
        case 'cancel_translation':
            handleCancel();
            break;
            
        default:
            console.warn(`Unknown controller worker message type: ${type}`);
    }
};
