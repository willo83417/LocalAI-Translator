import { pipeline, env } from '@huggingface/transformers';
import { toTraditionalChinese, isTraditionalChinese } from '../utils/chineseConverter';

// --- Environment Configuration ---
env.backends.onnx.executionProviders = ['webgpu', 'wasm'];
env.useBrowserCache = true;
env.useWasmCache = true;
env.cacheDir = 'transformers-cache';

interface WorkerMessage {
    type: 'load' | 'transcribe' | 'unload' | 'cancel';
    payload?: any;
}

interface AppMessage {
    type: 'log' | 'transcription' | 'transcription-partial' | 'loaded' | 'error' | 'progress' | 'unloaded';
    payload: any;
}

const post = (message: AppMessage) => self.postMessage(message);

class Transcriber {
    private transcriber: any = null;
    private loading: boolean = false;
    private currentModelId: string | null = null;
    private currentQuantization: any = null;
    private isProcessing: boolean = false;
    private nextRequest: { audioData: Float32Array, asrLanguage: string, promptLanguage: string, isFinal: boolean } | null = null;
    private ignoreCurrentResult: boolean = false;

    async load(payload: { modelId: string, quantization: any }) {
        const { modelId, quantization } = payload;
        
        if (this.loading) {
            post({ type: 'log', payload: `Already loading model: ${this.currentModelId}` });
            return;
        }

        if (this.transcriber && this.currentModelId === modelId) {
            post({ type: 'log', payload: `Model ${modelId} is already loaded.` });
            post({ type: 'loaded', payload: true });
            return;
        }

        this.loading = true;
        this.currentModelId = modelId;
        this.currentQuantization = quantization;
        post({ type: 'log', payload: `Initializing pipeline for model: ${modelId} with quantization: ${JSON.stringify(quantization)}` });

        try {
            if (this.transcriber) {
                post({ type: 'log', payload: `Disposing previous model: ${this.currentModelId}`});
                await this.transcriber.dispose();
                this.transcriber = null;
            }

            let lastProgressTime = 0;
            this.transcriber = await pipeline('automatic-speech-recognition', modelId, {
                progress_callback: (progress: any) => {
                     const now = Date.now();
                     if (now - lastProgressTime > 100 || progress.status === 'done' || progress.status === 'ready' || progress.status === 'progress_total') {
                         post({ type: 'progress', payload: progress });
                         lastProgressTime = now;
                     }
                },
                dtype: quantization,
                device: "webgpu",
            });
            
            post({ type: 'loaded', payload: true });
            post({ type: 'log', payload: `Model ${modelId} loaded successfully on WebGPU.` });
        } catch (error) {
            const errorMessage = error instanceof Error ? error.message : String(error);
            console.error('ASR Load Error:', error);
            post({ type: 'error', payload: `Error loading ASR model: ${errorMessage}` });
            this.currentModelId = null;
            this.transcriber = null;
        } finally {
            this.loading = false;
        }
    }

    async transcribe(audioData: Float32Array, asrLanguage: string, promptLanguage: string, isFinal: boolean = true) {
        if (!this.transcriber) {
            const errorMsg = `Transcriber not ready. The pipeline was not initialized correctly. Current model ID: ${this.currentModelId}`;
            console.error(errorMsg);
            post({ type: 'error', payload: errorMsg });
            return;
        }

        // Whisper is an utterance-level model; ignore empty or near-silent tiny buffers (< 0.25s)
        if (!audioData || audioData.length < 4000) {
            if (isFinal) {
                post({ type: 'transcription', payload: { text: '', isFinal: true } });
            }
            return;
        }

        // If another transcription is currently running on WebGPU, store as next pending request
        if (this.isProcessing) {
            if (isFinal) {
                this.nextRequest = { audioData, asrLanguage, promptLanguage, isFinal };
            }
            return;
        }

        this.isProcessing = true;
        this.ignoreCurrentResult = false;

        try {
            post({ type: 'log', payload: `Starting Whisper transcription (ASR Lang: ${asrLanguage}, Prompt Lang: ${promptLanguage})...` });

            const shouldConvertToTraditional = isTraditionalChinese(promptLanguage) || isTraditionalChinese(asrLanguage);

            const generationOptions: any = {
                language: asrLanguage?.startsWith('zh') ? 'chinese' : (asrLanguage === 'auto' ? undefined : asrLanguage),
                task: 'transcribe',
            };

            // Enable chunking for long audio (> 28 seconds) to prevent Whisper's 30s limit
            if (audioData.length > 16000 * 28) {
                generationOptions.chunk_length_s = 30;
                generationOptions.stride_length_s = 5;
            }

            const output = await this.transcriber(audioData, generationOptions);

            if (this.ignoreCurrentResult) {
                post({ type: 'log', payload: 'Whisper result ignored due to cancellation.' });
                return;
            }

            const rawFinalText = (Array.isArray(output) ? output[0].text : output.text) || '';
            const finalText = (shouldConvertToTraditional ? toTraditionalChinese(rawFinalText) : rawFinalText).trim();

            post({ type: 'transcription', payload: { text: finalText, isFinal: true } });
            post({ type: 'log', payload: 'Whisper transcription completed successfully.' });
        } catch (error) {
            const errorMessage = error instanceof Error ? error.message : String(error);
            console.error('Whisper Transcription Error:', error);
            post({ type: 'error', payload: `Transcription error: ${errorMessage}` });
        } finally {
            this.isProcessing = false;
            if (this.nextRequest) {
                const req = this.nextRequest;
                this.nextRequest = null;
                this.transcribe(req.audioData, req.asrLanguage, req.promptLanguage, req.isFinal);
            }
        }
    }

    async unload() {
        if (!this.transcriber) {
            post({ type: 'log', payload: 'No ASR model is currently loaded. Nothing to unload.' });
            return;
        }

        post({ type: 'log', payload: `Unloading model: ${this.currentModelId}` });
        try {
            await this.transcriber.dispose();
            this.transcriber = null;
            this.currentModelId = null;
            this.nextRequest = null;
            this.isProcessing = false;
            post({ type: 'unloaded', payload: true });
            post({ type: 'log', payload: 'ASR pipeline instance disposed and released.' });
        } catch (error) {
            const errorMessage = error instanceof Error ? error.message : String(error);
            console.error('ASR Unload Error:', error);
            post({ type: 'error', payload: `Error unloading ASR model: ${errorMessage}` });
        }
    }

    cancel() {
        this.nextRequest = null;
        if (this.isProcessing) {
            this.ignoreCurrentResult = true;
        }
        post({ type: 'log', payload: 'ASR transcription cancelled.' });
    }
}

const transcriber = new Transcriber();

self.onmessage = (event: MessageEvent<WorkerMessage>) => {
    const { type, payload } = event.data;
    switch (type) {
        case 'load':
            if (payload && payload.modelId && payload.quantization) {
                transcriber.load(payload);
            } else {
                post({ type: 'error', payload: 'Invalid load payload: modelId and quantization are required.' });
            }
            break;
        case 'transcribe':
            const { audio, asrLanguage, promptLanguage, isFinal } = payload;
            if (audio && asrLanguage !== undefined && promptLanguage !== undefined) {
                transcriber.transcribe(audio, asrLanguage, promptLanguage, isFinal ?? true);
            } else {
                post({ type: 'error', payload: 'Invalid transcribe payload: audio, asrLanguage, and promptLanguage are required.' });
            }
            break;
        case 'unload':
            transcriber.unload();
            break;
        case 'cancel':
            transcriber.cancel();
            break;
        default:
            console.warn(`ASR Worker received unknown message type: ${type}`);
            break;
    }
};
