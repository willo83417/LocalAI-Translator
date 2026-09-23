import { pipeline, env, TextStreamer, StoppingCriteria, StoppingCriteriaList } from '@huggingface/transformers';
import { toTraditionalChinese, isTraditionalChinese } from '../utils/chineseConverter';

// --- Environment Configuration ---
env.backends.onnx.executionProviders = ['webgpu', 'wasm'];
env.useBrowserCache = true;
env.useWasmCache = true;    //new wasm cache setting
env.cacheDir = 'transformers-cache';

// --- Worker State & Communication ---
interface WorkerMessage {
    type: 'load' | 'transcribe' | 'unload' | 'cancel';
    payload?: any;
}

interface AppMessage {
    type: 'log' | 'transcription' | 'transcription-partial' | 'loaded' | 'error' | 'progress' | 'unloaded';
    payload: any;
}

const post = (message: AppMessage) => self.postMessage(message);

// Maps specific language codes to prompts that guide the Whisper model's output format.
const PROMPT_MAP: Record<string, string> = {
    'zh-TW': '請使用(zh-Hant)繁體中文輸出。',
    'zh-HK': '請使用(zh-Hant-HK)香港繁體中文輸出。',
    'zh-Hans': '请使用(zh-Hans)简体中文输出。'
};

class InterruptableStoppingCriteria extends StoppingCriteria {
    interrupted: boolean = false;
    
    interrupt() {
        this.interrupted = true;
    }
    
    reset() {
        this.interrupted = false;
    }
    
    _call(input_ids: any, scores: any) {
        const batchSize = input_ids.dims ? input_ids.dims[0] : (input_ids.length || 1);
        return new Array(batchSize).fill(this.interrupted);
    }
}

class Transcriber {
    private transcriber: any = null;
    private loading: boolean = false;
    private currentModelId: string | null = null;
    private processingQueue: Array<{ audioData: Float32Array, asrLanguage: string, promptLanguage: string, isFinal: boolean }> = [];
    private isProcessing: boolean = false;
    private abortCurrent: boolean = false;
    private currentProcessingIsFinal: boolean = false;
    private stoppingCriteria: InterruptableStoppingCriteria;

    constructor() {
        this.stoppingCriteria = new InterruptableStoppingCriteria();
    }

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
        post({ type: 'log', payload: `Initializing pipeline for model: ${modelId} with quantization: ${JSON.stringify(quantization)}` });

        try {
            // Dispose previous transcriber if it exists to free memory
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

        if (isFinal) {
            // If we are currently processing a non-final request, abort it so we can process the final request immediately
            if (this.isProcessing && !this.currentProcessingIsFinal) {
                this.abortCurrent = true;
                this.stoppingCriteria.interrupt();
            }
            // Clear any pending non-final requests from the queue to process final audio right away
            this.processingQueue = this.processingQueue.filter(item => item.isFinal);
        } else {
            // For realtime preview, discard older pending interim requests to prevent queue lag
            this.processingQueue = this.processingQueue.filter(item => item.isFinal);
        }

        this.processingQueue.push({ audioData, asrLanguage, promptLanguage, isFinal });
        this.processQueue();
    }

    private async processQueue() {
        if (this.isProcessing || this.processingQueue.length === 0) {
            return;
        }

        this.isProcessing = true;
        this.abortCurrent = false;
        this.stoppingCriteria.reset();
        const { audioData, asrLanguage, promptLanguage, isFinal } = this.processingQueue.shift()!;
        this.currentProcessingIsFinal = isFinal;

        // Skip realtime processing if audio is too short (< 0.6s at 16kHz = 9600 samples)
        // Whisper with less than 0.6s audio cannot reliably infer tokens and produces hallucinations.
        if (!isFinal && audioData.length < 9600) {
            this.isProcessing = false;
            this.processQueue();
            return;
        }

        post({ type: 'log', payload: `Starting transcription (ASR Lang: ${asrLanguage}, Prompt Lang: ${promptLanguage}, isFinal: ${isFinal})...` });

        try {
            const tokenizer = this.transcriber.tokenizer;
            const shouldConvertToTraditional = isTraditionalChinese(promptLanguage) || isTraditionalChinese(asrLanguage);
            let fullTranscription = "";

            const generationOptions: any = {
                language: asrLanguage?.startsWith('zh') ? 'chinese' : (asrLanguage === 'auto' ? undefined : asrLanguage),
                task: 'transcribe',
                temperature: 0.0, // Greedy decoding produces deterministic results and drastically suppresses hallucinations
                repetition_penalty: 1.2, // Penalize repeating loops on incomplete audio
                condition_on_previous_text: false, // Prevents partial hallucinations from conditioning subsequent output
                stopping_criteria: new StoppingCriteriaList([this.stoppingCriteria]),
                chunk_length_s: 30, // Limits memory scaling and improves continuous long-form decoding
                stride_length_s: 5, // Adds overlap between chunks to prevent word-cutting boundaries
            };

            if (isFinal) {
                // Use TextStreamer for final audio processing
                const streamer = new TextStreamer(tokenizer, {
                    skip_prompt: true,
                    skip_special_tokens: true,
                    callback_function: (text: string) => {
                        if (this.abortCurrent || this.stoppingCriteria.interrupted) {
                            throw new Error('ABORTED');
                        }
                        fullTranscription += text;
                        const outputPartial = shouldConvertToTraditional ? toTraditionalChinese(fullTranscription) : fullTranscription;
                        post({ type: 'transcription-partial', payload: outputPartial });
                    }
                });
                generationOptions.streamer = streamer;

                // Use the specific promptLanguage code to look up the correct prompt for final transcription
                const promptText = PROMPT_MAP[promptLanguage];
                if (promptText) {
                    post({ type: 'log', payload: `Applying prompt for ${promptLanguage}: "${promptText}"` });
                    const { input_ids } = await this.transcriber.tokenizer(promptText);
                    generationOptions.prompt_ids = input_ids.data.slice(0, -1);
                }
            }

            const output = await this.transcriber(audioData, generationOptions);
            
            if (this.abortCurrent || this.stoppingCriteria.interrupted) {
                 throw new Error('ABORTED');
            }

            // Ensure we send the final authoritative result from the pipeline output
            const rawFinalText = (Array.isArray(output) ? output[0].text : output.text) || '';
            const finalText = shouldConvertToTraditional ? toTraditionalChinese(rawFinalText) : rawFinalText;
            post({ type: 'transcription', payload: { text: finalText.trim(), isFinal } });
            post({ type: 'log', payload: 'Transcription completed successfully.' });
        } catch (error) {
            const errorMessage = error instanceof Error ? error.message : String(error);
            if (errorMessage === 'ABORTED') {
                post({ type: 'log', payload: 'Transcription aborted for newer request.' });
                // We don't send an error message to the UI for aborted requests
            } else {
                console.error('ASR Transcription Error:', error);
                post({ type: 'error', payload: `Transcription error: ${errorMessage}` });
            }
        } finally {
            this.isProcessing = false;
            this.abortCurrent = false;
            this.processQueue();
        }
    }

    async unload() {
        if (!this.transcriber) {
            post({ type: 'log', payload: 'No ASR model is currently loaded. Nothing to unload.' });
            return;
        }

        post({ type: 'log', payload: `Unloading model: ${this.currentModelId}` });
        try {
            // Explicitly call dispose() to free up memory (including WebGPU/WebGL resources).
            await this.transcriber.dispose();
            this.transcriber = null;
            this.currentModelId = null;
            post({ type: 'unloaded', payload: true });
            post({ type: 'log', payload: 'ASR pipeline instance disposed and released.' });

        } catch (error) {
            const errorMessage = error instanceof Error ? error.message : String(error);
            console.error('ASR Unload Error:', error);
            post({ type: 'error', payload: `Error unloading ASR model: ${errorMessage}` });
        }
    }

    cancel() {
        this.processingQueue = [];
        if (this.isProcessing) {
            this.abortCurrent = true;
            this.stoppingCriteria.interrupt();
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
