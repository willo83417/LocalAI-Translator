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

const isMobileDevice = /Android|iPhone|iPad|iPod/i.test(self.navigator?.userAgent || '');

/**
 * Checks whether the newly added audio slice [startIdx .. audio.length) contains
 * actual speech energy (RMS > 0.008 in any 50ms window).
 */
function hasSpeechEnergy(audio: Float32Array, startIdx: number): boolean {
    const start = Math.max(0, startIdx);
    if (audio.length - start < 1600) return false;
    const win = 800; // 50ms at 16kHz
    const thresholdSq = 0.008 * 0.008 * win;
    for (let pos = start; pos + win <= audio.length; pos += win) {
        let sumSq = 0;
        for (let k = pos; k < pos + win; k++) {
            const s = audio[k];
            sumSq += s * s;
        }
        if (sumSq > thresholdSq) return true;
    }
    return false;
}

class Transcriber {
    private transcriber: any = null;
    private loading: boolean = false;
    private currentModelId: string | null = null;
    private currentQuantization: any = null;
    private processingQueue: Array<{ audioData: Float32Array, asrLanguage: string, promptLanguage: string, isFinal: boolean }> = [];
    private isProcessing: boolean = false;
    private abortCurrent: boolean = false;
    private wasJustAborted: boolean = false;
    private currentProcessingIsFinal: boolean = false;
    private currentAudioLength: number = 0;
    private upgradeCurrentToFinal: boolean = false;
    private lastTranscribedSampleCount: number = 0;
    private lastInterimText: string = '';
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
        this.currentQuantization = quantization;
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

        // Detect start of a new utterance (buffer length reset)
        if (audioData.length < this.lastTranscribedSampleCount) {
            this.lastTranscribedSampleCount = 0;
            this.lastInterimText = '';
            this.upgradeCurrentToFinal = false;
        }

        if (isFinal) {
            this.processingQueue = [];
            if (this.isProcessing && !this.currentProcessingIsFinal) {
                // Check if the user actually spoke any new words after the currently running interim pass started
                const hasNewSpeech = hasSpeechEnergy(audioData, this.currentAudioLength);
                if (!hasNewSpeech && this.currentAudioLength >= 8000) {
                    // The currently running interim pass already covers 100% of the spoken audio (rest is VAD hangover silence).
                    // Do NOT abort and re-run a 30s Whisper Encoder+Decoder pass! Upgrade the running pass to final.
                    this.upgradeCurrentToFinal = true;
                    return;
                }
                this.abortCurrent = true;
                this.stoppingCriteria.interrupt();
            } else if (!this.isProcessing) {
                // If an interim pass already finished and the remaining audio since lastTranscribedSampleCount is pure silence:
                const hasNewSpeech = this.lastTranscribedSampleCount === 0 || hasSpeechEnergy(audioData, this.lastTranscribedSampleCount);
                if (!hasNewSpeech && this.lastInterimText) {
                    const finalOut = this.lastInterimText;
                    this.lastTranscribedSampleCount = 0;
                    this.lastInterimText = '';
                    post({ type: 'transcription', payload: { text: finalOut, isFinal: true } });
                    return;
                }
            }
        } else {
            // For realtime preview:
            // 1. If we are already processing on GPU, drop incoming interim chunk (keep GPU 100% focused on current pass)
            if (this.isProcessing) {
                return;
            }
            // 2. Skip if no new speech energy since last transcribed sample count
            const minNewSamples = isMobileDevice ? 12800 : 6400; // ~0.8s on mobile, ~0.4s on desktop
            if (
                audioData.length - this.lastTranscribedSampleCount < minNewSamples ||
                !hasSpeechEnergy(audioData, this.lastTranscribedSampleCount)
            ) {
                post({ type: 'transcription-partial', payload: { text: '', stillBusy: false } });
                return;
            }
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
        this.upgradeCurrentToFinal = false;
        this.stoppingCriteria.reset();
        const { audioData, asrLanguage, promptLanguage, isFinal } = this.processingQueue.shift()!;
        this.currentProcessingIsFinal = isFinal;
        this.currentAudioLength = audioData.length;

        // Skip realtime processing if audio is too short (< 0.5s at 16kHz = 8000 samples)
        // Whisper with less than 0.5s audio cannot reliably infer tokens and produces hallucinations.
        if (!isFinal && audioData.length < 8000) {
            this.isProcessing = false;
            post({ type: 'transcription-partial', payload: { text: '', stillBusy: false } });
            this.processQueue();
            return;
        }

        // If the previous pass was interrupted on mobile WebGPU, yield a short macrotask
        // so Dawn/Vulkan finishes reclaiming freed GPUBuffers before allocating a new 30s Whisper pass.
        if (this.wasJustAborted) {
            this.wasJustAborted = false;
            await new Promise(resolve => setTimeout(resolve, 25));
        }

        post({ type: 'log', payload: `Starting transcription (ASR Lang: ${asrLanguage}, Prompt Lang: ${promptLanguage}, isFinal: ${isFinal})...` });

        try {
            const tokenizer = this.transcriber.tokenizer;
            const shouldConvertToTraditional = isTraditionalChinese(promptLanguage) || isTraditionalChinese(asrLanguage);
            let fullTranscription = "";

            // For non-final realtime preview, bound audio to latest 8s (128,000 samples)
            const effectiveAudio = (!isFinal && audioData.length > 128000)
                ? audioData.subarray(audioData.length - 128000)
                : audioData;

            const streamer = new TextStreamer(tokenizer, {
                skip_prompt: true,
                skip_special_tokens: true,
                callback_function: (text: string) => {
                    if (this.abortCurrent || this.stoppingCriteria.interrupted) {
                        // CRITICAL: Do NOT throw inside TextStreamer.callback_function!
                        // Throwing inside streamer.put() jumps out of @huggingface/transformers generate()
                        // BEFORE it disposes encoder_outputs and all 48 past_key_values GPUBuffers,
                        // causing WebGPU Device Lost (mapAsync status 4) on mobile!
                        // Instead, return and let StoppingCriteria exit generate() cleanly on the very same step.
                        return;
                    }
                    fullTranscription += text;
                    const trimmed = fullTranscription.trim();
                    if (trimmed) {
                        const outputPartial = shouldConvertToTraditional ? toTraditionalChinese(trimmed) : trimmed;
                        this.lastInterimText = outputPartial;
                        post({ type: 'transcription-partial', payload: { text: outputPartial, stillBusy: true } });
                    }
                }
            });

            const generationOptions: any = {
                language: asrLanguage?.startsWith('zh') ? 'chinese' : (asrLanguage === 'auto' ? undefined : asrLanguage),
                task: 'transcribe',
                temperature: 0.0, // Greedy decoding produces deterministic results and drastically suppresses hallucinations
                repetition_penalty: 1.2, // Penalize repeating loops on incomplete audio
                condition_on_previous_text: false, // Prevents partial hallucinations from conditioning subsequent output
                stopping_criteria: new StoppingCriteriaList([this.stoppingCriteria]),
                streamer,
                max_new_tokens: (isFinal || this.upgradeCurrentToFinal) ? 128 : 48,
            };

            if (effectiveAudio.length > 16000 * 28) {
                generationOptions.chunk_length_s = 30;
                generationOptions.stride_length_s = 5;
            }

            // Note: On mobile WebGPU, injecting prompt_ids changes decoder_model_merged prefill seq_len from 4 to ~19,
            // forcing dynamic WebGPU shader recompilation and a 5x larger logits GPUBuffer download on isFinal.
            // Since toTraditionalChinese() already converts zh-TW / zh-HK deterministically on CPU, only apply prompt_ids on desktop.
            if (isFinal && !isMobileDevice) {
                const promptText = PROMPT_MAP[promptLanguage];
                if (promptText) {
                    post({ type: 'log', payload: `Applying prompt for ${promptLanguage}: "${promptText}"` });
                    const { input_ids } = await this.transcriber.tokenizer(promptText);
                    generationOptions.prompt_ids = input_ids.data.slice(0, -1);
                }
            }

            const output = await this.transcriber(effectiveAudio, generationOptions);
            
            if (this.abortCurrent || this.stoppingCriteria.interrupted) {
                this.wasJustAborted = true;
                post({ type: 'log', payload: 'Transcription aborted cleanly for newer final request.' });
                return;
            }

            const effectiveIsFinal = isFinal || this.upgradeCurrentToFinal;
            const rawFinalText = (Array.isArray(output) ? output[0].text : output.text) || '';
            const finalText = (shouldConvertToTraditional ? toTraditionalChinese(rawFinalText) : rawFinalText).trim();

            if (effectiveIsFinal) {
                this.lastTranscribedSampleCount = 0;
                this.lastInterimText = '';
                this.upgradeCurrentToFinal = false;
                post({ type: 'transcription', payload: { text: finalText, isFinal: true } });
            } else {
                this.lastTranscribedSampleCount = audioData.length;
                if (finalText) {
                    this.lastInterimText = finalText;
                }
                post({ type: 'transcription', payload: { text: finalText, isFinal: false } });
            }
            post({ type: 'log', payload: 'Transcription completed successfully.' });
        } catch (error) {
            const errorMessage = error instanceof Error ? error.message : String(error);
            if (errorMessage === 'ABORTED') {
                this.wasJustAborted = true;
                post({ type: 'log', payload: 'Transcription aborted for newer request.' });
            } else {
                console.error('ASR Transcription Error:', error);
                post({ type: 'error', payload: `Transcription error: ${errorMessage}` });
            }
        } finally {
            this.isProcessing = false;
            this.abortCurrent = false;
            if (this.processingQueue.length > 0) {
                this.processQueue();
            } else {
                post({ type: 'transcription-partial', payload: { text: '', stillBusy: false } });
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
            // Explicitly call dispose() to free up memory (including WebGPU/WebGL resources).
            await this.transcriber.dispose();
            this.transcriber = null;
            this.currentModelId = null;
            this.lastTranscribedSampleCount = 0;
            this.lastInterimText = '';
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
        this.lastTranscribedSampleCount = 0;
        this.lastInterimText = '';
        this.upgradeCurrentToFinal = false;
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
