// Polyfill SharedArrayBuffer if not defined in worker scope
if (typeof (globalThis as any).SharedArrayBuffer === 'undefined') {
    (globalThis as any).SharedArrayBuffer = ArrayBuffer;
}

import { AsrEngine } from "@jeffpeng3/nemotron-asr-core";
import { toTraditionalChinese, isTraditionalChinese } from '../utils/chineseConverter';

interface WorkerMessage {
    type: 'load' | 'transcribe' | 'unload' | 'cancel';
    payload?: any;
}

interface AppMessage {
    type: 'log' | 'transcription' | 'transcription-partial' | 'loaded' | 'error' | 'progress' | 'unloaded';
    payload: any;
}

const post = (message: AppMessage) => self.postMessage(message);

class NemotronTranscriber {
    private engine: AsrEngine | null = null;
    private session: any = null;
    private loading: boolean = false;
    private isProcessing: boolean = false;
    private abortCurrent: boolean = false;
    private processingQueue: Array<{ audioData: Float32Array, asrLanguage: string, promptLanguage: string, isFinal: boolean }> = [];
    private currentProfile: string = '';
    private currentBeamWidth: number = 1;
    private lastProcessedIndex: number = 0;
    private accumulatedText: string = '';

    async load(payload: { asrProfile?: string, asrBeamWidth?: number }) {
        if (this.loading) return;

        const profile = payload.asrProfile || 'NORMAL';
        const beamWidth = payload.asrBeamWidth || 1;

        if (this.engine && this.currentProfile === profile && this.currentBeamWidth === beamWidth) {
            post({ type: 'loaded', payload: true });
            return;
        }

        this.loading = true;
        try {
            if (this.engine) {
                if (this.currentBeamWidth === beamWidth && this.currentProfile !== profile) {
                    post({ type: 'log', payload: `Switching Nemotron ASR profile to ${profile}...` });
                    await (this.engine as any).switchProfile(profile);
                    this.currentProfile = profile;
                    this.session = null;
                    this.lastProcessedIndex = 0;
                    this.accumulatedText = '';
                    post({ type: 'loaded', payload: true });
                    post({ type: 'log', payload: `Nemotron ASR profile switched successfully to ${profile}.` });
                    this.loading = false;
                    return;
                }
                this.session = null;
                this.engine = null;
            }

            post({ type: 'log', payload: `Initializing Nemotron ASR (Profile: ${profile}, Beam: ${beamWidth})...` });
            this.engine = new AsrEngine({
                progress(label, loaded, total, cached) {
                    const percent = total > 0 ? (loaded / total) * 100 : 0;
                    post({ type: 'progress', payload: { status: 'progress', file: label, progress: percent } });
                },
                status(detail) {
                    post({ type: 'log', payload: `Nemotron status: ${detail}` });
                },
                ep(isEncoder, provider, note) {
                    post({ type: 'log', payload: `Nemotron EP [${isEncoder ? 'encoder' : 'decoder'}]: ${provider} ${note || ''}` });
                }
            }, {
                wasmPaths:'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/',
				profile: profile,
                beamWidth: beamWidth,
                numThreads: 0
            });

            await this.engine.init();
            
            this.currentProfile = profile;
            this.currentBeamWidth = beamWidth;
            post({ type: 'loaded', payload: true });
            post({ type: 'log', payload: `Nemotron ASR loaded successfully.` });
        } catch (error) {
            console.error('Nemotron Load Error:', error);
            post({ type: 'error', payload: `Error loading Nemotron model: ${error}` });
            this.engine = null;
        } finally {
            this.loading = false;
        }
    }

    async transcribe(audioData: Float32Array, asrLanguage: string, promptLanguage: string, isFinal: boolean = true) {
        if (!this.engine) {
            post({ type: 'error', payload: 'Nemotron engine not initialized.' });
            return;
        }
        
        this.processingQueue = this.processingQueue.filter(req => req.isFinal);

        if (this.isProcessing && !isFinal) {
             // In streaming mode, we don't abort, we just queue it.
             // Actually, we process everything sequentially.
             // We can let the queue build up if processing is slow, 
             // but we shouldn't drop intermediate chunks otherwise we lose audio.
             // Wait, the app sends the FULL accumulated audioData every time.
             // So if we drop intermediate requests, the next request will have the accumulated data anyway!
             // That's perfect.
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
        const { audioData, asrLanguage, promptLanguage, isFinal } = this.processingQueue.shift()!;
        const shouldConvertToTraditional = isTraditionalChinese(promptLanguage) || isTraditionalChinese(asrLanguage);

        try {
            let langId = 101;
            if (asrLanguage?.startsWith('zh')) langId = 4;
            if (asrLanguage === 'en' || asrLanguage === 'en-US') langId = 0;
            if (asrLanguage === 'ja') langId = 10;
            if (asrLanguage === 'es') langId = 3;
            if (asrLanguage === 'fr') langId = 8;
            if (asrLanguage === 'de') langId = 9;
            if (asrLanguage === 'ru') langId = 11;
            if (asrLanguage === 'ko') langId = 14;
            if (asrLanguage === 'it') langId = 15;

            // Handle new session if audioData is shorter than lastProcessedIndex (new recording)
            if (audioData.length < this.lastProcessedIndex) {
                this.session = null;
                this.lastProcessedIndex = 0;
                this.accumulatedText = '';
            }

            if (!this.session) {
                this.session = this.engine!.session(langId);
                this.lastProcessedIndex = 0;
                this.accumulatedText = '';
            }

            const newChunk = audioData.subarray(this.lastProcessedIndex);
            
            if (newChunk.length > 0) {
                const partial = await this.session.feed(newChunk);
                this.lastProcessedIndex = audioData.length;
                
                if (partial && !isFinal && !this.abortCurrent) {
                    const partialText = partial.map((p: any) => p.text).filter(Boolean).join(' ');
                    if (partialText) {
                        this.accumulatedText = (this.accumulatedText ? this.accumulatedText + ' ' : '') + partialText;
                    }
                }
            }

            if (isFinal) {
                const finalResult = await this.session.end();
                this.session = null;
                this.lastProcessedIndex = 0;
                this.accumulatedText = '';
                
                if (!this.abortCurrent) {
                    const rawFinal = finalResult?.text || '';
                    const finalText = shouldConvertToTraditional ? toTraditionalChinese(rawFinal) : rawFinal;
                    post({ type: 'transcription', payload: { text: finalText, isFinal: true } });
                }
            } else {
                if (!this.abortCurrent) {
                    const outputPartial = shouldConvertToTraditional ? toTraditionalChinese(this.accumulatedText) : this.accumulatedText;
                    post({ type: 'transcription-partial', payload: outputPartial });
                }
            }
        } catch (error) {
            console.error('Nemotron Transcription Error:', error);
            post({ type: 'error', payload: `Nemotron transcription error: ${error}` });
            this.session = null;
            this.lastProcessedIndex = 0;
            this.accumulatedText = '';
        } finally {
            this.isProcessing = false;
            this.abortCurrent = false;
            this.processQueue();
        }
    }

    async unload() {
        this.engine = null;
        this.session = null;
        post({ type: 'unloaded', payload: true });
    }

    cancel() {
        this.processingQueue = [];
        if (this.isProcessing) {
            this.abortCurrent = true;
        } else {
            this.session = null;
            this.lastProcessedIndex = 0;
            this.accumulatedText = '';
        }
    }
}

const transcriber = new NemotronTranscriber();

self.onmessage = (event: MessageEvent<WorkerMessage>) => {
    const { type, payload } = event.data;
    switch (type) {
        case 'load':
            transcriber.load(payload || {});
            break;
        case 'transcribe':
            const { audio, asrLanguage, promptLanguage, isFinal } = payload;
            if (audio) {
                transcriber.transcribe(audio, asrLanguage, promptLanguage, isFinal ?? true);
            }
            break;
        case 'unload':
            transcriber.unload();
            break;
        case 'cancel':
            transcriber.cancel();
            break;
    }
};
