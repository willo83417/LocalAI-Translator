// Polyfill SharedArrayBuffer if not defined in worker scope
if (typeof (globalThis as any).SharedArrayBuffer === 'undefined') {
    (globalThis as any).SharedArrayBuffer = ArrayBuffer;
}

import { AsrEngine, detok } from "@jeffpeng3/nemotron-asr-core";
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
    private lastEmittedPartial: string = '';

    async load(payload: { asrProfile?: string, asrBeamWidth?: number }) {
        if (this.loading) return;

        const validProfiles = ['TURBO', 'FAST', 'BALANCED', 'NORMAL', 'HIGH'] as const;
        const rawProfile = payload.asrProfile === 'BEST' ? 'HIGH' : (payload.asrProfile || 'NORMAL');
        const profile = (validProfiles.includes(rawProfile as any) ? rawProfile : 'NORMAL') as any;
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
                    this.resetSessionState();
                    post({ type: 'loaded', payload: true });
                    post({ type: 'log', payload: `Nemotron ASR profile switched successfully to ${profile}.` });
                    this.loading = false;
                    return;
                }
                this.resetSessionState();
                this.engine = null;
            }

            post({ type: 'log', payload: `Initializing Nemotron ASR (Profile: ${profile}, Beam: ${beamWidth})...` });
            this.engine = new AsrEngine({
                progress(label, loaded, total) {
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
                wasmPaths: 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/',
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

    private resetSessionState() {
        if (this.session && this.session._state) {
            try { this.session._state.cchTensor?.dispose?.(); } catch {}
            try { this.session._state.cctTensor?.dispose?.(); } catch {}
        }
        this.session = null;
        this.lastProcessedIndex = 0;
        this.lastEmittedPartial = '';
    }

    async transcribe(audioData: Float32Array, asrLanguage: string, promptLanguage: string, isFinal: boolean = true) {
        if (!this.engine) {
            post({ type: 'error', payload: 'Nemotron engine not initialized.' });
            return;
        }

        // Keep only final requests in queue; latest interim audioData supersedes older interim items
        this.processingQueue = this.processingQueue.filter(req => req.isFinal);
        this.processingQueue.push({ audioData, asrLanguage, promptLanguage, isFinal });
        this.processQueue();
    }

    /**
     * Runs a non-destructive lookahead preview on any unconsumed mel frames + 16 frames (160ms)
     * of silence right-context so the FastConformer RNN-T immediately emits the latest spoken words
     * without waiting for the next 560ms block boundary or mutating the session's committed state.
     */
    private async runSpeculativePreview(shouldConvertToTraditional: boolean): Promise<string> {
        const eng = this.engine as any;
        const sess = this.session;
        if (!eng || !sess || !sess._started || !sess._state || !eng._enc || !eng._gpuEnc) {
            return '';
        }

        const s = sess._state;
        const remaining = sess._frames.length + sess._frameOffset - sess._consumed;
        const committedText = detok(s.emitted, eng._vocab).text || '';

        if (remaining <= 0 && s.ccl === 0) {
            return shouldConvertToTraditional ? toTraditionalChinese(committedText) : committedText;
        }

        const base = sess._consumed;
        const buf = eng._getEncBuf();
        const N_MELS = 128;
        for (let i = 0; i < eng._encIn; i++) {
            const gi = base - eng._cacheFrames + i;
            const li = gi - sess._frameOffset;
            const fr = gi >= 0 && li >= 0 && li < sess._frames.length ? sess._frames[li] : null;
            for (let m = 0; m < N_MELS; m++) {
                buf[i * N_MELS + m] = fr ? fr[m] : 0;
            }
        }

        // Include +16 frames (160ms) of zero-padded right-context lookahead to flush FastConformer's rc=13 holdback
        const previewValidNew = Math.min(eng._newFrames, Math.max(1, remaining) + 16);
        eng._uploadEncoderInputs(eng._cacheFrames + previewValidNew, s.ccl, s.langId);
        const g = eng._gpuEnc;
        const er = await eng._enc.run({
            audio_signal: g.audio.tensor,
            length: g.length.tensor,
            cache_last_channel: s.cchTensor || eng._f32(s.cch, [1, 24, 56, 1024]),
            cache_last_time: s.cctTensor || eng._f32(s.cct, [1, 24, 1024, 8]),
            cache_last_channel_len: g.cacheLen.tensor,
            lang_id: g.lang.tensor,
        });

        // Dispose temporary preview GPU tensors so s.cchTensor and s.cctTensor remain intact for the next real block
        try { er.cache_last_channel_next?.dispose?.(); } catch {}
        try { er.cache_last_time_next?.dispose?.(); } catch {}

        const previewState = {
            h: s.h.slice(),
            c: s.c.slice(),
            decOut: s.decOut ? s.decOut.slice() : null,
            emitted: [...s.emitted],
        };

        await eng._greedyDecode(er.outputs.data, er.outputs.dims[1], previewState, eng._encFrameBuf, null);
        const { text } = detok(previewState.emitted, eng._vocab);
        return shouldConvertToTraditional ? toTraditionalChinese(text) : text;
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
                this.resetSessionState();
            }

            if (!this.session) {
                this.session = this.engine!.session(langId);
                this.lastProcessedIndex = 0;
                this.lastEmittedPartial = '';
            }

            const eng = this.engine as any;
            const sess = this.session;
            const newChunk = audioData.subarray(this.lastProcessedIndex);
            this.lastProcessedIndex = audioData.length;

            if (newChunk.length > 0) {
                if (!sess._started) {
                    sess._state = await eng._newState(sess._langId);
                    const { StreamingMel } = await import("@jeffpeng3/nemotron-asr-core");
                    sess._mel = new StreamingMel(eng._melFB, eng._window);
                    sess._diag = { start: performance.now(), encoder: 0, joint: 0, decoder: 0 };
                    sess._started = true;
                }

                const newFrames = sess._mel.push(newChunk);
                for (const fr of newFrames) sess._frames.push(fr);

                // Process full blocks and emit streaming partial after EVERY block
                while (sess._frames.length + sess._frameOffset - sess._consumed >= eng._newFrames) {
                    if (this.abortCurrent) break;
                    await sess._runBlock(eng._newFrames);
                    sess._lastEmittedLen = sess._state.emitted.length;

                    if (sess._state.emitted.length > 0 && !this.abortCurrent) {
                        const rawBlockText = detok(sess._state.emitted, eng._vocab).text;
                        if (rawBlockText) {
                            const formatted = shouldConvertToTraditional ? toTraditionalChinese(rawBlockText) : rawBlockText;
                            if (formatted !== this.lastEmittedPartial) {
                                this.lastEmittedPartial = formatted;
                                post({ type: 'transcription-partial', payload: formatted });
                                await new Promise(r => setTimeout(r, 0));
                            }
                        }
                    }
                }
            }

            if (isFinal) {
                // Append 160ms of trailing silence (16 mel frames) before finalizing so FastConformer rc=13 flushes the final word
                if (sess._started && sess._mel) {
                    const tailSilence = sess._mel.push(new Float32Array(2560));
                    for (const fr of tailSilence) sess._frames.push(fr);
                }

                const finalResult = await sess.end();
                const prevPartial = this.lastEmittedPartial;
                this.resetSessionState();

                if (!this.abortCurrent) {
                    const rawEndText = (finalResult?.text || '').trim();
                    const finalText = shouldConvertToTraditional ? toTraditionalChinese(rawEndText) : rawEndText;
                    const effectiveFinal = finalText || prevPartial;

                    // If final text added multiple characters at once that weren't streamed yet, stream them smoothly
                    if (effectiveFinal.length > prevPartial.length + 2) {
                        const startLen = Math.max(1, prevPartial.length);
                        const step = Math.max(1, Math.ceil((effectiveFinal.length - startLen) / 4));
                        for (let len = startLen + step; len < effectiveFinal.length; len += step) {
                            if (this.abortCurrent) break;
                            post({ type: 'transcription-partial', payload: effectiveFinal.slice(0, len) });
                            await new Promise(r => setTimeout(r, 12));
                        }
                    }

                    post({ type: 'transcription', payload: { text: effectiveFinal, isFinal: true } });
                }
            } else {
                if (!this.abortCurrent) {
                    // Run non-destructive speculative lookahead on remaining unconsumed frames so live words appear immediately
                    const previewText = await this.runSpeculativePreview(shouldConvertToTraditional);
                    const outText = previewText || this.lastEmittedPartial;
                    if (outText) {
                        this.lastEmittedPartial = outText;
                    }
                    post({ type: 'transcription-partial', payload: outText });
                }
            }
        } catch (error) {
            console.error('Nemotron Transcription Error:', error);
            post({ type: 'error', payload: `Nemotron transcription error: ${error}` });
            this.resetSessionState();
        } finally {
            this.isProcessing = false;
            this.abortCurrent = false;
            this.processQueue();
        }
    }

    async unload() {
        this.resetSessionState();
        this.engine = null;
        post({ type: 'unloaded', payload: true });
    }

    cancel() {
        this.processingQueue = [];
        if (this.isProcessing) {
            this.abortCurrent = true;
        }
        this.resetSessionState();
        post({ type: 'transcription-partial', payload: '' });
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
