// Polyfill SharedArrayBuffer if not defined in worker scope
if (typeof (globalThis as any).SharedArrayBuffer === 'undefined') {
    (globalThis as any).SharedArrayBuffer = ArrayBuffer;
}

// CRITICAL: Import from 'onnxruntime-web/webgpu' (ort.webgpu.bundle.min.mjs).
// Importing the bare 'onnxruntime-web' defaults to the wasm-only bundle, which fails to run on GPU
// and returns flat/empty logits (argmax token 0 = '!'), causing repeated exclamation marks.
import * as ort from 'onnxruntime-web/webgpu';
import { toTraditionalChinese, isTraditionalChinese } from '../utils/chineseConverter';

const PROMPT_MAP: Record<string, string> = {
    'zh-TW': '請使用(zh-Hant)繁體中文輸出。',
    'zh-HK': '請使用(zh-Hant-HK)香港繁體中文輸出。',
    'zh-Hans': '请使用(zh-Hans)简体中文输出。'
};

interface WorkerMessage {
    type: 'load' | 'transcribe' | 'unload' | 'cancel';
    payload?: any;
}

interface AppMessage {
    type: 'log' | 'transcription' | 'transcription-partial' | 'loaded' | 'error' | 'progress' | 'unloaded';
    payload: any;
}

const post = (message: AppMessage) => self.postMessage(message);

const CACHE_NAME = 'qwen3-asr-0.6b-onnx';
const HF_BASE_URL = 'https://huggingface.co/jiangzhuo9357/Qwen3-ASR-0.6B-ONNX/resolve/main/';

// Approximate byte sizes for progress estimation
const FILE_SIZES: Record<string, number> = {
    'prompt_config.json': 2578,
    'mel_filters.json': 132427,
    'tokenizer.json': 11429377,
    'tokenizer_config.json': 12488,
    'embed_scales.f32.bin': 607744,
    'embed_tokens.int8.bin': 155582464,
    // q4f16 files
    'encoder.fp16.onnx': 376261247,
    'decoder_init.q4f16.onnx': 348427,
    'decoder_step.q4f16.onnx': 350380,
    'decoder_weights.q4f16.data': 344670208,
    // q4 files
    'encoder.onnx': 745766355,
    'decoder_init.int4.onnx': 353078,
    'decoder_step.int4.onnx': 354803,
    'decoder_weights.int4.data': 382035968,
};

// ─── Half-precision float utilities (IEEE 754 binary16) ─────────────────────────
const hasFloat16Array = typeof (globalThis as any).Float16Array !== 'undefined';

function f32ToF16Fallback(src: Float32Array): Uint16Array {
    const out = new Uint16Array(src.length);
    const f32 = new Float32Array(1);
    const u32 = new Uint32Array(f32.buffer);
    for (let i = 0; i < src.length; i++) {
        f32[0] = src[i];
        const x = u32[0];
        const sign = (x >>> 16) & 0x8000;
        const exp = (x >>> 23) & 0xff;
        const mant = x & 0x7fffff;
        if (exp === 0xff) {
            out[i] = sign | 0x7c00 | (mant ? 0x200 : 0);
            continue;
        }
        const e = exp - 112; // rebias 127 -> 15
        if (e >= 0x1f) {
            out[i] = sign | 0x7c00;
            continue;
        }
        if (e <= 0) {
            if (e < -10) {
                out[i] = sign;
                continue;
            }
            const m = mant | 0x800000;
            const shift = 14 - e;
            let half = m >>> shift;
            const rem = m & ((1 << shift) - 1);
            const halfway = 1 << (shift - 1);
            if (rem > halfway || (rem === halfway && (half & 1))) half++;
            out[i] = sign | half;
            continue;
        }
        let half = (e << 10) | (mant >>> 13);
        const rem = mant & 0x1fff;
        if (rem > 0x1000 || (rem === 0x1000 && (half & 1))) half++;
        out[i] = sign | half;
    }
    return out;
}

function f32ToF16(src: Float32Array): Uint16Array {
    if (hasFloat16Array) {
        const F16 = (globalThis as any).Float16Array;
        return new Uint16Array(new F16(src).buffer);
    }
    return f32ToF16Fallback(src);
}

function f16ToF32(src: Uint16Array): Float32Array {
    if (hasFloat16Array) {
        const F16 = (globalThis as any).Float16Array;
        return new Float32Array(new F16(src.buffer, src.byteOffset, src.length));
    }
    const out = new Float32Array(src.length);
    for (let i = 0; i < src.length; i++) {
        const h = src[i];
        const s = h & 0x8000 ? -1 : 1;
        const e = (h >> 10) & 0x1f;
        const f = h & 0x3ff;
        out[i] = e === 0 ? s * Math.pow(2, -14) * (f / 1024) : e === 31 ? (f ? NaN : s * Infinity) : s * Math.pow(2, e - 15) * (1 + f / 1024);
    }
    return out;
}

function logitsAsF32(t: ort.Tensor): Float32Array {
    if (t.type === 'float16') return f16ToF32(t.data as Uint16Array);
    return t.data instanceof Float32Array ? t.data : Float32Array.from(t.data as ArrayLike<number>);
}

function argmax(a: ArrayLike<number>): number {
    let best = 0;
    for (let i = 1; i < a.length; i++) {
        if (a[i] > a[best]) best = i;
    }
    return best;
}

// ─── Minimal Byte-Level BPE Decoder (Matching Sokuji) ───────────────────────────
interface TokenizerJsonLike {
    model: { vocab: Record<string, number> };
    added_tokens?: { id: number; content: string }[];
}

interface BpeDecoder {
    special: Map<number, string>;
    decode(ids: number[], opts?: { skipSpecial?: boolean }): string;
}

function charToByteTable(): Map<string, number> {
    const bs: number[] = [];
    for (let b = '!'.charCodeAt(0); b <= '~'.charCodeAt(0); b++) bs.push(b);
    for (let b = 0xa1; b <= 0xac; b++) bs.push(b);
    for (let b = 0xae; b <= 0xff; b++) bs.push(b);
    const cs = bs.slice();
    let n = 0;
    for (let b = 0; b < 256; b++) {
        if (!bs.includes(b)) {
            bs.push(b);
            cs.push(256 + n);
            n++;
        }
    }
    const table = new Map<string, number>();
    for (let i = 0; i < bs.length; i++) table.set(String.fromCharCode(cs[i]), bs[i]);
    return table;
}

function createBpeDecoder(tokenizerJson: TokenizerJsonLike): BpeDecoder {
    const idToToken: string[] = [];
    for (const [tok, id] of Object.entries(tokenizerJson.model.vocab)) idToToken[id] = tok;
    const special = new Map<number, string>();
    for (const t of tokenizerJson.added_tokens ?? []) special.set(t.id, t.content);
    const charToByte = charToByteTable();
    const utf8 = new TextDecoder('utf-8');

    return {
        special,
        decode(ids: number[], { skipSpecial = true } = {}) {
            const bytes: number[] = [];
            let out = '';
            const flush = () => {
                if (bytes.length) {
                    out += utf8.decode(new Uint8Array(bytes));
                    bytes.length = 0;
                }
            };

            for (const id of ids) {
                const sp = special.get(id);
                if (sp !== undefined) {
                    flush();
                    if (!skipSpecial) out += sp;
                    continue;
                }
                const tok = idToToken[id];
                if (tok === undefined) {
                    flush();
                    out += `<unk:${id}>`;
                    continue;
                }
                for (const ch of tok) {
                    const b = charToByte.get(ch);
                    if (b === undefined) {
                        flush();
                        out += ch;
                    } else {
                        bytes.push(b);
                    }
                }
            }
            flush();
            return out;
        }
    };
}

// ─── Whisper-style Log-Mel Spectrogram Front End ───────────────────────────────
export interface MelFilterbank {
    n_mels: number;
    n_freqs: number;
    data: number[][];
}

export interface LogMel {
    data: Float32Array;
    nMels: number;
    T: number;
}

const N_FFT = 400;
const HOP = 160;
const N_FREQS = N_FFT / 2 + 1; // 201
let dftTablesCache: { cos: Float32Array; sin: Float32Array; window: Float32Array } | null = null;

function getDftTables() {
    if (dftTablesCache) return dftTablesCache;
    const cos = new Float32Array(N_FREQS * N_FFT);
    const sin = new Float32Array(N_FREQS * N_FFT);
    for (let k = 0; k < N_FREQS; k++) {
        for (let n = 0; n < N_FFT; n++) {
            const a = (2 * Math.PI * k * n) / N_FFT;
            cos[k * N_FFT + n] = Math.cos(a);
            sin[k * N_FFT + n] = Math.sin(a);
        }
    }
    const window = new Float32Array(N_FFT);
    for (let n = 0; n < N_FFT; n++) window[n] = 0.5 - 0.5 * Math.cos((2 * Math.PI * n) / N_FFT); // periodic Hann
    dftTablesCache = { cos, sin, window };
    return dftTablesCache;
}

export function logMel(audio: Float32Array, filters: MelFilterbank): LogMel {
    if (filters.n_freqs !== N_FREQS) {
        throw new Error(`mel filterbank has ${filters.n_freqs} bins, expected ${N_FREQS}`);
    }
    const { cos, sin, window } = getDftTables();
    const nMels = filters.n_mels;
    const fb = new Float32Array(nMels * N_FREQ_COUNT);
    for (let m = 0; m < nMels; m++) {
        for (let k = 0; k < N_FREQ_COUNT; k++) fb[m * N_FREQ_COUNT + k] = filters.data[m][k];
    }
    const pad = N_FFT / 2;
    const L = audio.length;
    const padded = new Float32Array(L + 2 * pad);
    for (let i = 0; i < pad; i++) padded[i] = audio[pad - i]; // reflect padding
    padded.set(audio, pad);
    for (let i = 0; i < pad; i++) padded[pad + L + i] = audio[L - 2 - i];

    const nFrames = 1 + Math.floor((padded.length - N_FFT) / HOP);
    const T = Math.max(1, nFrames - 1); // WhisperFeatureExtractor drops the last frame
    const out = new Float32Array(nMels * T);
    const frame = new Float32Array(N_FFT);
    const power = new Float32Array(N_FREQ_COUNT);
    let gmax = -Infinity;

    for (let t = 0; t < T; t++) {
        const off = t * HOP;
        for (let n = 0; n < N_FFT; n++) frame[n] = padded[off + n] * window[n];
        for (let k = 0; k < N_FREQ_COUNT; k++) {
            let re = 0;
            let im = 0;
            const base = k * N_FFT;
            for (let n = 0; n < N_FFT; n++) {
                const s = frame[n];
                re += s * cos[base + n];
                im -= s * sin[base + n];
            }
            power[k] = re * re + im * im;
        }
        for (let m = 0; m < nMels; m++) {
            let sum = 0;
            const fbase = m * N_FREQ_COUNT;
            for (let k = 0; k < N_FREQ_COUNT; k++) sum += fb[fbase + k] * power[k];
            const val = Math.log10(Math.max(sum, 1e-10));
            out[m * T + t] = val;
            if (val > gmax) gmax = val;
        }
    }

    // Dynamic range clamp to (gmax - 8.0), scale (x + 4.0) / 4.0
    const minVal = gmax - 8.0;
    for (let i = 0; i < out.length; i++) {
        const clamped = Math.max(out[i], minVal);
        out[i] = (clamped + 4.0) / 4.0;
    }

    return { data: out, nMels, T };
}

const N_FREQ_COUNT = 201;

// ─── Prompt Configuration & Language Prefix Mapping ────────────────────────────
export interface Qwen3AsrPromptConfig {
    layout_version: number;
    mel?: { sample_rate: number; n_fft: number; hop_length: number; n_mels: number; filters_file: string; drop_last_frame: boolean };
    prompt: {
        prefix_ids: number[];
        suffix_ids: number[];
        audio_pad_id: number;
        asr_text_id: number;
        eos_ids: number[];
        max_new_tokens: number;
    };
    language_prefix_ids: Record<string, number[]>;
    audio_tokens: { conv_window: number; tokens_per_window: number };
    embedding: { file: string; dtype: 'int8' | 'float16' | 'float32'; shape: [number, number]; scales_file?: string };
    decoder: { num_layers: number; num_key_value_heads: number; head_dim: number; hidden_size: number; vocab_size: number };
    variants: Record<string, { encoder: string; decoder_init: string; decoder_step: string; weights: string; required_features: string[] }>;
}

const convOut = (t: number): number => Math.floor((t + 1) / 2);

export function audioTokenCount(melFrames: number, a: Qwen3AsrPromptConfig['audio_tokens']): number {
    const remainder = melFrames % a.conv_window;
    const tail = convOut(convOut(convOut(remainder)));
    return tail + Math.floor(melFrames / a.conv_window) * a.tokens_per_window;
}

export interface BuiltPrompt {
    ids: number[];
    audioStart: number;
    forced: boolean;
}

export function buildPromptIds(nAudio: number, cfg: Qwen3AsrPromptConfig, forceLang?: string): BuiltPrompt {
    const p = cfg.prompt;
    const ids = [...p.prefix_ids];
    const audioStart = ids.length;
    for (let i = 0; i < nAudio; i++) ids.push(p.audio_pad_id);
    ids.push(...p.suffix_ids);
    const forcedIds = forceLang ? cfg.language_prefix_ids[forceLang] : undefined;
    if (forcedIds) ids.push(...forcedIds);
    return { ids, audioStart, forced: Boolean(forcedIds) };
}

export interface SplitOutput {
    prefixIds: number[];
    textIds: number[];
    detectedPrefix: boolean;
}

export function splitGenerated(ids: number[], cfg: Qwen3AsrPromptConfig): SplitOutput {
    const eos = new Set(cfg.prompt.eos_ids);
    const body = ids.filter((id) => !eos.has(id));
    const cut = body.indexOf(cfg.prompt.asr_text_id);
    if (cut < 0) return { prefixIds: [], textIds: body, detectedPrefix: false };
    return { prefixIds: body.slice(0, cut), textIds: body.slice(cut + 1), detectedPrefix: true };
}

const LANG_ALIASES: Record<string, string> = {
    cantonese: 'yue',
    tl: 'fil',
    jap: 'ja',
    'zh-cn': 'zh',
    'zh-tw': 'zh',
    'zh-hk': 'yue',
    'zh-hans': 'zh',
    'zh-hant': 'zh',
    'en-us': 'en',
    'en-gb': 'en',
    'ja-jp': 'ja',
    'ko-kr': 'ko',
};

export function normalizeLangForPrefix(lang: string | undefined, cfg: Qwen3AsrPromptConfig): string | undefined {
    if (!lang || lang === 'auto' || lang === 'Auto Detect' || lang === 'autodetect') return undefined;
    const lower = lang.toLowerCase().trim();
    const primary = (LANG_ALIASES[lower] ?? lower).split(/[-_]/)[0];
    return primary in cfg.language_prefix_ids ? primary : undefined;
}

// ─── Qwen3-ASR Engine Class ────────────────────────────────────────────────────
class Qwen3AsrEngine {
    private variant: 'q4f16' | 'q4' = 'q4';
    private cfg: Qwen3AsrPromptConfig | null = null;
    private filters: MelFilterbank | null = null;
    private decoder: BpeDecoder | null = null;
    private eos: Set<number> = new Set();
    private hidden: number = 1024;

    private embI8: Int8Array | null = null;
    private embScales: Float32Array | null = null;
    private embF16: Uint16Array | null = null;
    private embF32: Float32Array | null = null;

    private encoderSession: ort.InferenceSession | null = null;
    private decoderInitSession: ort.InferenceSession | null = null;
    private decoderStepSession: ort.InferenceSession | null = null;
    private encoderMelType: string = 'float32';

    private isLoading = false;
    private isProcessing = false;
    private abortCurrent = false;
    private currentProcessingIsFinal = false;

    async checkShaderF16Support(): Promise<boolean> {
        try {
            if (typeof navigator !== 'undefined' && (navigator as any).gpu) {
                const adapter = await (navigator as any).gpu.requestAdapter();
                if (adapter && adapter.features && adapter.features.has('shader-f16')) {
                    return true;
                }
            }
        } catch (e) {
            console.warn('[Qwen3 ASR] GPU adapter detection error:', e);
        }
        return false;
    }

    private embedRowInto(id: number, out: Float32Array, offset: number): void {
        const h = this.hidden;
        const o = id * h;
        if (this.embI8 && this.embScales) {
            const s = this.embScales[id] || 1.0;
            for (let i = 0; i < h; i++) out[offset + i] = this.embI8[o + i] * s;
        } else if (this.embF16) {
            out.set(f16ToF32(this.embF16.subarray(o, o + h)), offset);
        } else if (this.embF32) {
            out.set(this.embF32.subarray(o, o + h), offset);
        }
    }

    async load() {
        if (this.isLoading) return;
        if (this.encoderSession && this.decoderInitSession && this.decoderStepSession) {
            post({ type: 'loaded', payload: true });
            return;
        }

        this.isLoading = true;
        try {
            post({ type: 'log', payload: 'Checking WebGPU capabilities for Qwen3-ASR...' });
            const hasShaderF16 = await this.checkShaderF16Support();
            this.variant = hasShaderF16 ? 'q4f16' : 'q4';
            post({
                type: 'log',
                payload: `Selected variant: ${this.variant} (shader-f16 supported: ${hasShaderF16})`
            });

            // List required files
            const requiredFiles = [
                'prompt_config.json',
                'mel_filters.json',
                'tokenizer.json',
                'tokenizer_config.json',
                'embed_scales.f32.bin',
                'embed_tokens.int8.bin',
            ];

            if (this.variant === 'q4f16') {
                requiredFiles.push(
                    'encoder.fp16.onnx',
                    'decoder_init.q4f16.onnx',
                    'decoder_step.q4f16.onnx',
                    'decoder_weights.q4f16.data'
                );
            } else {
                requiredFiles.push(
                    'encoder.onnx',
                    'decoder_init.int4.onnx',
                    'decoder_step.int4.onnx',
                    'decoder_weights.int4.data'
                );
            }

            const totalBytesAll = requiredFiles.reduce((acc, f) => acc + (FILE_SIZES[f] || 1000000), 0);
            let downloadedOverall = 0;

            const cache = await caches.open(CACHE_NAME);
            const fileBuffers: Record<string, ArrayBuffer> = {};

            for (const file of requiredFiles) {
                const url = `${HF_BASE_URL}${file}`;
                const matched = await cache.match(url);

                if (matched) {
                    post({
                        type: 'log',
                        payload: `[Qwen3 ASR] Loaded from cache: ${file}`
                    });
                    fileBuffers[file] = await matched.arrayBuffer();
                    downloadedOverall += FILE_SIZES[file] || 1000000;
                    const percent = Math.min(95, (downloadedOverall / totalBytesAll) * 100);
                    post({
                        type: 'progress',
                        payload: { status: 'progress', file: `Loaded ${file} from cache`, progress: percent }
                    });
                } else {
                    post({
                        type: 'log',
                        payload: `[Qwen3 ASR] Downloading ${file}...`
                    });
                    post({
                        type: 'progress',
                        payload: {
                            status: 'progress',
                            file: `Downloading ${file}...`,
                            progress: Math.min(95, (downloadedOverall / totalBytesAll) * 100)
                        }
                    });

                    const response = await fetch(url);
                    if (!response.ok) {
                        throw new Error(`Failed to download ${file}: HTTP ${response.status}`);
                    }

                    const contentLength = Number(response.headers.get('content-length')) || FILE_SIZES[file] || 0;
                    const reader = response.body?.getReader();

                    if (!reader) {
                        const buf = await response.arrayBuffer();
                        fileBuffers[file] = buf;
                        await cache.put(url, new Response(buf));
                        downloadedOverall += contentLength;
                    } else {
                        const chunks: Uint8Array[] = [];
                        let fileDownloaded = 0;

                        while (true) {
                            const { done, value } = await reader.read();
                            if (done) break;
                            if (value) {
                                chunks.push(value);
                                fileDownloaded += value.length;
                                const filePercent = contentLength > 0 ? (fileDownloaded / contentLength) * 100 : 50;
                                const totalPercent = Math.min(95, ((downloadedOverall + fileDownloaded) / totalBytesAll) * 100);
                                post({
                                    type: 'progress',
                                    payload: {
                                        status: 'progress',
                                        file: `Downloading ${file} (${Math.round(filePercent)}%)`,
                                        progress: totalPercent
                                    }
                                });
                            }
                        }

                        const totalBuf = new Uint8Array(fileDownloaded);
                        let offset = 0;
                        for (const chunk of chunks) {
                            totalBuf.set(chunk, offset);
                            offset += chunk.length;
                        }

                        fileBuffers[file] = totalBuf.buffer;
                        await cache.put(url, new Response(totalBuf.buffer));
                        downloadedOverall += fileDownloaded;
                    }
                }
            }

            post({ type: 'progress', payload: { status: 'progress', file: 'Parsing configs and tokenizer...', progress: 95 } });

            const textDecoder = new TextDecoder();
            this.cfg = JSON.parse(textDecoder.decode(fileBuffers['prompt_config.json'])) as Qwen3AsrPromptConfig;
            this.filters = JSON.parse(textDecoder.decode(fileBuffers['mel_filters.json'])) as MelFilterbank;
            this.decoder = createBpeDecoder(JSON.parse(textDecoder.decode(fileBuffers['tokenizer.json'])));
            this.eos = new Set(this.cfg.prompt.eos_ids);
            this.hidden = this.cfg.decoder?.hidden_size || 1024;

            // Load Embedding table
            const emb = this.cfg.embedding;
            const embBytes = new Uint8Array(fileBuffers[emb.file]);
            this.embI8 = emb.dtype === 'int8' ? new Int8Array(embBytes.buffer, embBytes.byteOffset, embBytes.byteLength) : null;
            this.embScales = emb.dtype === 'int8' && emb.scales_file
                ? new Float32Array(fileBuffers[emb.scales_file])
                : null;
            this.embF16 = emb.dtype === 'float16' ? new Uint16Array(embBytes.buffer, embBytes.byteOffset, embBytes.byteLength / 2) : null;
            this.embF32 = emb.dtype === 'float32' ? new Float32Array(embBytes.buffer, embBytes.byteOffset, embBytes.byteLength / 4) : null;

            // Setup ONNX Runtime Web
            ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/';
			ort.env.webgpu.powerPreference = 'high-performance';
            ort.env.debug = false;
			ort.env.logLevel = 'error';

            post({ type: 'progress', payload: { status: 'progress', file: 'Creating WebGPU Inference Sessions...', progress: 97 } });

            const encoderFileName = this.variant === 'q4f16' ? 'encoder.fp16.onnx' : 'encoder.onnx';
            const decoderInitFileName = this.variant === 'q4f16' ? 'decoder_init.q4f16.onnx' : 'decoder_init.int4.onnx';
            const decoderStepFileName = this.variant === 'q4f16' ? 'decoder_step.q4f16.onnx' : 'decoder_step.int4.onnx';
            const weightsFileName = this.variant === 'q4f16' ? 'decoder_weights.q4f16.data' : 'decoder_weights.int4.data';
            const weightsBytes = new Uint8Array(fileBuffers[weightsFileName]);

            // WebGPU Configuration matching Sokuji:
            // Keeping KV cache in 'gpu-buffer' completely prevents CPU-GPU memory thrashing
            const gpuOpts = (extra: Record<string, unknown> = {}) => ({
                executionProviders: ['webgpu'],
                graphOptimizationLevel: 'all' as const,
                ...extra,
            });
            const kvOnGpu = {
                preferredOutputLocation: {
                    present_keys: 'gpu-buffer',
                    present_values: 'gpu-buffer'
                }
            };

            post({ type: 'log', payload: `Initializing WebGPU sessions with ${this.variant}...` });

            try {
                this.encoderSession = await ort.InferenceSession.create(fileBuffers[encoderFileName], gpuOpts());
                this.decoderInitSession = await ort.InferenceSession.create(
                    fileBuffers[decoderInitFileName],
                    gpuOpts({ externalData: [{ path: weightsFileName, data: weightsBytes }], ...kvOnGpu })
                );
                this.decoderStepSession = await ort.InferenceSession.create(
                    fileBuffers[decoderStepFileName],
                    gpuOpts({ externalData: [{ path: weightsFileName, data: weightsBytes }], ...kvOnGpu })
                );
                this.encoderMelType = (this.encoderSession as any).inputMetadata?.[0]?.type ?? 'float32';
                post({ type: 'log', payload: 'WebGPU sessions created successfully with GPU-buffered KV cache.' });
            } catch (webgpuErr: any) {
                console.warn('[Qwen3 ASR] WebGPU session creation failed, falling back to WASM:', webgpuErr);
                post({ type: 'log', payload: `WebGPU failed (${webgpuErr?.message || webgpuErr}), falling back to WASM...` });

                const wasmOpts = (extra: Record<string, unknown> = {}) => ({
                    executionProviders: ['wasm'],
                    graphOptimizationLevel: 'all' as const,
                    enableCpuMemArena: true,
                    ...extra,
                });
                this.encoderSession = await ort.InferenceSession.create(fileBuffers[encoderFileName], wasmOpts());
                this.decoderInitSession = await ort.InferenceSession.create(
                    fileBuffers[decoderInitFileName],
                    wasmOpts({ externalData: [{ path: weightsFileName, data: weightsBytes }] })
                );
                this.decoderStepSession = await ort.InferenceSession.create(
                    fileBuffers[decoderStepFileName],
                    wasmOpts({ externalData: [{ path: weightsFileName, data: weightsBytes }] })
                );
                this.encoderMelType = (this.encoderSession as any).inputMetadata?.[0]?.type ?? 'float32';
            }

            // WebGPU Warmup: run 1 second of silence so shaders are pre-compiled and subsequent decodes are fast!
            post({ type: 'log', payload: 'Warming up WebGPU shaders with silence...' });
            try {
                await this.transcribeAudioInternal(new Float32Array(16000), undefined, undefined, true);
            } catch (warmupErr) {
                console.warn('[Qwen3 ASR] Warmup skipped:', warmupErr);
            }

            post({ type: 'loaded', payload: true });
            post({ type: 'log', payload: 'Qwen3-ASR model loaded and ready!' });
        } catch (err: any) {
            console.error('[Qwen3 ASR Load Error]:', err);
            post({ type: 'error', payload: `Error loading Qwen3-ASR: ${err?.message || err}` });
            this.encoderSession = null;
            this.decoderInitSession = null;
            this.decoderStepSession = null;
        } finally {
            this.isLoading = false;
        }
    }

    async transcribe(audio: Float32Array, asrLanguage: string, promptLanguage: string, isFinal: boolean = true) {
        if (!this.encoderSession || !this.decoderInitSession || !this.decoderStepSession || !this.filters || !this.cfg || !this.decoder) {
            post({ type: 'error', payload: 'Qwen3 ASR engine is not loaded.' });
            return;
        }

        // If audio is too short (< 0.6s at 16kHz) and not final, skip to avoid hallucinations
        if (!isFinal && audio.length < 9600) {
            return;
        }

        if (this.isProcessing) {
            if (!isFinal) return; // Skip non-final frames if already busy
            // If final request arrived while non-final is processing, abort the non-final generation immediately!
            if (!this.currentProcessingIsFinal) {
                this.abortCurrent = true;
            }
            while (this.isProcessing) {
                await new Promise((res) => setTimeout(res, 10));
            }
        }

        this.isProcessing = true;
        this.abortCurrent = false;
        this.currentProcessingIsFinal = isFinal;
        try {
            await this.transcribeAudioInternal(audio, asrLanguage, promptLanguage, false, isFinal);
        } catch (err: any) {
            if (err?.message !== 'ABORTED') {
                console.error('[Qwen3 ASR Transcribe Error]:', err);
                post({ type: 'error', payload: `Transcription error: ${err?.message || err}` });
            }
        } finally {
            this.isProcessing = false;
            this.abortCurrent = false;
        }
    }

    private async transcribeAudioInternal(
        audio: Float32Array,
        asrLanguage?: string,
        promptLanguage?: string,
        isWarmup: boolean = false,
        isFinal: boolean = true
    ) {
        if (audio.length < 320) {
            if (!isWarmup) post({ type: 'transcription', payload: { text: '', isFinal } });
            return;
        }

        const shouldConvertToTraditional = isTraditionalChinese(promptLanguage) || isTraditionalChinese(asrLanguage);

        // 1. Extract Whisper-style log-mel spectrogram
        const mel = logMel(audio, this.filters!);
        const melData = this.encoderMelType === 'float16' ? f32ToF16(mel.data) : mel.data;

        // 2. Run Encoder
        const melTensor = new ort.Tensor(this.encoderMelType as any, melData as any, [1, mel.nMels, mel.T]);
        const encResults = await this.encoderSession!.run({ mel: melTensor });
        const af = encResults.audio_features as ort.Tensor;
        const nAudio = af.dims[1];

        const afF32 = af.type === 'float16' ? f16ToF32(af.data as Uint16Array) : (af.data as Float32Array);

        // 3. Construct prompt tokens sequence (prefix + audio pads + suffix + language prefix)
        const targetLang = normalizeLangForPrefix(promptLanguage || asrLanguage, this.cfg!);
        const { ids, audioStart } = buildPromptIds(nAudio, this.cfg!, targetLang);

        // 4. Build prompt embeddings (dequantize text embeddings and splice audio features)
        const promptLen = ids.length;
        const promptEmbeds = new Float32Array(promptLen * this.hidden);
        for (let i = 0; i < promptLen; i++) {
            this.embedRowInto(ids[i], promptEmbeds, i * this.hidden);
        }
        promptEmbeds.set(afF32.subarray(0, nAudio * this.hidden), audioStart * this.hidden);

        // 5. Decode Deps & Input Types
        const initInputMeta = (this.decoderInitSession as any).inputMetadata;
        const stepInputMeta = (this.decoderStepSession as any).inputMetadata;
        const embType = initInputMeta?.[0]?.type ?? 'float32';
        const stepEmbType = stepInputMeta?.[0]?.type ?? embType;

        const makeEmbTensor = (data: Float32Array, type: string, dims: number[]) => {
            const tensorData = type === 'float16' ? f32ToF16(data) : data;
            return new ort.Tensor(type as any, tensorData as any, dims);
        };

        const makePosTensor = (startPos: number, len: number) => {
            const bigArray = new BigInt64Array(len);
            for (let i = 0; i < len; i++) bigArray[i] = BigInt(startPos + i);
            return new ort.Tensor('int64', bigArray, [1, len]);
        };

        // 6. Prefill step: decoder_init
        const initOut = await this.decoderInitSession!.run({
            input_embeds: makeEmbTensor(promptEmbeds, embType, [1, promptLen, this.hidden]),
            position_ids: makePosTensor(0, promptLen),
        });

        // out.logits in layout-v2 is already [1, 1, vocab_size] (last token)
        let nextToken = argmax(logitsAsF32(initOut.logits as ort.Tensor));
        const generatedIds = [nextToken];

        let pastK = initOut.present_keys as ort.Tensor;
        let pastV = initOut.present_values as ort.Tensor;
        let pos = promptLen;
        const maxTokens = isFinal ? Math.min(256, this.cfg!.prompt.max_new_tokens || 256) : 64;

        // 7. Autoregressive Greedy Generation Loop
        try {
            // Pre-allocate step buffers to avoid GC pressure and repeated allocations per token
            const stepRow = new Float32Array(this.hidden);
            const posBigArray = new BigInt64Array(1);

            while (!this.eos.has(nextToken) && generatedIds.length < maxTokens) {
                if (this.abortCurrent) {
                    throw new Error('ABORTED');
                }
                this.embedRowInto(nextToken, stepRow, 0);
                posBigArray[0] = BigInt(pos);

                const stepEmbedsTensor = makeEmbTensor(stepRow, stepEmbType, [1, 1, this.hidden]);
                const stepPosTensor = new ort.Tensor('int64', posBigArray, [1, 1]);

                const stepOut = await this.decoderStepSession!.run({
                    input_embeds: stepEmbedsTensor,
                    position_ids: stepPosTensor,
                    past_keys: pastK,
                    past_values: pastV,
                });

                // Dispose step input tensors and previous step's KV GPU buffers immediately to prevent memory growth
                stepEmbedsTensor.dispose?.();
                stepPosTensor.dispose?.();
                pastK.dispose?.();
                pastV.dispose?.();

                pastK = stepOut.present_keys as ort.Tensor;
                pastV = stepOut.present_values as ort.Tensor;
                nextToken = argmax(logitsAsF32(stepOut.logits as ort.Tensor));
                generatedIds.push(nextToken);
                pos++;

                // Emit partial progress every 4 tokens
                if (!isWarmup && generatedIds.length > 2 && generatedIds.length % 4 === 0) {
                    const { textIds } = splitGenerated(generatedIds, this.cfg!);
                    const partialText = this.decoder!.decode(textIds).trim();
                    if (partialText) {
                        const formattedPartial = shouldConvertToTraditional ? toTraditionalChinese(partialText) : partialText;
                        post({ type: 'transcription-partial', payload: formattedPartial });
                    }
                }
            }
        } finally {
            pastK.dispose?.();
            pastV.dispose?.();
        }

        if (isWarmup) return;

        // 8. Split generated tokens (cuts at <asr_text> and removes EOS)
        const { textIds } = splitGenerated(generatedIds, this.cfg!);
        const finalText = this.decoder!.decode(textIds).trim();
        const formattedFinal = shouldConvertToTraditional ? toTraditionalChinese(finalText) : finalText;

        post({ type: 'transcription', payload: { text: formattedFinal, isFinal } });
    }

    unload() {
        this.encoderSession = null;
        this.decoderInitSession = null;
        this.decoderStepSession = null;
        this.decoder = null;
        this.filters = null;
        this.cfg = null;
        post({ type: 'unloaded', payload: true });
    }
}

const engine = new Qwen3AsrEngine();

self.onmessage = async (e: MessageEvent<WorkerMessage>) => {
    const { type, payload } = e.data;
    switch (type) {
        case 'load':
            await engine.load();
            break;
        case 'transcribe':
            await engine.transcribe(payload.audio, payload.asrLanguage, payload.promptLanguage, payload.isFinal);
            break;
        case 'unload':
            engine.unload();
            break;
        case 'cancel':
            break;
        default:
            break;
    }
};
