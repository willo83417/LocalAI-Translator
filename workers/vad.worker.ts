import { createVad } from "@fluidinference/fluidvad";
// @ts-ignore
import fluidvadWasmUrl from "@fluidinference/fluidvad/dist/fluidvad_bg.wasm?url";

let cachedWasmBuffer: ArrayBuffer | null = null;
let vadInstance: any = null;
let currentMinSilenceDuration = 0;

let audioPort: MessagePort | null = null;
let isRecording = false;
let isAsrBusy = false;
let streamIntervalFrames = 12;
let hasStreamCallback = false;

// Contiguous pre-allocated buffer (up to 35 seconds at 16kHz = 560,000 samples)
const MAX_SAMPLES = 16000 * 35;
let sampleBuffer = new Float32Array(MAX_SAMPLES);
let totalSamples = 0;
let speechStartOffset = 0;
let lastStreamedLength = 0;
let lastSpeechSampleCount = 0;
let hasDetectedSpeech = false;
let framesSinceLastStream = 0;

async function ensureVadInitialized(isMobileDevice: boolean) {
    const targetSilence = isMobileDevice ? 1.15 : 1.35;
    if (vadInstance && currentMinSilenceDuration === targetSilence) {
        vadInstance.reset();
        return vadInstance;
    }

    if (!cachedWasmBuffer) {
        const res = await fetch(fluidvadWasmUrl);
        cachedWasmBuffer = await res.arrayBuffer();
    }

    if (vadInstance) {
        try { vadInstance.free?.(); } catch {}
        vadInstance = null;
    }

    vadInstance = await createVad(
        {
            threshold: 0.5,
            minSilenceDuration: targetSilence,
            maxSpeechDuration: 30,
            speechPadding: 0.35
        },
        { wasm: cachedWasmBuffer.slice(0) }
    );
    currentMinSilenceDuration = targetSilence;
    return vadInstance;
}

function getActiveAudioSlice(): Float32Array {
    if (totalSamples <= 0) return new Float32Array(0);
    return sampleBuffer.slice(0, totalSamples);
}

function handleAudioFrame(buf: ArrayBuffer) {
    if (!isRecording || !vadInstance) return;

    const samples = new Float32Array(buf);
    if (totalSamples + samples.length > sampleBuffer.length) {
        const nextBuf = new Float32Array(sampleBuffer.length + 16000 * 15);
        nextBuf.set(sampleBuffer.subarray(0, totalSamples), 0);
        sampleBuffer = nextBuf;
    }

    sampleBuffer.set(samples, totalSamples);
    totalSamples += samples.length;

    const events = vadInstance.push(samples);
    for (const event of events) {
        if (event.isStart) {
            hasDetectedSpeech = true;
            lastSpeechSampleCount = totalSamples;
        } else {
            // SpeechEnd detected by VAD
            isRecording = false;
            const finalSamples = getActiveAudioSlice();
            (self as any).postMessage(
                { type: 'speech-end', samples: finalSamples },
                [finalSamples.buffer]
            );
            return;
        }
    }

    if (vadInstance.isSpeaking) {
        hasDetectedSpeech = true;
        lastSpeechSampleCount = totalSamples;
    }

    // Only treat as unstreamed speech if actual speech samples were recorded after lastStreamedLength
    const hasUnstreamedSpeech = hasDetectedSpeech && (lastSpeechSampleCount - lastStreamedLength >= 2048);

    if (hasStreamCallback && (vadInstance.isSpeaking || hasUnstreamedSpeech)) {
        if (!isAsrBusy) {
            framesSinceLastStream++;
            if (framesSinceLastStream >= streamIntervalFrames) {
                framesSinceLastStream = 0;
                lastStreamedLength = totalSamples;
                const chunkSamples = getActiveAudioSlice();
                (self as any).postMessage(
                    { type: 'stream-chunk', samples: chunkSamples },
                    [chunkSamples.buffer]
                );
            }
        }
    }
}

self.onmessage = async (e: MessageEvent) => {
    const { type, payload } = e.data || {};

    switch (type) {
        case 'warmup': {
            try {
                await ensureVadInitialized(Boolean(payload?.isMobileDevice));
            } catch {}
            break;
        }
        case 'start': {
            const { isMobileDevice, intervalFrames, enableStreaming } = payload || {};
            const port: MessagePort | undefined = e.ports?.[0] || payload?.port;

            await ensureVadInitialized(Boolean(isMobileDevice));

            totalSamples = 0;
            speechStartOffset = 0;
            lastStreamedLength = 0;
            lastSpeechSampleCount = 0;
            hasDetectedSpeech = false;
            framesSinceLastStream = 0;
            isAsrBusy = false;
            streamIntervalFrames = intervalFrames || 12;
            hasStreamCallback = Boolean(enableStreaming);
            isRecording = true;

            if (audioPort) {
                try { audioPort.onmessage = null; audioPort.close(); } catch {}
            }
            if (port) {
                audioPort = port;
                audioPort.onmessage = (ev: MessageEvent) => {
                    if (ev.data instanceof ArrayBuffer) {
                        handleAudioFrame(ev.data);
                    }
                };
                audioPort.start?.();
            }
            break;
        }
        case 'set-busy': {
            isAsrBusy = Boolean(payload);
            if (isAsrBusy) {
                framesSinceLastStream = 0;
            }
            break;
        }
        case 'stop': {
            isRecording = false;
            if (audioPort) {
                try { audioPort.onmessage = null; audioPort.close(); } catch {}
                audioPort = null;
            }
            const finalSamples = getActiveAudioSlice();
            totalSamples = 0;
            speechStartOffset = 0;
            (self as any).postMessage(
                { type: 'stopped', samples: finalSamples },
                [finalSamples.buffer]
            );
            break;
        }
        default:
            break;
    }
};
