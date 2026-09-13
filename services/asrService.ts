import { env, ModelRegistry } from '@huggingface/transformers';

// --- Cache Configuration ---
// Configure transformers.js to use WebGPU and the browser's cache.
// This is done here to ensure it's configured on the main thread as well as the worker.
env.backends.onnx.executionProviders = ['webgpu', 'wasm'];
env.useBrowserCache = true;
env.useWasmCache = true;
env.cacheDir = 'transformers-cache';

/**
 * Checks if a specific ASR model is already present in the browser cache
 * @param modelId The Hugging Face ID of the model to check.
 * @returns A promise that resolves to true if cached, false otherwise.
 */
export const checkAsrModelCacheStatus = async (modelId: string, quantization?: any): Promise<boolean> => {
    try {
        if (!('caches' in self)) return false;

        if (modelId === 'nemotron') {
            const cache = await caches.open('nemotron-asr-int4-1');
            const keys = await cache.keys();
            // nemotron-asr-core caches several files (encoder, decoder, joint, vocab, etc.)
            // We just check if the cache has at least one file, or maybe we can check for encoder.onnx
            const urls = keys.map(k => k.url);
            return urls.some(url => url.includes('encoder.onnx'));
        }

        if (modelId === 'qwen3' || modelId === 'qwen3-asr') {
            const cache = await caches.open('qwen3-asr-0.6b-onnx');
            const keys = await cache.keys();
            const urls = keys.map(k => k.url);
            return urls.some(url => url.includes('encoder') && (url.includes('encoder.onnx') || url.includes('encoder.fp16.onnx')));
        }

        // We do a robust custom Cache API check because ModelRegistry.is_pipeline_cached 
        // sometimes incorrectly returns false for Whisper models due to optional files.
        const cache = await caches.open('transformers-cache');
        const keys = await cache.keys();
        const urls = keys.map(k => k.url);

        // Check for essential base files
        const hasConfig = urls.some(url => url.includes(`${modelId}/resolve/main/config.json`));
        const hasTokenizer = urls.some(url => url.includes(`${modelId}/resolve/main/tokenizer.json`) || url.includes(`${modelId}/resolve/main/tokenizer_config.json`));
        
        if (!hasConfig || !hasTokenizer) {
            return false;
        }

        // Check for required model binaries depending on quantization
        if (quantization && typeof quantization === 'object') {
            const hasAllBinaries = Object.entries(quantization).every(([key, val]) => {
                const checkString1 = `${modelId}/resolve/main/onnx/${key}_${val}.onnx`;
                const checkString2 = `${modelId}/resolve/main/onnx/${key}.onnx`;
                return urls.some(url => url.includes(checkString1) || url.includes(checkString2));
            });
            if (!hasAllBinaries) return false;
        } else {
            const hasEnc = urls.some(url => url.includes(`${modelId}/resolve/main/onnx/encoder_model`));
            const hasDec = urls.some(url => url.includes(`${modelId}/resolve/main/onnx/decoder_model`));
            if (!hasEnc || !hasDec) return false;
        }

        return true;
    } catch (err) {
        console.error(`Error checking ASR cache status for ${modelId}:`, err);
        return false;
    }
};

/**
 * Deletes the entire transformers.js ASR model cache from the browser.
 */
export const clearAsrCache = async (): Promise<void> => {
    if (!('caches' in window)) return;

    try {
        // transformers.js v4 uses different cache names, commonly "transformers-cache"
        await caches.delete('transformers-cache');
        // Delete nemotron cache
        await caches.delete('nemotron-asr-int4-1');
        // Delete qwen3 cache
        await caches.delete('qwen3-asr-0.6b-onnx');
        console.log("ASR model cache cleared successfully.");
    } catch (err) {
        console.error("Error clearing ASR model cache:", err);
        throw err;
    }
};


/**
 * Converts, processes, and resamples an audio blob to a mono, 16kHz Float32Array
 * suitable for the Whisper model.
 * @param audioBlob The audio data captured from the microphone.
 * @param options Optional settings for audio processing like noise suppression and gain.
 * @returns A promise that resolves to the processed audio as a Float32Array.
 */
export const processAudioForTranscription = async (
    audioBlob: Blob,
    options: { noiseSuppression?: boolean; gain?: number } = {}
): Promise<Float32Array> => {
    const { noiseSuppression = false, gain = 1.0 } = options;
    const AudioContext = window.AudioContext || (window as any).webkitAudioContext;
    const OfflineAudioContext = window.OfflineAudioContext || (window as any).webkitOfflineAudioContext;

    if (!AudioContext) {
        throw new Error("Your browser does not support the Web Audio API, which is required for transcription.");
    }
    
    const audioBlobArrayBuffer = await audioBlob.arrayBuffer();
    const tempAudioContext = new AudioContext();
    const decodedAudioBuffer = await tempAudioContext.decodeAudioData(audioBlobArrayBuffer);
    
    let monoChannelData: Float32Array;
    if (decodedAudioBuffer.numberOfChannels > 1) {
        monoChannelData = new Float32Array(decodedAudioBuffer.length);
        for (let i = 0; i < decodedAudioBuffer.length; i++) {
            let mixed = 0;
            for (let channel = 0; channel < decodedAudioBuffer.numberOfChannels; channel++) {
                mixed += decodedAudioBuffer.getChannelData(channel)[i];
            }
            monoChannelData[i] = mixed / decodedAudioBuffer.numberOfChannels;
        }
    } else {
        monoChannelData = decodedAudioBuffer.getChannelData(0);
    }
    
    const targetSampleRate = 16000;
    
    const monoBuffer = tempAudioContext.createBuffer(1, monoChannelData.length, decodedAudioBuffer.sampleRate);
    monoBuffer.copyToChannel(monoChannelData, 0);

    const offlineContext = new OfflineAudioContext(
        1,
        Math.ceil(monoBuffer.duration * targetSampleRate),
        targetSampleRate
    );
    
    const bufferSource = offlineContext.createBufferSource();
    bufferSource.buffer = monoBuffer;

    let lastNode: AudioNode = bufferSource;

    if (noiseSuppression) {
        const highpassFilter = offlineContext.createBiquadFilter();
        highpassFilter.type = 'highpass';
        highpassFilter.frequency.value = 100; // Cut off frequencies below 100Hz
        lastNode.connect(highpassFilter);
        lastNode = highpassFilter;
    }

    if (gain && gain !== 1.0) {
        const gainNode = offlineContext.createGain();
        gainNode.gain.value = gain;
        lastNode.connect(gainNode);
        lastNode = gainNode;
    }

    lastNode.connect(offlineContext.destination);
    
    bufferSource.start();
    
    const resampledAudioBuffer = await offlineContext.startRendering();
    
    await tempAudioContext.close();
    
    return resampledAudioBuffer.getChannelData(0);
};