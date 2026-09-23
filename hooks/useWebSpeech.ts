import { useState, useRef, useCallback, useEffect, useMemo } from 'react';
import { createVad } from '@fluidinference/fluidvad';
import fluidvadWasmUrl from '@fluidinference/fluidvad/dist/fluidvad_bg.wasm?url';

interface UseWebSpeechOptions {
    onResult: (transcript: string, isFinal: boolean) => void;
    onError: (error: string) => void;
    onStart: () => void;
    onEnd: () => void;
}

// Safely access vendor-prefixed SpeechRecognition
const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;

// Exact Google Speech Recognition / Android SpeechRecognizer BCP 47 locale mapping
// Derived from official Google Web Speech API & Android RecognizerIntent specifications:
// Chinese (Taiwan): 'cmn-Hant-TW'
// Chinese (China Mainland): 'cmn-Hans-CN'
// Chinese (Hong Kong): 'zh-HK' or 'cmn-Hans-HK'
// English: 'en-US', etc.
export function normalizeWebSpeechLang(rawLang: string): string {
    if (!rawLang || rawLang === 'auto' || rawLang === 'Auto Detect' || rawLang === 'autodetect') {
        const browserLang = (typeof navigator !== 'undefined' && (navigator.language || (navigator as any).userLanguage)) || 'cmn-Hant-TW';
        return normalizeWebSpeechLang(browserLang);
    }

    const trimmed = rawLang.trim();

    // Map according to Google's official Web Speech API specification table
    const googleSpeechMap: Record<string, string> = {
        // Traditional Chinese (Taiwan)
        'zh-TW': 'cmn-Hant-TW',
        'zh-Hant': 'cmn-Hant-TW',
        'cmn-Hant-TW': 'cmn-Hant-TW',
        'cmn-TW': 'cmn-Hant-TW',

        // Traditional Chinese (Hong Kong)
        'zh-HK': 'yue-Hant-HK',
        'cmn-Hans-HK': 'cmn-Hans-HK',
        'yue-Hant-HK': 'yue-Hant-HK',

        // Simplified Chinese (Mainland China)
        'zh-CN': 'cmn-Hans-CN',
        'zh-Hans': 'cmn-Hans-CN',
        'cmn-Hans-CN': 'cmn-Hans-CN',
        'cmn-CN': 'cmn-Hans-CN',
        'zh-SG': 'cmn-Hans-CN',
        'zh': 'cmn-Hant-TW',

        // English
        'en': 'en-US',
        'en-US': 'en-US',
        'en-GB': 'en-GB',
        'en-AU': 'en-AU',
        'en-CA': 'en-CA',
        'en-IN': 'en-IN',

        // Japanese
        'ja': 'ja-JP',
        'ja-JP': 'ja-JP',

        // Korean
        'ko': 'ko-KR',
        'ko-KR': 'ko-KR',

        // Spanish
        'es': 'es-ES',
        'es-ES': 'es-ES',
        'es-US': 'es-US',
        'es-MX': 'es-MX',

        // French
        'fr': 'fr-FR',
        'fr-FR': 'fr-FR',

        // German
        'de': 'de-DE',
        'de-DE': 'de-DE',

        // Russian
        'ru': 'ru-RU',
        'ru-RU': 'ru-RU',

        // Italian
        'it': 'it-IT',
        'it-IT': 'it-IT',

        // Vietnamese, Thai, Indonesian, Portuguese
        'vi': 'vi-VN',
        'vi-VN': 'vi-VN',
        'th': 'th-TH',
        'th-TH': 'th-TH',
        'id': 'id-ID',
        'id-ID': 'id-ID',
        'pt': 'pt-BR',
        'pt-BR': 'pt-BR',
        'pt-PT': 'pt-PT',
        'nl': 'nl-NL',
        'nl-NL': 'nl-NL',
        'pl': 'pl-PL',
        'pl-PL': 'pl-PL',
    };

    if (googleSpeechMap[trimmed]) return googleSpeechMap[trimmed];
    if (/^zh[-_]Hant/i.test(trimmed) || /^zh[-_]TW/i.test(trimmed)) return 'cmn-Hant-TW';
    if (/^zh[-_]HK/i.test(trimmed)) return 'yue-Hant-HK';
    if (/^zh[-_]Hans/i.test(trimmed) || /^zh[-_]CN/i.test(trimmed)) return 'cmn-Hans-CN';

    const prefix = trimmed.split(/[-_]/)[0].toLowerCase();
    return googleSpeechMap[prefix] || trimmed;
}

// Fallback alternative language in case a browser or engine specifically lacks cmn-Hant-TW (e.g. older Android WebViews)
export function getFallbackWebSpeechLang(currentLang: string): string {
    const fallbacks: Record<string, string> = {
        'cmn-Hant-TW': 'zh-TW',
        'zh-TW': 'cmn-Hant-TW',
        'cmn-Hans-CN': 'zh-CN',
        'zh-CN': 'cmn-Hans-CN',
        'yue-Hant-HK': 'zh-HK',
        'zh-HK': 'cmn-Hant-TW',
    };
    return fallbacks[currentLang] || 'en-US';
}

// Detect mobile environments (Android / iOS) where mic hardware access is strictly exclusive
const isMobileDevice = typeof navigator !== 'undefined' && 
    /Android|iPhone|iPad|iPod|webOS|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent || '');

// Helper to intelligently combine speech segments (respects CJK character boundaries and prevents duplication)
export function combineTranscript(phrases: string[], interim: string): string {
    const parts = phrases.map(p => p.trim()).filter(Boolean);
    const cleanInterim = interim.trim();
    if (cleanInterim) {
        parts.push(cleanInterim);
    }
    if (parts.length === 0) return '';

    const isCjk = (char: string) => /[\p{Unified_Ideograph}\p{Script=Hiragana}\p{Script=Katakana}\p{Script=Hangul}]/u.test(char);
    let result = parts[0];

    for (let i = 1; i < parts.length; i++) {
        const prev = parts[i - 1];
        const curr = parts[i];
        if (!curr) continue;

        // Skip exact duplicate phrases
        if (curr === prev) {
            continue;
        }

        // If curr already contains prev as a prefix
        if (curr.startsWith(prev) && prev.length >= 3) {
            result = result.slice(0, result.length - prev.length) + curr;
            continue;
        }
        // If prev already ends with curr
        if (prev.endsWith(curr) && curr.length >= 3) {
            continue;
        }

        const lastChar = result.slice(-1);
        const firstChar = curr.slice(0, 1);
        if (isCjk(lastChar) && isCjk(firstChar)) {
            result += curr;
        } else {
            result += ' ' + curr;
        }
    }
    return result.trim();
}

export const useWebSpeech = ({ onResult, onError, onStart, onEnd }: UseWebSpeechOptions) => {
    const [isListening, setIsListening] = useState(false);
    
    // Stable callback refs to eliminate re-render loops and stale closures
    const onResultRef = useRef(onResult);
    const onErrorRef = useRef(onError);
    const onStartRef = useRef(onStart);
    const onEndRef = useRef(onEnd);

    useEffect(() => {
        onResultRef.current = onResult;
        onErrorRef.current = onError;
        onStartRef.current = onStart;
        onEndRef.current = onEnd;
    });

    const recognitionRef = useRef<any | null>(null);
    const shouldRestart = useRef(false);
    const restartTimeoutRef = useRef<NodeJS.Timeout | null>(null);
    const hasTriedFallbackLangRef = useRef(false);

    // Utterance tracking and single-flush guard
    const isFlushedRef = useRef(false);
    const hasSpokenRef = useRef(false);
    const currentTranscriptRef = useRef('');

    // FluidVAD resources
    const audioContextRef = useRef<AudioContext | null>(null);
    const mediaStreamRef = useRef<MediaStream | null>(null);
    const workletNodeRef = useRef<AudioWorkletNode | null>(null);
    const fallbackSilenceTimerRef = useRef<NodeJS.Timeout | null>(null);

    // Stop and teardown FluidVAD audio pipeline
    const stopFluidVad = useCallback(() => {
        if (fallbackSilenceTimerRef.current) {
            clearTimeout(fallbackSilenceTimerRef.current);
            fallbackSilenceTimerRef.current = null;
        }
        if (workletNodeRef.current) {
            try {
                workletNodeRef.current.disconnect();
            } catch {}
            workletNodeRef.current = null;
        }
        if (mediaStreamRef.current) {
            try {
                mediaStreamRef.current.getTracks().forEach(track => track.stop());
            } catch {}
            mediaStreamRef.current = null;
        }
        if (audioContextRef.current) {
            try {
                audioContextRef.current.close();
            } catch {}
            audioContextRef.current = null;
        }
    }, []);

    // Flush accumulated transcript as final and stop recognition
    const flushAsFinal = useCallback(() => {
        if (isFlushedRef.current) return;
        isFlushedRef.current = true;

        if (fallbackSilenceTimerRef.current) {
            clearTimeout(fallbackSilenceTimerRef.current);
            fallbackSilenceTimerRef.current = null;
        }
        if (restartTimeoutRef.current) {
            clearTimeout(restartTimeoutRef.current);
            restartTimeoutRef.current = null;
        }

        shouldRestart.current = false;
        stopFluidVad();

        if (recognitionRef.current) {
            try {
                recognitionRef.current.stop();
            } catch {}
        }

        setIsListening(false);

        const textToEmit = currentTranscriptRef.current.trim();
        if (textToEmit) {
            onResultRef.current(textToEmit, true);
        }
        currentTranscriptRef.current = '';
        hasSpokenRef.current = false;

        onEndRef.current();
    }, [stopFluidVad]);

    // Initialize FluidVAD with mic (only on desktop where concurrent capture is supported by OS)
    const startFluidVad = useCallback(async () => {
        // On mobile devices, microphone hardware cannot be shared concurrently with SpeechRecognition
        if (isMobileDevice) {
            return;
        }

        stopFluidVad();

        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    channelCount: 1,
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true,
                }
            });
            mediaStreamRef.current = stream;

            const AudioCtx = window.AudioContext || (window as any).webkitAudioContext;
            const audioContext = new AudioCtx({ sampleRate: 16000 });
            audioContextRef.current = audioContext;

            // Ensure worklet is registered
            if (!(globalThis as any).fluidVadWorkletUrl) {
                const workletCode = `
                class FluidVadProcessor extends AudioWorkletProcessor {
                    constructor() {
                        super();
                        this._buffer = new Float32Array(512);
                        this._fill = 0;
                    }
                    process(inputs) {
                        const channel = inputs[0] && inputs[0][0];
                        if (!channel) return true;
                        let offset = 0;
                        while (offset < channel.length) {
                            const take = Math.min(this._buffer.length - this._fill, channel.length - offset);
                            this._buffer.set(channel.subarray(offset, offset + take), this._fill);
                            this._fill += take;
                            offset += take;
                            if (this._fill === this._buffer.length) {
                                const frame = this._buffer.slice();
                                this.port.postMessage(frame.buffer, [frame.buffer]);
                                this._fill = 0;
                            }
                        }
                        return true;
                    }
                }
                registerProcessor("fluidvad-processor", FluidVadProcessor);
                `;
                const blob = new Blob([workletCode], { type: 'application/javascript' });
                (globalThis as any).fluidVadWorkletUrl = URL.createObjectURL(blob);
            }

            await audioContext.audioWorklet.addModule((globalThis as any).fluidVadWorkletUrl);

            const wasmResponse = await fetch(fluidvadWasmUrl);
            const wasmBuffer = await wasmResponse.arrayBuffer();
            // Configure VAD: 1.5s silence duration gives plenty of breathing room across pauses
            const vad = await createVad(
                { threshold: 0.5, minSilenceDuration: 1.5, maxSpeechDuration: 30, speechPadding: 0.3 },
                { wasm: wasmBuffer }
            );

            const source = audioContext.createMediaStreamSource(stream);
            const processor = new AudioWorkletNode(audioContext, "fluidvad-processor");
            workletNodeRef.current = processor;

            processor.port.onmessage = (e) => {
                const samples = new Float32Array(e.data);
                const events = vad.push(samples);

                for (const event of events) {
                    if (event.isStart) {
                        // User started speaking: cancel any pending silence fallback
                        if (fallbackSilenceTimerRef.current) {
                            clearTimeout(fallbackSilenceTimerRef.current);
                            fallbackSilenceTimerRef.current = null;
                        }
                    } else {
                        // Silence exceeded minSilenceDuration: utterance is genuinely finished!
                        flushAsFinal();
                    }
                }
            };

            source.connect(processor);
            processor.connect(audioContext.destination);
        } catch (err) {
            console.warn('[FluidVAD] Could not initialize FluidVAD on mic stream, using fallback silence detector:', err);
        }
    }, [stopFluidVad, flushAsFinal]);

    // Schedule fallback timer in case FluidVAD is unavailable or un-triggered
    const scheduleFallbackSilence = useCallback(() => {
        if (fallbackSilenceTimerRef.current) {
            clearTimeout(fallbackSilenceTimerRef.current);
        }
        // 1.5s silence window before auto-committing
        fallbackSilenceTimerRef.current = setTimeout(() => {
            flushAsFinal();
        }, 1500);
    }, [flushAsFinal]);

    // Setup SpeechRecognition once
    useEffect(() => {
        if (!SpeechRecognition) {
            console.warn("Web Speech API is not supported by this browser.");
            return;
        }

        const recognition = new SpeechRecognition();
        recognition.continuous = true; // Keep listening across multiple pauses
        recognition.interimResults = true; // Emit interim results

        recognition.onstart = () => {
            if (restartTimeoutRef.current) {
                clearTimeout(restartTimeoutRef.current);
                restartTimeoutRef.current = null;
            }
            setIsListening(true);
            onStartRef.current();
        };

        recognition.onend = () => {
            if (restartTimeoutRef.current) {
                clearTimeout(restartTimeoutRef.current);
                restartTimeoutRef.current = null;
            }

            if (isFlushedRef.current) {
                return;
            }

            // If user has spoken something and onend fired, this utterance has finished!
            if (hasSpokenRef.current && currentTranscriptRef.current.trim()) {
                flushAsFinal();
                return;
            }

            // If user hasn't spoken anything yet, allow restart to stay listening
            if (shouldRestart.current && !hasSpokenRef.current) {
                restartTimeoutRef.current = setTimeout(() => {
                    if (shouldRestart.current && !hasSpokenRef.current && recognitionRef.current) {
                        try {
                            recognitionRef.current.start();
                        } catch (e) {
                            console.warn('[WebSpeech] Recognition restart error:', e);
                            setIsListening(false);
                            flushAsFinal();
                            stopFluidVad();
                            onEndRef.current();
                        }
                    }
                }, 80);
            } else {
                setIsListening(false);
                flushAsFinal();
                stopFluidVad();
                onEndRef.current();
            }
        };

        recognition.onerror = (event: any) => {
            console.warn('[WebSpeech] Recognition error:', event.error);

            // 'no-speech' is common on pauses, ignore it so recognition can keep listening
            if (event.error === 'no-speech' || event.error === 'aborted') {
                return;
            }

            // If audio capture conflict occurs, immediately release FluidVAD's mic stream
            if (event.error === 'audio-capture') {
                stopFluidVad();
            }

            // Language not supported on this specific mobile engine: try graceful fallback
            if (event.error === 'language-not-supported' && !hasTriedFallbackLangRef.current) {
                hasTriedFallbackLangRef.current = true;
                const currentLang = recognitionRef.current?.lang || '';
                const fallback = getFallbackWebSpeechLang(currentLang);

                if (fallback && fallback !== currentLang && recognitionRef.current) {
                    console.warn(`[WebSpeech] Fallback from ${currentLang} to ${fallback}`);
                    recognitionRef.current.lang = fallback;
                    try {
                        recognitionRef.current.start();
                        return;
                    } catch {}
                }
            }

            shouldRestart.current = false;
            onErrorRef.current(event.error);
            flushAsFinal();
            stopFluidVad();
        };

        recognition.onresult = (event: any) => {
            const finalParts: string[] = [];
            let sessionInterim = '';

            // Read directly from event.results to avoid accumulating duplicates across events
            for (let i = 0; i < event.results.length; ++i) {
                const res = event.results[i];
                if (!res || !res[0]) continue;
                const transcript = (res[0].transcript || '').trim();
                if (!transcript) continue;

                if (res.isFinal) {
                    const last = finalParts[finalParts.length - 1];
                    if (last) {
                        if (last === transcript) continue;
                        if (transcript.startsWith(last)) {
                            finalParts[finalParts.length - 1] = transcript;
                            continue;
                        }
                        if (last.endsWith(transcript)) continue;
                    }
                    finalParts.push(transcript);
                } else {
                    sessionInterim += (sessionInterim ? ' ' : '') + res[0].transcript;
                }
            }

            // Strip interim text if it duplicates or contains already finalized text
            let cleanInterim = sessionInterim.trim();
            const joinedFinal = combineTranscript(finalParts, '');
            if (joinedFinal && cleanInterim) {
                if (cleanInterim === joinedFinal) {
                    cleanInterim = '';
                } else if (cleanInterim.startsWith(joinedFinal)) {
                    cleanInterim = cleanInterim.slice(joinedFinal.length).trim();
                }
            }

            const fullTranscript = combineTranscript(finalParts, cleanInterim);

            if (fullTranscript) {
                hasSpokenRef.current = true;
                currentTranscriptRef.current = fullTranscript;
                // Emit as interim (isFinal: false) so the UI stays up-to-date
                onResultRef.current(fullTranscript, false);
                scheduleFallbackSilence();
            }
        };

        recognitionRef.current = recognition;

        return () => {
            shouldRestart.current = false;
            if (restartTimeoutRef.current) {
                clearTimeout(restartTimeoutRef.current);
                restartTimeoutRef.current = null;
            }
            try {
                recognition.stop();
            } catch {}
            stopFluidVad();
        };
    }, [flushAsFinal, stopFluidVad, scheduleFallbackSilence]);

    const startRecognition = useCallback((lang: string) => {
        if (recognitionRef.current && !isListening) {
            try {
                isFlushedRef.current = false;
                hasSpokenRef.current = false;
                currentTranscriptRef.current = '';
                hasTriedFallbackLangRef.current = false;
                const normalizedLang = normalizeWebSpeechLang(lang);
                recognitionRef.current.lang = normalizedLang;
                shouldRestart.current = true;
                recognitionRef.current.start();
                startFluidVad();
            } catch (e) {
                console.error("Could not start speech recognition:", e);
                onErrorRef.current("Failed to start recognition.");
            }
        }
    }, [isListening, startFluidVad]);

    const stopRecognition = useCallback(() => {
        shouldRestart.current = false;
        if (restartTimeoutRef.current) {
            clearTimeout(restartTimeoutRef.current);
            restartTimeoutRef.current = null;
        }
        flushAsFinal();
        stopFluidVad();
        if (recognitionRef.current && isListening) {
            try {
                recognitionRef.current.stop();
            } catch {}
        }
        setIsListening(false);
    }, [isListening, flushAsFinal, stopFluidVad]);

    return useMemo(() => ({
        isListening,
        startRecognition,
        stopRecognition
    }), [isListening, startRecognition, stopRecognition]);
};
