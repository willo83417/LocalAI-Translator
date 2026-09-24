
import React, { useState, useCallback, useEffect, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import TranslationInput from './components/TranslationInput';
import TranslationOutput from './components/TranslationOutput';
import CameraView from './components/CameraView';
import SettingsModal from './components/SettingsModal';
import HistoryModal from './components/HistoryModal';
import ExpandedTextModal from './components/ExpandedTextModal';
import { translateTextStream as translateTextGeminiStream, translateImage as translateImageGemini, transcribeAudioGemini } from './services/geminiService';
import { translateTextStream as translateTextOpenAIStream, translateImage as translateImageOpenAI, transcribeAudioOpenAI } from './services/openaiService';
import { downloadManager, type DownloadProgress } from './services/downloadManager';
import { processAudioForTranscription, checkAsrModelCacheStatus, clearAsrCache } from './services/asrService';
import { useWebSpeech } from './hooks/useWebSpeech';
import { usePaddleOcr } from './hooks/usePaddleOcr';
import { deleteOcrModelCache } from './utils/db';
import { filterVoicesForLanguage, normalizeWebSpeechLang } from './utils/speechUtils';
import type { Language, TranslationHistoryItem, CustomOfflineModel, EsearchOCROutput, EsearchOCRItem, AsrEngineType, NemotronProfile, NemotronBeamWidth } from './types';
import { LANGUAGES, OFFLINE_MODELS, OFFLINE_MODELS_TS, ASR_MODELS, OCR_MODELS } from './constants';
import { GeminiLiveService } from './services/geminiLiveService';
import { createVad } from '@fluidinference/fluidvad';
// @ts-ignore
import fluidvadWasmUrl from '@fluidinference/fluidvad/dist/fluidvad_bg.wasm?url';

interface SpeechRecognition {
    continuous: boolean;
    interimResults: boolean;
    lang: string;
    onresult: (event: any) => void;
    onerror: (event: any) => void;
    onend: () => void;
    start: () => void;
    stop: () => void;
}

const SpeechRecognitionImpl = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;

// Helper function to convert AudioBuffer to a mono WAV Blob
const audioBufferToWav = (buffer: AudioBuffer): Blob => {
    // Downmix to mono if necessary
    let monoChannel: Float32Array;
    if (buffer.numberOfChannels > 1) {
        monoChannel = new Float32Array(buffer.length);
        for (let i = 0; i < buffer.length; i++) {
            let mixed = 0;
            for (let j = 0; j < buffer.numberOfChannels; j++) {
                mixed += buffer.getChannelData(j)[i];
            }
            monoChannel[i] = mixed / buffer.numberOfChannels;
        }
    } else {
        monoChannel = buffer.getChannelData(0);
    }
    
    const numOfChan = 1; // mono
    const length = monoChannel.length * numOfChan * 2 + 44;
    const bufferArray = new ArrayBuffer(length);
    const view = new DataView(bufferArray);
    let pos = 0;

    // Helper functions for writing data
    const setUint16 = (data: number) => {
        view.setUint16(pos, data, true);
        pos += 2;
    };
    const setUint32 = (data: number) => {
        view.setUint32(pos, data, true);
        pos += 4;
    };
    const setString = (str: string) => {
        for(let i = 0; i < str.length; i++) {
            view.setUint8(pos + i, str.charCodeAt(i));
        }
        pos += str.length;
    }
    
    // RIFF header
    setString('RIFF');
    setUint32(length - 8);
    setString('WAVE');

    // fmt chunk
    setString('fmt ');
    setUint32(16); // chunk size
    setUint16(1); // audio format (1 = PCM)
    setUint16(numOfChan);
    setUint32(buffer.sampleRate);
    setUint32(buffer.sampleRate * 2 * numOfChan); // byte rate
    setUint16(numOfChan * 2); // block align
    setUint16(16); // bits per sample

    // data chunk
    setString('data');
    setUint32(monoChannel.length * 2);

    // Write PCM data
    for (let i = 0; i < monoChannel.length; i++) {
        let sample = Math.max(-1, Math.min(1, monoChannel[i])); // clamp
        view.setInt16(pos, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true);
        pos += 2;
    }

    return new Blob([view], { type: 'audio/wav' });
};

const processAudioToWav = async (audioBlob: Blob, noiseSuppression: boolean, gain: number): Promise<Blob> => {
    const audioData = await processAudioForTranscription(audioBlob, { noiseSuppression, gain });
    const tempAudioContext = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 16000 });
    const audioBuffer = tempAudioContext.createBuffer(1, audioData.length, 16000);
    audioBuffer.copyToChannel(audioData, 0);
    const pcmWavBlob = audioBufferToWav(audioBuffer);
    await tempAudioContext.close();
    return pcmWavBlob;
};

// --- OCR Processing Logic ---
const processOcrResult = (result: EsearchOCROutput): string => {
    // 1. Prefer esearch-ocr's native layout analysis columns
    if (result.columns && result.columns.length > 0) {
        const columnTexts = result.columns
            .map(col => col.parragraphs.map(p => p.parse.text).filter(Boolean).join('\n'))
            .filter(t => t.trim().length > 0);
        if (columnTexts.length > 0) {
            return columnTexts.join('\n\n');
        }
    }

    // 2. Next fallback: native reading-order paragraphs
    if (result.parragraphs && result.parragraphs.length > 0) {
        const text = result.parragraphs.map(p => p.text).filter(Boolean).join('\n');
        if (text.trim().length > 0) return text;
    }

    // 3. Fallback: raw detected items
    if (result.src && result.src.length > 0) {
        return result.src.map(i => i.text).filter(Boolean).join('\n');
    }

    return "";
};
// --- END OCR ---


interface AppMessage {
    type: 'log' | 'transcription' | 'transcription-partial' | 'loaded' | 'error' | 'progress' | 'unloaded';
    payload: any;
}

const App: React.FC = () => {
    const { t, i18n } = useTranslation();
    const [inputText, setInputText] = useState('');
    const [translatedText, setTranslatedText] = useState('');
    const [sourceLang, setSourceLang] = useState<Language>(() => {
        const saved = localStorage.getItem('source-lang');
        if (saved) {
            const lang = LANGUAGES.find(l => l.code === saved);
            if (lang) return lang;
        }
        return LANGUAGES[0];
    });
    const [targetLang, setTargetLang] = useState<Language>(() => {
        const saved = localStorage.getItem('target-lang');
        if (saved) {
            const lang = LANGUAGES.find(l => l.code === saved);
            if (lang) return lang;
        }
        return LANGUAGES[6];
    });

    const sourceLangRef = useRef<Language>(sourceLang);
    const targetLangRef = useRef<Language>(targetLang);

    useEffect(() => {
        sourceLangRef.current = sourceLang;
    }, [sourceLang]);

    useEffect(() => {
        targetLangRef.current = targetLang;
    }, [targetLang]);

    const [isLoading, setIsLoading] = useState(false);
    const [isTranscribing, setIsTranscribing] = useState(false);
    
    const [isRecording, setIsRecording] = useState(false);
    const [isAstRecording, setIsAstRecording] = useState(false);
    const [isCameraOpen, setIsCameraOpen] = useState(false);
    
    const [isSettingsOpen, setIsSettingsOpen] = useState(false);
    const [isHistoryOpen, setIsHistoryOpen] = useState(false);
    const [isExpandedTextOpen, setIsExpandedTextOpen] = useState(false);
    const [history, setHistory] = useState<TranslationHistoryItem[]>([]);
    
    // Shared settings
    const [apiKey, setApiKey] = useState('');
    const [modelName, setModelName] = useState('gemini-3.5-flash-lite');

    // Online provider settings
    const [onlineProvider, setOnlineProvider] = useState('gemini');
    const [openaiApiUrl, setOpenaiApiUrl] = useState('');
    
    // Offline settings
    const [huggingFaceApiKey, setHuggingFaceApiKey] = useState('');
    const [offlineModelName, setOfflineModelName] = useState('');
    const [isTwoStepJpCn, setIsTwoStepJpCn] = useState(false);
    const [customModels, setCustomModels] = useState<CustomOfflineModel[]>([]);
    
    const [voices, setVoices] = useState<SpeechSynthesisVoice[]>([]);
    const [isOnline, setIsOnline] = useState(navigator.onLine);
    const [isSpeaking, setIsSpeaking] = useState(false);
    const [speakingGender, setSpeakingGender] = useState<'female' | 'male' | null>(null);

    // Offline TTS settings
    const [isOfflineTtsEnabled, setIsOfflineTtsEnabled] = useState(false);
    const [offlineTtsVoiceURI, setOfflineTtsVoiceURI] = useState('');
    const [offlineTtsRate, setOfflineTtsRate] = useState(1);
    const [offlineTtsPitch, setOfflineTtsPitch] = useState(1);

    const [isOfflineModeEnabled, setIsOfflineModeEnabled] = useState(false);
    const [downloadProgress, setDownloadProgress] = useState<Record<string, DownloadProgress>>({});
    const [isOfflineModelInitializing, setIsOfflineModelInitializing] = useState(false);
    const [isOfflineModelInitialized, setIsOfflineModelInitialized] = useState(false);

    // Offline Model Parameters
    const [offlineMaxTokens, setOfflineMaxTokens] = useState(2048);
    const [offlineTopK, setOfflineTopK] = useState(40);
    const [offlineTemperature, setOfflineTemperature] = useState(0.3);
    const [offlineRandomSeed, setOfflineRandomSeed] = useState(1);
    const [offlineSupportAudio, setOfflineSupportAudio] = useState(false);
    const [offlineAudioRealtime, setOfflineAudioRealtime] = useState(false);
    const [offlineMaxNumImages, setOfflineMaxNumImages] = useState(0);

    // ASR State
    const [isWebSpeechApiEnabled, setIsWebSpeechApiEnabled] = useState<boolean>(() => {
        try {
            const saved = localStorage.getItem('is-web-speech-api-enabled');
            return saved !== null ? JSON.parse(saved) : true;
        } catch {
            return true;
        }
    });
    const [isOfflineAsrEnabled, setIsOfflineAsrEnabled] = useState<boolean>(() => {
        try {
            const saved = localStorage.getItem('is-offline-asr-enabled');
            return saved !== null ? JSON.parse(saved) : false;
        } catch {
            return false;
        }
    });
    const [isRealtimeAsrEnabled, setIsRealtimeAsrEnabled] = useState<boolean>(() => {
        try {
            const saved = localStorage.getItem('is-realtime-asr-enabled');
            return saved !== null ? JSON.parse(saved) : false;
        } catch {
            return false;
        }
    });
    const [asrModelId, setAsrModelId] = useState<string>(() => {
        const savedEngine = localStorage.getItem('asr-engine');
        if (savedEngine === 'nemotron') return 'nemotron';
        if (savedEngine === 'qwen3') return 'qwen3';
        return localStorage.getItem('asr-model-id') || ASR_MODELS[0].id;
    });
    const [asrEngine, setAsrEngine] = useState<AsrEngineType>(() => {
        const saved = localStorage.getItem('asr-engine');
        if (saved === 'nemotron' || saved === 'qwen3' || saved === 'whisper') {
            return saved as AsrEngineType;
        }
        return 'whisper';
    });
    const [asrProfile, setAsrProfile] = useState<NemotronProfile>(() => {
        const saved = localStorage.getItem('asr-profile');
        if (saved === 'FAST' || saved === 'NORMAL' || saved === 'BEST') {
            return saved as NemotronProfile;
        }
        return 'NORMAL';
    });
    const [asrBeamWidth, setAsrBeamWidth] = useState<NemotronBeamWidth>(() => {
        try {
            const saved = localStorage.getItem('asr-beam-width');
            return saved !== null ? (JSON.parse(saved) as NemotronBeamWidth) : 1;
        } catch {
            return 1;
        }
    });
    const [isAsrInitializing, setIsAsrInitializing] = useState(false);
    const [isAsrInitialized, setIsAsrInitialized] = useState(false);
    const [asrModelsCacheStatus, setAsrModelsCacheStatus] = useState<Record<string, boolean>>({});
    const [asrLoadingProgress, setAsrLoadingProgress] = useState({ file: '', progress: 0 });
    const [isNoiseCancellationEnabled, setIsNoiseCancellationEnabled] = useState<boolean>(() => {
        try {
            const saved = localStorage.getItem('is-noise-cancellation-enabled');
            return saved !== null ? JSON.parse(saved) : true;
        } catch {
            return true;
        }
    });
    const [audioGainValue, setAudioGainValue] = useState<number>(() => {
        try {
            const saved = localStorage.getItem('audio-gain-value');
            return saved !== null ? JSON.parse(saved) : 1.0;
        } catch {
            return 1.0;
        }
    });

    // Offline recording countdown state
    const [recordingCountdown, setRecordingCountdown] = useState<number | null>(null);
    const [isAwaitingAstTranslation, setIsAwaitingAstTranslation] = useState(false);

    // OCR State
    const { status: ocrEngineStatus, error: ocrEngineError, recognize, initializeOcr, unloadOcr } = usePaddleOcr();
    const [selectedOcrModel, setSelectedOcrModel] = useState<keyof typeof OCR_MODELS>('PP_v6_small');
    const [isOcrAutoInitEnabled, setIsOcrAutoInitEnabled] = useState(false);

    // Sequential Loading Queue
    const [loadingQueue, setLoadingQueue] = useState<string[]>([]);


    type NotificationType = 'error' | 'success' | 'info';
    interface Notification {
        message: string;
        type: NotificationType;
    }
    const [notification, setNotification] = useState<Notification | null>(null);
    const notificationTimerRef = useRef<number | null>(null);

    // MediaPipe Worker (Offline LLM)
    const workerRef = useRef<Worker | null>(null);
    const messageHandlerRef = useRef((event: MessageEvent) => {});

    // Whisper Worker (Offline ASR)
    const asrWorkerRef = useRef<Worker | null>(null);

    const showNotification = useCallback((message: string, type: NotificationType = 'error') => {
        if (notificationTimerRef.current) {
            clearTimeout(notificationTimerRef.current);
        }
        setNotification({ message, type });
        notificationTimerRef.current = window.setTimeout(() => {
            setNotification(null);
        }, 5000);
    }, []);

    const recognitionRef = useRef<SpeechRecognition | null>(null);
    const astRecognitionRef = useRef<SpeechRecognition | null>(null);
    const transcribedTextRef = useRef('');
    const mediaStreamRef = useRef<MediaStream | null>(null);
    const mediaRecorderRef = useRef<MediaRecorder | null>(null);
    const audioChunksRef = useRef<Blob[]>([]);
    const lastProcessedSampleIndexRef = useRef<number>(0);
    
    // Countdown timer
    const countdownTimerRef = useRef<number | null>(null);
    // Translation abort controller
    const translationAbortControllerRef = useRef<AbortController | null>(null);
    // Audio abort controller
    const audioAbortControllerRef = useRef<AbortController | null>(null);
    const onStopRecordingCallbackRef = useRef<((blob: Blob) => void) | null>(null);
    // Flag to handle reverse translation logic after ASR
    const isReverseTranslateRef = useRef(false);
    const isAsrProcessingRef = useRef(false);

    // Gemini 3.5 Live Service State & Ref
    const liveServiceRef = useRef<GeminiLiveService | null>(null);
    const [liveStatus, setLiveStatus] = useState<'disconnected' | 'connecting' | 'connected' | 'listening'>('disconnected');

    // Refs for streaming text accumulation in Gemini 3.5 Live Translate
    const accumulatedInputRef = useRef('');
    const accumulatedTranslatedRef = useRef('');
    const currentActiveInputRef = useRef('');
    const currentActiveTranslatedRef = useRef('');

    useEffect(() => {
        liveServiceRef.current = new GeminiLiveService({
            onOpen: () => {
                console.log("Live Translate Session Opened");
            },
            onClose: (reason) => {
                console.log("Live Translate Session Closed:", reason);
				// Commit any remaining active transcripts before ending
                if (currentActiveInputRef.current) {
                    accumulatedInputRef.current = accumulatedInputRef.current + currentActiveInputRef.current;
                    currentActiveInputRef.current = '';
                }
                if (currentActiveTranslatedRef.current) {
                    accumulatedTranslatedRef.current = accumulatedTranslatedRef.current  + currentActiveTranslatedRef.current;
                    currentActiveTranslatedRef.current = '';
                }

                // Save to translation history if we have any dynamic final output
                const finalIn = accumulatedInputRef.current;
                const finalOut = accumulatedTranslatedRef.current;
                if (finalIn.trim() || finalOut.trim()) {
                    const newHistoryItem: TranslationHistoryItem = {
                        id: Date.now(),
                        inputText: finalIn,
                        translatedText: finalOut,
                        sourceLang: isReverseTranslateRef.current ? targetLangRef.current : sourceLangRef.current,
                        targetLang: isReverseTranslateRef.current ? sourceLangRef.current : targetLangRef.current,
                    };
                    setHistory(prevHistory => {
                        const isDuplicate = prevHistory.length > 0 && 
                            prevHistory[0].inputText === finalIn && 
                            prevHistory[0].translatedText === finalOut;
                        if (isDuplicate) return prevHistory;

                        const updatedHistory = [newHistoryItem, ...prevHistory].slice(0, 50);
                        localStorage.setItem('translation-history', JSON.stringify(updatedHistory));
                        return updatedHistory;
                    });
                }

                setIsRecording(false);
                setIsAstRecording(false);
                isReverseTranslateRef.current = false;
            },
            onError: (err) => {
                showNotification(err, 'error');
                setIsRecording(false);
                setIsAstRecording(false);
                isReverseTranslateRef.current = false;
            },
            onText: (text) => {
                // If modelTurn text chunk is received, append it to current active translation,
                // but only if we haven't already got a longer outputTranscription (to avoid conflicts)
                if (text && text.length > 0) {
                    if (currentActiveTranslatedRef.current.length < text.length) {
                        currentActiveTranslatedRef.current += text;
                    }
                    const displayInput = accumulatedInputRef.current + currentActiveInputRef.current;
                    const displayTranslated = accumulatedTranslatedRef.current + currentActiveTranslatedRef.current;
                    setInputText(displayInput);
                    setTranslatedText(displayTranslated);
                }
            },
			onInputTranscription: (text) => {
                const lastText = currentActiveInputRef.current;
                if (lastText && text !== lastText) {
                    const normalizedLast = lastText.trim().toLowerCase();
                    const normalizedNew = text.trim().toLowerCase();
                    
                    // Detect if a brand new segment has started
                    const isContinuation = normalizedNew.startsWith(normalizedLast) || 
                                          (normalizedLast.length > 3 && normalizedNew.startsWith(normalizedLast.slice(0, Math.min(normalizedLast.length, 8))));
                    
                    if (!isContinuation) {
                        // Commit the completed segment!
                        accumulatedInputRef.current = accumulatedInputRef.current + lastText;
                        accumulatedTranslatedRef.current = accumulatedTranslatedRef.current + currentActiveTranslatedRef.current;
                        currentActiveTranslatedRef.current = '';
                    }
                }
                currentActiveInputRef.current = text;

                const displayInput = accumulatedInputRef.current + text;
                const displayTranslated = accumulatedTranslatedRef.current + currentActiveTranslatedRef.current;
                
                setInputText(displayInput);
                setTranslatedText(displayTranslated);
            },
            onOutputTranscription: (text) => {
                // Since onOutputTranscription is absolute / growing for the current turn, update it directly
                currentActiveTranslatedRef.current = text;

                const displayInput = accumulatedInputRef.current + currentActiveInputRef.current;
                const displayTranslated = accumulatedTranslatedRef.current + text;

                setInputText(displayInput);
                setTranslatedText(displayTranslated);
            },
            onStatusChange: (status) => {
                setLiveStatus(status);
            },
            onSpeakingChange: (isLiveSpeaking) => {
                setIsSpeaking(isLiveSpeaking);
                setSpeakingGender(isLiveSpeaking ? 'female' : null);
            }
        });

        return () => {
            if (liveServiceRef.current) {
                liveServiceRef.current.stop();
            }
        };
    }, [showNotification]);

    // --- MediaPipe Worker Logic (Offline LLM) ---
    const getOrCreateWorker = useCallback(() => {
        if (workerRef.current) {
            return workerRef.current;
        }
        const worker = new Worker(new URL('./workers/offline.worker.ts', import.meta.url), { type: 'module' });
        workerRef.current = worker;
    
        const onMessage = (event: MessageEvent) => messageHandlerRef.current(event);
        worker.addEventListener('message', onMessage);
    
        worker.onerror = (error) => {
            console.error('Worker error:', error);
            showNotification(`A critical worker error occurred: ${error.message}`, 'error');
            setIsOfflineModelInitializing(false);
            setIsLoading(false);
        };
        return worker;
    }, [showNotification]);

    useEffect(() => {
        return () => {
            if (workerRef.current) {
                workerRef.current.terminate();
                workerRef.current = null;
            }
        };
    }, []);

    const isTSModelSelected = OFFLINE_MODELS_TS.some(m => m.value === offlineModelName);
    const isModelDownloaded = isOfflineModeEnabled && !!offlineModelName && (isTSModelSelected || downloadProgress[offlineModelName]?.status === 'completed');
    const isOfflineModelReady = isModelDownloaded && isOfflineModelInitialized;
    const isGeminiLiveModel = onlineProvider === 'gemini' && !isOfflineModeEnabled;
    const textModelName = modelName === 'gemini-3.5-live-translate-preview' ? 'gemini-3.5-flash-lite' : modelName;

    const performTranslate = useCallback(async (textToTranslate: string) => {
        if (!textToTranslate.trim()) {
            setTranslatedText('');
            return;
        }
    
        if (translationAbortControllerRef.current) {
            translationAbortControllerRef.current.abort();
        }
        const controller = new AbortController();
        translationAbortControllerRef.current = controller;
    
        setIsLoading(true);
        setTranslatedText('');
    
        try {
            const sourceLangEn = sourceLang.code === 'auto' ? 'Auto Detect' : sourceLang.code;
            const targetLangEn = targetLang.code;

            if (isOfflineModeEnabled) {
                if (!offlineModelName) throw new Error('Please select an offline model in settings.');
                if (isOfflineModelInitializing) throw new Error('Offline model is still initializing.');
                if (!isOfflineModelReady) throw new Error('Selected offline model is not ready.');
                
                const worker = getOrCreateWorker();
                worker.postMessage({
                    type: 'translate',
                    payload: {
                        text: textToTranslate,
                        sourceLang: sourceLangEn,
                        targetLang: targetLangEn,
                        sourceLangCode: sourceLang.code,
                        targetLangCode: targetLang.code,
                        isTwoStepEnabled: isTwoStepJpCn,
                    }
                });
            } else { // Online mode
                if (!isOnline) throw new Error("You are offline. Enable offline mode or connect to the internet.");
    
                let finalResult = '';
                const onChunk = (chunk: string) => {
                    finalResult += chunk;
                    setTranslatedText(prev => prev.length === 0 ? chunk.trimStart() : prev + chunk);
                };
                if (onlineProvider === 'openai') {
                    if (!apiKey) throw new Error("OpenAI API Key is not set. Please add it in the settings.");
                    if (!openaiApiUrl) throw new Error("OpenAI API URL is not set. Please add it in the settings.");
                    finalResult = await translateTextOpenAIStream(textToTranslate, sourceLangEn, targetLangEn, apiKey, textModelName, openaiApiUrl, onChunk, controller.signal);
                } else { // Gemini is default
                    if (!apiKey) throw new Error("Gemini API Key is not set. Please add it in the settings.");
                    finalResult = await translateTextGeminiStream(textToTranslate, sourceLangEn, targetLangEn, apiKey, textModelName, onChunk, controller.signal);
                }
    
                const newHistoryItem: TranslationHistoryItem = {
                    id: Date.now(), inputText: textToTranslate, translatedText: finalResult, sourceLang, targetLang,
                };
                setHistory(prevHistory => {
                    const updatedHistory = [newHistoryItem, ...prevHistory].slice(0, 50);
                    localStorage.setItem('translation-history', JSON.stringify(updatedHistory));
                    return updatedHistory;
                });
                setIsLoading(false);
            }
    
        } catch (err) {
            if (err instanceof DOMException && err.name === 'AbortError') {
                setIsLoading(false);
                return;
            }
            const errorMessage = err instanceof Error ? err.message : 'An unknown error occurred.';
            showNotification(t('notifications.translationFailed', { errorMessage }), 'error');
            if (err instanceof Error && (err.message.includes('select an offline model') || err.message.includes('API Key is not set') || err.message.includes('API URL is not set'))) {
                setIsSettingsOpen(true);
            }
            console.error(err);
            setIsLoading(false);
        } finally {
            if (!isOfflineModeEnabled && translationAbortControllerRef.current === controller) {
                translationAbortControllerRef.current = null;
            }
        }
    }, [sourceLang, targetLang, apiKey, modelName, textModelName, isOnline, isOfflineModeEnabled, offlineModelName, isOfflineModelReady, isOfflineModelInitializing, showNotification, onlineProvider, openaiApiUrl, isTwoStepJpCn, t, i18n, getOrCreateWorker]);

    const performReverseTranslate = useCallback(async (textToTranslate: string) => {
        if (!textToTranslate.trim()) {
            setInputText('');
            return;
        }

        if (translationAbortControllerRef.current) {
            translationAbortControllerRef.current.abort();
        }
        const controller = new AbortController();
        translationAbortControllerRef.current = controller;

        setIsLoading(true);
        setTranslatedText('');

        try {
            // SWAPPED LANGUAGES for reverse translation
            const sourceLangEn = targetLang.code;
            const targetLangEn = sourceLang.code;

            if (isOfflineModeEnabled) {
                if (!offlineModelName) throw new Error('Please select an offline model in settings.');
                if (isOfflineModelInitializing) throw new Error('Offline model is still initializing.');
                if (!isOfflineModelReady) throw new Error('Selected offline model is not ready.');
                
                const worker = getOrCreateWorker();
                worker.postMessage({
                    type: 'translate',
                    payload: {
                        text: textToTranslate,
                        sourceLang: sourceLangEn,
                        targetLang: targetLangEn,
                        sourceLangCode: targetLang.code, // SWAPPED
                        targetLangCode: sourceLang.code, // SWAPPED
                        isTwoStepEnabled: isTwoStepJpCn,
                    }
                });
            } else { // Online mode
                if (!isOnline) throw new Error("You are offline. Enable offline mode or connect to the internet.");

                let finalResult = '';
                const onChunk = (chunk: string) => {
                    finalResult += chunk;
                    setTranslatedText(prev => prev.length === 0 ? chunk.trimStart() : prev + chunk);
                };
                if (onlineProvider === 'openai') {
                    if (!apiKey) throw new Error("OpenAI API Key is not set.");
                    if (!openaiApiUrl) throw new Error("OpenAI API URL is not set.");
                    finalResult = await translateTextOpenAIStream(textToTranslate, sourceLangEn, targetLangEn, apiKey, textModelName, openaiApiUrl, onChunk, controller.signal);
                } else {
                    if (!apiKey) throw new Error("Gemini API Key is not set.");
                    finalResult = await translateTextGeminiStream(textToTranslate, sourceLangEn, targetLangEn, apiKey, textModelName, onChunk, controller.signal);
                }

                const newHistoryItem: TranslationHistoryItem = {
                    id: Date.now(), inputText: textToTranslate, translatedText: finalResult, sourceLang: targetLang, targetLang: sourceLang, // SWAPPED
                };
                setHistory(prevHistory => {
                    const updatedHistory = [newHistoryItem, ...prevHistory].slice(0, 50);
                    localStorage.setItem('translation-history', JSON.stringify(updatedHistory));
                    return updatedHistory;
                });
                setIsLoading(false);
            }
        } catch (err) {
            if (err instanceof DOMException && err.name === 'AbortError') {
                setIsLoading(false);
                return;
            }
            const errorMessage = err instanceof Error ? err.message : 'An unknown error occurred.';
            showNotification(t('notifications.translationFailed', { errorMessage }), 'error');
            console.error(err);
            setIsLoading(false);
        } finally {
            if (!isOfflineModeEnabled && translationAbortControllerRef.current === controller) {
                translationAbortControllerRef.current = null;
            }
        }
    }, [targetLang, sourceLang, apiKey, modelName, textModelName, isOnline, isOfflineModeEnabled, offlineModelName, isOfflineModelReady, isOfflineModelInitializing, showNotification, onlineProvider, openaiApiUrl, isTwoStepJpCn, t, i18n, getOrCreateWorker]);

    useEffect(() => {
        messageHandlerRef.current = (event: MessageEvent) => {
            const { type, payload } = event.data;
            switch(type) {
                case 'init_done':
                    setIsOfflineModelInitialized(true);
                    if (OFFLINE_MODELS_TS.some(m => m.value === payload.modelIdentifier)) {
                        setDownloadProgress(prev => ({
                            ...prev,
                            [payload.modelIdentifier]: {
                                status: 'completed',
                                downloaded: 1,
                                total: 1,
                                percent: 100
                            }
                        }));
                    }
                    showNotification(t('notifications.offlineModelInitSuccess', { modelIdentifier: payload.modelIdentifier }), 'success');
                    setIsOfflineModelInitializing(false);
                    break;
                case 'download_progress':
                    // Transformers.js download progress
                    setDownloadProgress(prev => ({
                        ...prev,
                        [payload.modelSource]: {
                            status: payload.status === 'done' ? 'completed' : 'downloading',
                            downloaded: payload.loaded || 0,
                            total: payload.total || 0,
                            percent: payload.total ? Math.round((payload.loaded / payload.total) * 100) : 0
                        }
                    }));
                    break;
                case 'init_error':
                    setIsOfflineModelInitialized(false);
                    if (OFFLINE_MODELS_TS.some(m => m.value === payload.modelSource || m.value === offlineModelName)) {
                         setDownloadProgress(prev => ({
                             ...prev,
                             [payload.modelSource || offlineModelName]: {
                                 status: 'error',
                                 downloaded: 0,
                                 total: 0,
                                 percent: 0
                             }
                         }));
                    }
                    showNotification(payload.error || t('notifications.offlineModelInitFailed'), 'error');
                    console.error('Offline model init failed via worker:', payload.error);
                    setIsOfflineModelInitializing(false);
                    break;
                case 'unload_done':
                    setIsOfflineModelInitialized(false);
                    setIsOfflineModelInitializing(false);
                    break;
                case 'translation_chunk':
                    setTranslatedText(prev => prev.length === 0 ? payload.chunk.trimStart() : prev + payload.chunk);
                    break;
                case 'translation_done': {
                    const currentSourceLang = isReverseTranslateRef.current ? targetLang : sourceLang;
                    const currentTargetLang = isReverseTranslateRef.current ? sourceLang : targetLang;
                    const newHistoryItem: TranslationHistoryItem = {
                        id: Date.now(), inputText, translatedText: payload.result, sourceLang: currentSourceLang, targetLang: currentTargetLang,
                    };
                    setHistory(prevHistory => {
                        const updatedHistory = [newHistoryItem, ...prevHistory].slice(0, 50);
                        localStorage.setItem('translation-history', JSON.stringify(updatedHistory));
                        return updatedHistory;
                    });
                    setIsLoading(false);
                    break;
                }
                case 'translation_error':
                    showNotification(t('notifications.translationFailed', { errorMessage: payload.error }), 'error');
                    setIsLoading(false);
                    break;
                case 'translation_cancelled':
                    setIsLoading(false);
                    break;
                case 'extract_text_start':
                    // Do nothing, UI is already showing processing state
                    break;
                case 'extract_text_chunk':
                    setInputText(prev => {
                        if (prev === t('notifications.processingImage') || prev === '') {
                            return payload.chunk.trimStart();
                        }
                        return prev + payload.chunk;
                    });
                    break;
                case 'extract_text_done':
                    setInputText(payload.text);
                    setTranslatedText('');
                    if (!payload.text.trim()) {
                        showNotification(t('notifications.noTextInImage'), "error");
                    }
                    setIsLoading(false);
                    break;
                case 'extract_text_error':
                    showNotification(t('notifications.imageProcessingFailed', { errorMessage }), 'error');
					setInputText(''); 
					setIsLoading(false);
                    break;
                case 'extract_text_cancelled':
                    setInputText('');
                    showNotification(t('notifications.imageProcessingCancelled') || 'Image processing cancelled', 'info');
                    setIsLoading(false);
                    break;
                case 'transcribe_start':
                    // Do not clear text here, as we are appending chunks
                    break;
                case 'transcribe_chunk':
                    setInputText(prev => {
                        if (prev === t('notifications.transcribing')) {
                            return payload.chunk;
                        }
                        return prev + payload.chunk;
                    });
                    break;
                case 'transcribe_done':
                    isAsrProcessingRef.current = false;
                    const transcribedText = payload.text?.trim() || '';
                    setIsLoading(false);
                    
                    if (payload.isChunk) {
                        setInputText(prev => prev + (prev.endsWith(' ') ? '' : ' '));
                    } else {
                        setIsTranscribing(false);
                        if (transcribedText) {
                            setInputText(transcribedText);
                            if (isAwaitingAstTranslation) {
                                setIsAwaitingAstTranslation(false);
                                performReverseTranslate(transcribedText);
                            }
                        } else {
                            setInputText(t('notifications.transcriptionFailedEmpty'));
                            setIsAwaitingAstTranslation(false);
                        }
                    }
                    break;
                case 'transcribe_error':
                    isAsrProcessingRef.current = false;
                    setIsTranscribing(false);
                    const errorMessage = payload.error || 'Unknown transcription error.';
                    showNotification(t('notifications.transcriptionFailed', { errorMessage }), 'error');
                    setInputText(t('notifications.transcriptionFailed', { errorMessage }));
                    setIsLoading(false);
                    setIsAwaitingAstTranslation(false);
                    break;
                case 'transcribe_cancelled':
                    isAsrProcessingRef.current = false;
                    setIsTranscribing(false);
                    setIsLoading(false);
                    setIsAwaitingAstTranslation(false);
                    break;
            }
        };
    }, [t, showNotification, inputText, sourceLang, targetLang, isAwaitingAstTranslation, performReverseTranslate, isReverseTranslateRef]);
    
    // --- Whisper Worker Logic (Offline ASR) ---
    const performReverseTranslateRef = useRef(performReverseTranslate);
    useEffect(() => {
        performReverseTranslateRef.current = performReverseTranslate;
    }, [performReverseTranslate]);

    const performTranslateRef = useRef(performTranslate);
    useEffect(() => {
        performTranslateRef.current = performTranslate;
    }, [performTranslate]);

    const isRealtimeAsrEnabledRef = useRef(isRealtimeAsrEnabled);
    useEffect(() => {
        isRealtimeAsrEnabledRef.current = isRealtimeAsrEnabled;
    }, [isRealtimeAsrEnabled]);

    const checkAllAsrCacheStatus = useCallback(async () => {
        const statuses: Record<string, boolean> = {};
        for (const model of ASR_MODELS) {
            statuses[model.id] = await checkAsrModelCacheStatus(model.id, model.quantization);
        }
        statuses['nemotron'] = await checkAsrModelCacheStatus('nemotron');
        statuses['qwen3'] = await checkAsrModelCacheStatus('qwen3');
        return statuses;
    }, []);

    const onAsrWorkerMessage = useCallback((e: MessageEvent<AppMessage>) => {
        const { type, payload } = e.data;
        switch (type) {
            case 'progress':
                if (payload.status === 'progress_total') {
                    setAsrLoadingProgress({ file: payload.name || payload.file || 'Loading model...', progress: payload.progress || 0 });
                } else if (payload.status === 'progress' || payload.status === 'download' || payload.status === 'init') {
                    setAsrLoadingProgress({ file: payload.name || payload.file || 'model_file', progress: payload.progress || 0 });
                } else if (payload.status === 'ready' || payload.status === 'done') {
                    setAsrLoadingProgress({ file: payload.name || payload.file || 'model_file', progress: 100 });
                }
                break;
            case 'error':
                isAsrProcessingRef.current = false;
                showNotification(payload, 'error');
                setInputText(t('notifications.transcriptionFailed', { errorMessage: payload }));
                setIsAsrInitializing(false);
                setIsTranscribing(false);
                break;
            case 'loaded':
                setIsAsrInitialized(true);
                setIsAsrInitializing(false);
                checkAllAsrCacheStatus().then(setAsrModelsCacheStatus);
                break;
            case 'unloaded':
                setIsAsrInitialized(false);
                showNotification(t('notifications.asrModelUnloaded'), 'info');
                break;
            case 'transcription-partial':
                setInputText(payload);
                break;
            case 'transcription':
                isAsrProcessingRef.current = false;
                const { text, isFinal } = payload;
                const transcribedText = (text || '').trim();
                
                // Filter out common Whisper hallucinations and meaningless sounds
                const isMeaningless = !transcribedText || 
                    /^[^a-zA-Z0-9\p{L}]+$/u.test(transcribedText) || // Only punctuation/symbols
                    /^[\[\(].*[\]\)]$/.test(transcribedText) ||
                    ['恩', '嗯', '啊', '哦', '喔', '呃', 'um', 'uh', 'ah', 'oh'].includes(transcribedText.toLowerCase());
                
                if (isFinal) {
                    setIsTranscribing(false);
                    if (transcribedText && !isMeaningless) {
                        setInputText(transcribedText);
                        if (isReverseTranslateRef.current) {
                            performReverseTranslateRef.current(transcribedText);
                            isReverseTranslateRef.current = false;
                        } else if (isRealtimeAsrEnabledRef.current) {
                            performTranslateRef.current(transcribedText);
                        }
                    } else if (isMeaningless && transcribedText) {
                        // If it's meaningless but not empty, just show it but don't translate
                        setInputText(transcribedText);
                        isReverseTranslateRef.current = false;
                    } else {
                        setInputText(t('notifications.transcriptionFailedEmpty'));
                        isReverseTranslateRef.current = false;
                    }
                } else {
                    if (transcribedText) {
                        setInputText(transcribedText);
                    }
                }
                break;
            case 'log':
                //console.log('[ASR Worker]:', payload);	//Close ASR log
                break;
            default:
                break;
        }
    }, [showNotification, checkAllAsrCacheStatus, t]);

    const onAsrWorkerMessageRef = useRef(onAsrWorkerMessage);
    useEffect(() => {
        onAsrWorkerMessageRef.current = onAsrWorkerMessage;
    }, [onAsrWorkerMessage]);

    const initializeAsrWorker = useCallback(() => {
        if (asrWorkerRef.current) {
            asrWorkerRef.current.terminate();
        }
        
        let newWorker: Worker;
        if (asrEngine === 'nemotron') {
            newWorker = new Worker(new URL('./workers/nemotron.worker.ts', import.meta.url), {
                type: 'module',
            });
        } else if (asrEngine === 'qwen3') {
            newWorker = new Worker(new URL('./workers/qwen3ASR.worker.ts', import.meta.url), {
                type: 'module',
            });
        } else {
            newWorker = new Worker(new URL('./workers/transformersASR.worker.ts', import.meta.url), {
                type: 'module',
            });
        }
        newWorker.addEventListener('message', (e) => {
            onAsrWorkerMessageRef.current(e);
        });
        asrWorkerRef.current = newWorker;
    }, [asrEngine]);

    // Effect to manage ASR Worker creation and destruction, ensuring listeners are always up-to-date
    useEffect(() => {
        if (isOfflineAsrEnabled) {
            initializeAsrWorker();
        }
        
        // Cleanup function for when the component unmounts or offline ASR is disabled or engine changes
        return () => {
            if (asrWorkerRef.current) {
                asrWorkerRef.current.terminate();
                asrWorkerRef.current = null;
                setIsAsrInitialized(false);
                setIsAsrInitializing(false);
            }
        };
    }, [isOfflineAsrEnabled, asrEngine, initializeAsrWorker]);


    // --- Common Effects ---
    useEffect(() => {
        const fetchStatuses = async () => {
            const statuses: Record<string, DownloadProgress> = {};
            for (const model of [...OFFLINE_MODELS, ...OFFLINE_MODELS_TS]) {
                if (model.value) {
                     statuses[model.value] = await downloadManager.getStatus(model.value, (model as any).dtype);
                }
            }
            setDownloadProgress(statuses);
        };
        fetchStatuses();
        
        checkAllAsrCacheStatus().then(statuses => {
            setAsrModelsCacheStatus(statuses);
        });

    }, [checkAllAsrCacheStatus]); 

    useEffect(() => {
        const handleOnline = () => setIsOnline(true);
        const handleOffline = () => setIsOnline(false);
        window.addEventListener('online', handleOnline);
        window.addEventListener('offline', handleOffline);
        return () => {
            window.removeEventListener('online', handleOnline);
            window.removeEventListener('offline', handleOffline);
        };
    }, []);
    
    useEffect(() => {
        // Load settings from localStorage
        const savedApiKey = localStorage.getItem('api-key');
        if (savedApiKey) setApiKey(savedApiKey);
        
        const savedModelName = localStorage.getItem('model-name');
        if (savedModelName) setModelName(savedModelName);

        const savedProvider = localStorage.getItem('online-provider');
        if (savedProvider) setOnlineProvider(savedProvider);

        const savedUrl = localStorage.getItem('openai-api-url');
        if (savedUrl) setOpenaiApiUrl(savedUrl);

        const savedHfApiKey = localStorage.getItem('hf-api-key');
        if (savedHfApiKey) setHuggingFaceApiKey(savedHfApiKey);
        
        const savedOfflineModel = localStorage.getItem('offline-model-name');
        if (savedOfflineModel) setOfflineModelName(savedOfflineModel);

        const savedOfflineMode = localStorage.getItem('offline-mode-enabled');
        if (savedOfflineMode) setIsOfflineModeEnabled(JSON.parse(savedOfflineMode));
        
        const savedTwoStep = localStorage.getItem('is-two-step-jp-cn-enabled');
        if (savedTwoStep) setIsTwoStepJpCn(JSON.parse(savedTwoStep));
        
        const savedTtsEnabled = localStorage.getItem('tts-enabled');
        if (savedTtsEnabled) setIsOfflineTtsEnabled(JSON.parse(savedTtsEnabled));

        const savedTtsVoice = localStorage.getItem('tts-voice-uri');
        if (savedTtsVoice) setOfflineTtsVoiceURI(savedTtsVoice);

        const savedTtsRate = localStorage.getItem('tts-rate');
        if (savedTtsRate) setOfflineTtsRate(JSON.parse(savedTtsRate));

        const savedTtsPitch = localStorage.getItem('tts-pitch');
        if (savedTtsPitch) setOfflineTtsPitch(JSON.parse(savedTtsPitch));

        const savedMaxTokens = localStorage.getItem('offline-max-tokens');
        if (savedMaxTokens) setOfflineMaxTokens(JSON.parse(savedMaxTokens));

        const savedTopK = localStorage.getItem('offline-top-k');
        if (savedTopK) setOfflineTopK(JSON.parse(savedTopK));

        const savedTemp = localStorage.getItem('offline-temperature');
        if (savedTemp) setOfflineTemperature(JSON.parse(savedTemp));
        
        const savedSeed = localStorage.getItem('offline-random-seed');
        if (savedSeed) setOfflineRandomSeed(JSON.parse(savedSeed));

        const savedAudio = localStorage.getItem('offline-support-audio');
        if (savedAudio) setOfflineSupportAudio(JSON.parse(savedAudio));

        const savedAudioRealtime = localStorage.getItem('offline-audio-realtime');
        if (savedAudioRealtime) setOfflineAudioRealtime(JSON.parse(savedAudioRealtime));

        const savedImages = localStorage.getItem('offline-max-num-images');
        if (savedImages) setOfflineMaxNumImages(JSON.parse(savedImages));

        // ASR Settings
        const savedAsrEnabled = localStorage.getItem('is-offline-asr-enabled');
        if (savedAsrEnabled) setIsOfflineAsrEnabled(JSON.parse(savedAsrEnabled));

        const savedRealtimeAsrEnabled = localStorage.getItem('is-realtime-asr-enabled');
        if (savedRealtimeAsrEnabled) setIsRealtimeAsrEnabled(JSON.parse(savedRealtimeAsrEnabled));

        const savedWebSpeechEnabled = localStorage.getItem('is-web-speech-api-enabled');
        if (savedWebSpeechEnabled) setIsWebSpeechApiEnabled(JSON.parse(savedWebSpeechEnabled));

        const savedAsrEngine = localStorage.getItem('asr-engine');
        if (savedAsrEngine) {
            setAsrEngine(savedAsrEngine as AsrEngineType);
            if (savedAsrEngine === 'nemotron') {
                setAsrModelId('nemotron');
            } else if (savedAsrEngine === 'qwen3') {
                setAsrModelId('qwen3');
            } else {
                const savedAsrModel = localStorage.getItem('asr-model-id');
                if (savedAsrModel) setAsrModelId(savedAsrModel);
            }
        } else {
            const savedAsrModel = localStorage.getItem('asr-model-id');
            if (savedAsrModel) setAsrModelId(savedAsrModel);
        }

        const savedAsrProfile = localStorage.getItem('asr-profile');
        if (savedAsrProfile) setAsrProfile(savedAsrProfile as NemotronProfile);

        const savedAsrBeamWidth = localStorage.getItem('asr-beam-width');
        if (savedAsrBeamWidth) setAsrBeamWidth(JSON.parse(savedAsrBeamWidth) as NemotronBeamWidth);

        const savedNoiseCancellation = localStorage.getItem('is-noise-cancellation-enabled');
        if (savedNoiseCancellation) setIsNoiseCancellationEnabled(JSON.parse(savedNoiseCancellation));

        const savedAudioGain = localStorage.getItem('audio-gain-value');
        if (savedAudioGain) setAudioGainValue(JSON.parse(savedAudioGain));

        const savedCustomModels = localStorage.getItem('custom-offline-models');
        if (savedCustomModels) setCustomModels(JSON.parse(savedCustomModels));

        const savedOcrModel = localStorage.getItem('selected-ocr-model');
        if (savedOcrModel && Object.prototype.hasOwnProperty.call(OCR_MODELS, savedOcrModel)) {
            setSelectedOcrModel(savedOcrModel as keyof typeof OCR_MODELS);
        }

        const savedOcrAutoInit = localStorage.getItem('is-ocr-auto-init-enabled');
        if (savedOcrAutoInit) setIsOcrAutoInitEnabled(JSON.parse(savedOcrAutoInit));

        try {
            const savedHistory = localStorage.getItem('translation-history');
            if (savedHistory) setHistory(JSON.parse(savedHistory));
        } catch (err: any) {
            console.error('Failed to load translation history:', err);
        }
    }, []);

    useEffect(() => {
        const loadVoices = () => setVoices(window.speechSynthesis.getVoices());
        window.speechSynthesis.onvoiceschanged = loadVoices;
        loadVoices();
    }, []);

    useEffect(() => {
        localStorage.setItem('source-lang', sourceLang.code);
    }, [sourceLang]);

    useEffect(() => {
        localStorage.setItem('target-lang', targetLang.code);
    }, [targetLang]);

    // --- Loading Queue Logic ---

    // Queue LLM initialization
    useEffect(() => {
        const modelToLoad = isModelDownloaded ? offlineModelName : null;
        if (isOfflineModeEnabled && modelToLoad && !isOfflineModelInitialized && !isOfflineModelInitializing) {
            setLoadingQueue(q => [...q, 'llm']);
        }
    }, [isModelDownloaded, offlineModelName, isOfflineModeEnabled, isOfflineModelInitialized, isOfflineModelInitializing]);

    // Unload LLM when disabled or model changed
    const previousOfflineModelRef = useRef(offlineModelName);
    
    useEffect(() => {
        if (!isOfflineModeEnabled && (isOfflineModelInitialized || isOfflineModelInitializing)) {
            if (workerRef.current) {
                console.log('Requesting offline model unload. (Offline mode disabled)');
                workerRef.current.postMessage({ type: 'unload' });
            }
        } else if (isOfflineModeEnabled && offlineModelName !== previousOfflineModelRef.current) {
             //console.log(`[App] Model changed from ${previousOfflineModelRef.current} to ${offlineModelName} - Unloading previous`);
             if (workerRef.current && (isOfflineModelInitialized || isOfflineModelInitializing)) {
                 workerRef.current.postMessage({ type: 'unload' });
             }
             setIsOfflineModelInitialized(false);
             setIsOfflineModelInitializing(false);
        }
        previousOfflineModelRef.current = offlineModelName;
    }, [isOfflineModeEnabled, offlineModelName, isOfflineModelInitialized, isOfflineModelInitializing]);

    // Queue ASR initialization
    useEffect(() => {
        if (isOfflineAsrEnabled && (asrModelId || asrEngine === 'nemotron' || asrEngine === 'qwen3') && !isAsrInitialized && !isAsrInitializing) {
            const checkAndQueue = async () => {
                const targetId = asrEngine === 'nemotron' ? 'nemotron' : (asrEngine === 'qwen3' ? 'qwen3' : asrModelId);
                const isCached = await checkAsrModelCacheStatus(targetId);
                if (isCached) {
                    setLoadingQueue(q => [...q, 'asr']);
                }
            };
            checkAndQueue();
        }
    }, [isOfflineAsrEnabled, asrModelId, asrEngine, isAsrInitialized, isAsrInitializing]);
    
    // Queue OCR initialization
    useEffect(() => {
        if (isOcrAutoInitEnabled && ocrEngineStatus === 'uninitialized') {
             setLoadingQueue(q => [...q, 'ocr']);
        }
    }, [isOcrAutoInitEnabled, ocrEngineStatus]);

    // Central orchestrator for the loading queue
    useEffect(() => {
        const isBusy = isOfflineModelInitializing || isAsrInitializing || ocrEngineStatus === 'initializing';
        if (isBusy || loadingQueue.length === 0) {
            return; // Wait until the current task is done or if the queue is empty
        }

        const nextTask = loadingQueue[0];        
        // Dequeue the task
        setLoadingQueue(q => q.slice(1));

        if (nextTask === 'llm') {
            const modelToLoad = offlineModelName;
            //console.log(`[Orchestrator] Sending init for LLM. Engine flag check!`, { modelToLoad });
            setIsOfflineModelInitializing(true);
            setIsOfflineModelInitialized(false);
            
            const isTSModel = OFFLINE_MODELS_TS.some(m => m.value === modelToLoad);
            //console.log(`[Orchestrator] isTSModel?`, isTSModel);
            
            if (isTSModel) {
                 const model = OFFLINE_MODELS_TS.find(m => m.value === modelToLoad);
                 downloadManager.resolveDtypeForModel(modelToLoad, model?.dtype || 'q4').then(resolvedDtype => {
                     const options = {
                         maxTokens: offlineMaxTokens, topK: offlineTopK, temperature: offlineTemperature,
                         randomSeed: offlineRandomSeed, supportAudio: offlineSupportAudio, audioRealtime: offlineAudioRealtime, maxNumImages: offlineMaxNumImages,
                         dtype: resolvedDtype,
                         generationMode: model?.generationMode || 'Gemma4ForConditionalGeneration'
                     };
                     getOrCreateWorker().postMessage({ type: 'init', payload: { engine: 'transformers', modelSource: modelToLoad, options } });
                 }).catch(err => {
                     const message = err instanceof Error ? err.message : t('notifications.offlineModelInitFailed');
                     showNotification(message, 'error');
                     setIsOfflineModelInitializing(false);
                 });
            } else {
                 downloadManager.getModelAsBlob(modelToLoad).then(modelBlob => {
                    if (!modelBlob) throw new Error(`Model blob for ${modelToLoad} not found.`);
                    const options = {
                        maxTokens: offlineMaxTokens, topK: offlineTopK, temperature: offlineTemperature,
                        randomSeed: offlineRandomSeed, supportAudio: offlineSupportAudio, audioRealtime: offlineAudioRealtime, maxNumImages: offlineMaxNumImages,
                    };
                    const engine = modelToLoad.toLowerCase().includes('gemma-4') ? 'litert-lm' : 'mediapipe';
                    getOrCreateWorker().postMessage({ type: 'init', payload: { engine, modelBlob, modelSource: modelToLoad, options } });
                 }).catch(err => {
                    const message = err instanceof Error ? err.message : t('notifications.offlineModelInitFailed');
                    showNotification(message, 'error');
                    setIsOfflineModelInitializing(false);
                 });
            }
        } else if (nextTask === 'asr') {
            if (asrEngine === 'nemotron') {
                if (asrWorkerRef.current) {
                    setIsAsrInitializing(true);
                    setAsrLoadingProgress({ file: '', progress: 0 });
                    asrWorkerRef.current.postMessage({ type: 'load', payload: { asrProfile, asrBeamWidth } });
                }
            } else if (asrEngine === 'qwen3') {
                if (asrWorkerRef.current) {
                    setIsAsrInitializing(true);
                    setAsrLoadingProgress({ file: '', progress: 0 });
                    asrWorkerRef.current.postMessage({ type: 'load', payload: {} });
                }
            } else {
                const model = ASR_MODELS.find(m => m.id === asrModelId);
                if (model && asrWorkerRef.current) {
                    setIsAsrInitializing(true);
                    setAsrLoadingProgress({ file: '', progress: 0 });
                    asrWorkerRef.current.postMessage({ type: 'load', payload: { modelId: model.id, quantization: model.quantization } });
                }
            }
        } else if (nextTask === 'ocr') {
            const ocrModelConfig = { key: selectedOcrModel, ...OCR_MODELS[selectedOcrModel].paths };
            initializeOcr(ocrModelConfig);
        }
    }, [
        loadingQueue, isOfflineModelInitializing, isAsrInitializing, ocrEngineStatus,
        offlineModelName, asrModelId, selectedOcrModel, asrEngine, asrProfile, asrBeamWidth,
        offlineMaxTokens, offlineTopK, offlineTemperature, offlineRandomSeed, offlineSupportAudio, offlineAudioRealtime, offlineMaxNumImages,
        getOrCreateWorker, initializeOcr, showNotification, t
    ]);


    const handleTranslate = useCallback(() => {
        performTranslate(inputText);
    }, [inputText, performTranslate]);

    const handleCancelTranslation = useCallback(() => {
        if (isOfflineModeEnabled) {
            if (workerRef.current) {
                workerRef.current.postMessage({ type: 'cancel_translation' });
            }
        } else {
            translationAbortControllerRef.current?.abort();
        }
        
        if (asrWorkerRef.current) {
            asrWorkerRef.current.postMessage({ type: 'cancel' });
        }
        
        if (audioAbortControllerRef.current) {
            audioAbortControllerRef.current.abort();
            audioAbortControllerRef.current = null;
        }
        setIsTranscribing(false);
    }, [isOfflineModeEnabled]);

    const handleSwapLanguages = useCallback(() => {
        if (sourceLang.code === 'auto') return;
        setSourceLang(targetLang);
        setTargetLang(sourceLang);
        setInputText(translatedText);
        setTranslatedText(inputText);
    }, [sourceLang, targetLang, inputText, translatedText]);

    const handleSpeak = useCallback(async (gender: 'female' | 'male') => {
        if (!translatedText) return;

        // Stop any current speech
        if (window.speechSynthesis.speaking) {
            window.speechSynthesis.cancel();
            if (isSpeaking && (isOfflineTtsEnabled || speakingGender === gender)) {
                setIsSpeaking(false);
                setSpeakingGender(null);
                return;
            }
        }

        setIsSpeaking(true);
        setSpeakingGender(gender); // Simplified logic for gender UI state

        // Browser/Offline TTS
        if (!('speechSynthesis' in window)) {
             setIsSpeaking(false);
             setSpeakingGender(null);
             return;
        }

        const utterance = new SpeechSynthesisUtterance(translatedText);
        utterance.lang = normalizeWebSpeechLang(targetLang.code);

        if (isOfflineTtsEnabled) {
            const selectedVoice = voices.find(v => v.voiceURI === offlineTtsVoiceURI);
            if (selectedVoice) {
                utterance.voice = selectedVoice;
            }
            utterance.rate = offlineTtsRate;
            utterance.pitch = offlineTtsPitch;
        } else {
            const langVoices = filterVoicesForLanguage(voices, targetLang.code);

            if (langVoices.length > 0) {
                const femaleVoice = langVoices.find(v => /female|women|girl|mei-jia|zira|ayumi|kyoko|xiaoxiao|yating|sin-ji/i.test(v.name));
                const maleVoice = langVoices.find(v => /male|men|boy|liang|ichiro|yunxi|danny|yunjhe/i.test(v.name));
                let selectedVoice: SpeechSynthesisVoice | undefined;
                if (gender === 'female') {
                    selectedVoice = femaleVoice || langVoices.find(v => v !== maleVoice) || langVoices[0];
                } else {
                    selectedVoice = maleVoice || langVoices.find(v => v !== femaleVoice) || langVoices[0];
                }
                utterance.voice = selectedVoice;
            }
        }

        utterance.onstart = () => {
            setIsSpeaking(true);
            setSpeakingGender(isOfflineTtsEnabled ? null : gender);
        };
        utterance.onend = () => {
            setIsSpeaking(false);
            setSpeakingGender(null);
        };
        utterance.onerror = (event) => {
            console.error('SpeechSynthesisUtterance.onerror', event);
            showNotification(t('notifications.speechError', { error: event.error }), 'error');
            setIsSpeaking(false);
            setSpeakingGender(null);
        };
        window.speechSynthesis.speak(utterance);
    }, [translatedText, targetLang, voices, isSpeaking, speakingGender, showNotification, isOfflineTtsEnabled, offlineTtsVoiceURI, offlineTtsRate, offlineTtsPitch, t, isOnline, onlineProvider, apiKey]);

    // Recording Logic
    const handleStopRecording = () => {
        if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
            mediaRecorderRef.current.stop();
        }
        if (audioAbortControllerRef.current) {
            audioAbortControllerRef.current.abort();
            audioAbortControllerRef.current = null;
        }
    };

    const handleStartRecording = (onStop: (audioBlob: Blob) => void, onDataAvailable?: (audioBlob: Blob) => void, onChunkAvailable?: (audioBlob: Blob) => void) => {
        if (isRecording || isAstRecording) return;

        navigator.mediaDevices.getUserMedia({ audio: true })
            .then(async stream => {
                lastProcessedSampleIndexRef.current = 0;
                onStopRecordingCallbackRef.current = onStop;

                const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 16000 });
                const source = audioContext.createMediaStreamSource(stream);
                let lastNode: AudioNode = source;

                if (isNoiseCancellationEnabled) {
                    const filter = audioContext.createBiquadFilter();
                    filter.type = 'highpass';
                    filter.frequency.value = 100;
                    lastNode.connect(filter);
                    lastNode = filter;
                }

                if (audioGainValue && audioGainValue !== 1.0) {
                    const gainNode = audioContext.createGain();
                    gainNode.gain.value = audioGainValue;
                    lastNode.connect(gainNode);
                    lastNode = gainNode;
                }

                if (!globalThis.fluidVadWorkletUrl) {
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
                const vad = await createVad({ threshold: 0.5, minSilenceDuration: 1.5, maxSpeechDuration:30, speechPadding:0.4 }, { wasm: wasmBuffer });
                
                const processor = new AudioWorkletNode(audioContext, "fluidvad-processor");

                let accumulatedTotal: Float32Array[] = [];
                let realtimeCounter = 0;

                const mergeFloat32Arrays = (arrays: Float32Array[]): Float32Array => {
                    const totalLength = arrays.reduce((acc, val) => acc + val.length, 0);
                    const merged = new Float32Array(totalLength);
                    let offset = 0;
                    for (const arr of arrays) {
                        merged.set(arr, offset);
                        offset += arr.length;
                    }
                    return merged;
                };

                const float32ToWavBlob = (samples: Float32Array): Blob => {
                    const buffer = audioContext.createBuffer(1, samples.length, 16000);
                    buffer.copyToChannel(samples, 0);
                    return audioBufferToWav(buffer);
                };

                processor.port.onmessage = (e) => {
                    const samples = new Float32Array(e.data);
                    accumulatedTotal.push(samples);

                    const events = vad.push(samples);
                    for (const event of events) {
                        if (!event.isStart && mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
                            mediaRecorderRef.current.stop();
                        }
                    }
                    
                    if (vad.isSpeaking) {
                        realtimeCounter++;
                        if (onDataAvailable && realtimeCounter >= 16) {
                            realtimeCounter = 0;
                            const merged = mergeFloat32Arrays(accumulatedTotal);
                            const blob = float32ToWavBlob(merged);
                            onDataAvailable(blob);
                        }
                    } else {
                        realtimeCounter = 0;
                    }
                };

                lastNode.connect(processor);
                processor.connect(audioContext.destination);

                mediaRecorderRef.current = {
                    state: 'recording',
                    stop: () => {
                        mediaRecorderRef.current.state = 'inactive';
                        let finalBlob = new Blob([], { type: 'audio/wav' });
                        if (accumulatedTotal.length > 0) {
                            const merged = mergeFloat32Arrays(accumulatedTotal);
                            finalBlob = float32ToWavBlob(merged);
                        }
                        
                        if (onStopRecordingCallbackRef.current) {
                            onStopRecordingCallbackRef.current(finalBlob);
                        }
                        
                        stream.getTracks().forEach(track => track.stop());
                        processor.disconnect();
                        audioContext.close();
                        mediaRecorderRef.current = null;
                        onStopRecordingCallbackRef.current = null;
                        setIsRecording(false);
                        setIsAstRecording(false);
                    },
                    mimeType: 'audio/wav'
                } as any;
            })
            .catch(err => {
                showNotification(`Could not start recording: ${err.message}`, 'error');
                setIsRecording(false);
                setIsAstRecording(false);
            });
    };

    // Memoized Callbacks for Web Speech to prevent infinite loops
    const handleWebSpeechResult = useCallback((transcript: string, isFinal: boolean) => {
        const transcribedText = transcript.trim();
        const isMeaningless = !transcribedText || 
            /^[^a-zA-Z0-9\p{L}]+$/u.test(transcribedText) || 
            /^[\[\(].*[\]\)]$/.test(transcribedText) || 
            ['恩', '嗯', '啊', '哦', '喔', '呃', 'um', 'uh', 'ah', 'oh'].includes(transcribedText.toLowerCase());

        setInputText(transcript);
        if (isFinal) {
            setIsRecording(false);
            setIsAstRecording(false);
            if (transcribedText && !isMeaningless) {
                if (isReverseTranslateRef.current) {
                    performReverseTranslateRef.current(transcript);
                    isReverseTranslateRef.current = false;
                } else if (isRealtimeAsrEnabledRef.current) {
                    performTranslateRef.current(transcript);
                }
            } else {
                isReverseTranslateRef.current = false;
            }
        }
    }, []);

    const handleWebSpeechError = useCallback((error: string) => {
        showNotification(t('notifications.speechRecognitionError', { error }), 'error');
    }, [showNotification, t]);

    const handleWebSpeechEnd = useCallback(() => {
        setIsRecording(false);
        setIsAstRecording(false);
        isReverseTranslateRef.current = false;
    }, []);

    // Create a stable callback for onStart to prevent infinite loops
    const handleWebSpeechStart = useCallback(() => {
        // No-op or logic if needed
    }, []);

    const webSpeech = useWebSpeech({
        onResult: handleWebSpeechResult,
        onError: handleWebSpeechError,
        onStart: handleWebSpeechStart,
        onEnd: handleWebSpeechEnd,
    });

    const stopAllRecordings = useCallback(() => {
        if (isGeminiLiveModel) {
            liveServiceRef.current?.stop();
        } else if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
            mediaRecorderRef.current.stop();
        } else if (webSpeech.isListening) {
            webSpeech.stopRecognition();
        }
    }, [webSpeech, isGeminiLiveModel]);

    useEffect(() => {
        // Only apply 30-second countdown if realtime ASR is disabled AND we are using offline ASR (Whisper)
        // Whisper has a strict 30-second audio limit.
        const shouldCountdown = isOfflineAsrEnabled && !isRealtimeAsrEnabled && (isRecording || isAstRecording);
        
        if (shouldCountdown) {
            setRecordingCountdown(30);
            countdownTimerRef.current = window.setInterval(() => {
                setRecordingCountdown(prev => {
                    if (prev !== null && prev <= 1) {
                        if (countdownTimerRef.current) {
                            clearInterval(countdownTimerRef.current);
                            countdownTimerRef.current = null;
                        }
                        stopAllRecordings();
                        return null;
                    }
                    return prev !== null ? prev - 1 : null;
                });
            }, 1000);
        } else {
            if (countdownTimerRef.current) {
                clearInterval(countdownTimerRef.current);
                countdownTimerRef.current = null;
            }
            setRecordingCountdown(null);
        }
        return () => {
            if (countdownTimerRef.current) clearInterval(countdownTimerRef.current);
        };
    }, [isRecording, isAstRecording, stopAllRecordings, isRealtimeAsrEnabled, isOfflineAsrEnabled]);


    const handleToggleRecording = useCallback(() => {
        if (isGeminiLiveModel) {
            if (isRecording) {
                liveServiceRef.current?.stop();
                setIsRecording(false);
                isReverseTranslateRef.current = false;
            } else {
                if (isAstRecording) {
                    liveServiceRef.current?.stop();
                    setIsAstRecording(false);
                }
				accumulatedInputRef.current = '';
                accumulatedTranslatedRef.current = '';
                currentActiveInputRef.current = '';
                currentActiveTranslatedRef.current = '';
                setInputText('');
                setIsRecording(true);
                isReverseTranslateRef.current = false;
                setInputText(t('notifications.liveTranslateListening'));
                const targetLangCode = targetLang.code === 'auto' ? 'zh' : (targetLang.code === 'zh-Hans' ? 'zh-Hans' : (targetLang.asrCode || targetLang.code));
                liveServiceRef.current?.start(apiKey, targetLangCode);
            }
            return;
        }

        const isCurrentlyRecording = isRecording || (!isOfflineAsrEnabled && isWebSpeechApiEnabled && webSpeech.isListening && !isReverseTranslateRef.current);
        if (isCurrentlyRecording) {
            if (isOfflineAsrEnabled || !isWebSpeechApiEnabled) {
                handleStopRecording();
            } else {
                webSpeech.stopRecognition();
            }
        } else {
            if (isAstRecording) handleStopRecording();
            if (sourceLang.code === 'auto' && (isOfflineAsrEnabled || !isWebSpeechApiEnabled)) {
                 showNotification(t('notifications.selectLanguageError'), 'info');
                 return;
            }
            
            setInputText('');
            setIsRecording(true);
            setInputText(t('translationInput.placeholderListening'));

            if (isOfflineAsrEnabled) {
                // Offline ASR Path (Whisper/Sherpa)
                handleStartRecording(async (audioBlob) => {
                    if (!isRealtimeAsrEnabled) {
                        setInputText(t('notifications.transcribing'));
                    }
                    setIsTranscribing(true);
                    try {
                        const audioData = await processAudioForTranscription(audioBlob, { noiseSuppression: false, gain: 1.0 });
                        if (asrWorkerRef.current) {
                            asrWorkerRef.current.postMessage({ 
                                type: 'transcribe', 
                                payload: { 
                                    audio: audioData, 
                                    asrLanguage: sourceLang.asrCode,
                                    promptLanguage: sourceLang.code,
                                    isFinal: true
                                } 
                            });
                        } else {
                            throw new Error('ASR Worker is not initialized.');
                        }
                    } catch (err) {
                        console.error(err);
                        setInputText('');
                        setIsTranscribing(false);
                        const message = err instanceof Error ? err.message : 'Transcription failed.';
                        showNotification(message, 'error');
                    }
                }, isRealtimeAsrEnabled ? async (audioBlob) => {
                    if (isAsrProcessingRef.current) return;
                    try {
                        isAsrProcessingRef.current = true;
                        const audioData = await processAudioForTranscription(audioBlob, { noiseSuppression: false, gain: 1.0 });
                        // Ignore fragments shorter than 0.8s (12800 samples at 16kHz) to avoid Whisper hallucinations on noise
                        if (audioData.length < 12800) {
                            isAsrProcessingRef.current = false;
                            return;
                        }
                        if (asrWorkerRef.current) {
                            asrWorkerRef.current.postMessage({ 
                                type: 'transcribe', 
                                payload: { 
                                    audio: audioData, 
                                    asrLanguage: sourceLang.asrCode,
                                    promptLanguage: sourceLang.code,
                                    isFinal: false
                                } 
                            });
                        } else {
                            isAsrProcessingRef.current = false;
                        }
                    } catch (err) {
                        console.error(err);
                        isAsrProcessingRef.current = false;
                    }
                } : undefined);
            } else if (isWebSpeechApiEnabled) { 
                webSpeech.startRecognition(sourceLang.code);
            } else if (isOfflineModeEnabled && offlineSupportAudio) {
                // Gemma 3N Audio Path
                if (isOfflineModelReady) {
                     setInputText(t('notifications.transcribing'));
                     setIsTranscribing(true);
                     handleStartRecording(async (audioBlob) => {
                        const audioData = await processAudioForTranscription(audioBlob, { noiseSuppression: false, gain: 1.0 }); 
                        const tempAudioContext = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 16000 });
                        const audioBuffer = tempAudioContext.createBuffer(1, audioData.length, 16000);
                        audioBuffer.copyToChannel(audioData, 0);
                        const pcmWavBlob = audioBufferToWav(audioBuffer);
                        await tempAudioContext.close();
                        
                        const worker = getOrCreateWorker();
                        worker.postMessage({ 
                            type: 'transcribe', 
                            payload: { 
                                audioData: pcmWavBlob, 
                                sourceLang: sourceLang.code === 'auto' ? 'Auto Detect' : sourceLang.code,
                                isStream: false
                            } 
                        });
                     }, undefined, offlineAudioRealtime ? async (audioBlob) => {
                        if (isAsrProcessingRef.current) return;
                        try {
                            isAsrProcessingRef.current = true;
                            const audioData = await processAudioForTranscription(audioBlob, { noiseSuppression: false, gain: 1.0 }); 
                            
                            const newAudioData = audioData.slice(lastProcessedSampleIndexRef.current);
                            lastProcessedSampleIndexRef.current = audioData.length;

                            if (newAudioData.length === 0) {
                                isAsrProcessingRef.current = false;
                                return;
                            }

                            const tempAudioContext = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 16000 });
                            const audioBuffer = tempAudioContext.createBuffer(1, newAudioData.length, 16000);
                            audioBuffer.copyToChannel(newAudioData, 0);
                            const pcmWavBlob = audioBufferToWav(audioBuffer);
                            await tempAudioContext.close();
                            
                            const worker = getOrCreateWorker();
                            worker.postMessage({ 
                                type: 'transcribe', 
                                payload: { 
                                    audioData: pcmWavBlob, 
                                    sourceLang: sourceLang.code === 'auto' ? 'Auto Detect' : sourceLang.code,
                                    isStream: true
                                } 
                            });
                        } catch (err) {
                            console.error(err);
                            isAsrProcessingRef.current = false;
                        }
                     } : undefined);
                } else {
                     showNotification(t('notifications.offlineModelNotReadyRecording'), 'error');
                     setIsRecording(false);
                }
            } else { 
                // Online Multimodal ASR (OpenAI/Gemini)
                handleStartRecording(async (audioBlob) => {
                    setInputText(t('notifications.transcribingApi'));
                    const controller = new AbortController();
                    audioAbortControllerRef.current = controller;
                    try {
                        const processedBlob = await processAudioToWav(audioBlob, false, 1.0);
                        let transcribedText = '';
                        if (onlineProvider === 'openai') {
                            const langCode = sourceLang.code.split('-')[0];
                            transcribedText = await transcribeAudioOpenAI(processedBlob, langCode, apiKey, openaiApiUrl, controller.signal);
                        } else {
                            const langName = sourceLang.code === 'auto' ? 'Auto Detect' : sourceLang.code;
                            transcribedText = await transcribeAudioGemini(processedBlob, langName, apiKey, textModelName, controller.signal);
                        }
                        setInputText(transcribedText);
                    } catch (err) {
                        if (err instanceof DOMException && err.name === 'AbortError') return;
                        const message = err instanceof Error ? err.message : 'Transcription failed.';
                        showNotification(message, 'error');
                        setInputText('');
                    } finally {
                        audioAbortControllerRef.current = null;
                    }
                });
            }
        }
    }, [isRecording, isAstRecording, showNotification, t, sourceLang, targetLang, isOfflineAsrEnabled, isRealtimeAsrEnabled, webSpeech, isNoiseCancellationEnabled, audioGainValue, isWebSpeechApiEnabled, onlineProvider, apiKey, openaiApiUrl, modelName, isOfflineModeEnabled, isOfflineModelReady, offlineSupportAudio, getOrCreateWorker, i18n, isGeminiLiveModel, liveServiceRef]);

    const handleToggleAstRecording = useCallback(() => {
        if (isGeminiLiveModel) {
            if (isAstRecording) {
                liveServiceRef.current?.stop();
                setIsAstRecording(false);
                isReverseTranslateRef.current = false;
            } else {
                if (isRecording) {
                    liveServiceRef.current?.stop();
                    setIsRecording(false);
                }
				accumulatedInputRef.current = '';
                accumulatedTranslatedRef.current = '';
                currentActiveInputRef.current = '';
                currentActiveTranslatedRef.current = '';
                setInputText('');
                setIsAstRecording(true);
                isReverseTranslateRef.current = true;
                setInputText(t('notifications.liveTranslateListeningReverse'));
                const targetLangCode = sourceLang.code === 'auto' ? 'zh' : (sourceLang.code === 'zh-Hans' ? 'zh-Hans' : (sourceLang.asrCode || sourceLang.code));
                liveServiceRef.current?.start(apiKey, targetLangCode);
            }
            return;
        }

        const isCurrentlyAstRecording = isAstRecording || (!isOfflineAsrEnabled && isWebSpeechApiEnabled && webSpeech.isListening && isReverseTranslateRef.current);

        if (isCurrentlyAstRecording) {
             if (isOfflineAsrEnabled || !isWebSpeechApiEnabled) {
                handleStopRecording();
            } else {
                webSpeech.stopRecognition();
            }
        } else {
            if (isRecording) handleStopRecording();
            if (targetLang.code === 'auto') {
                showNotification(t('notifications.astSelectLanguage'), 'info');
                return;
            }
            
            setInputText('');
            setTranslatedText('');
            setIsAstRecording(true);
            isReverseTranslateRef.current = true;
            
            if (isOfflineAsrEnabled) {
                 handleStartRecording(async (audioBlob) => {
                    if (!isRealtimeAsrEnabled) {
                        setInputText(t('notifications.transcribing'));
                    }
                    setIsTranscribing(true);
                    try {
                        const audioData = await processAudioForTranscription(audioBlob, { noiseSuppression: false, gain: 1.0 });
                        if (asrWorkerRef.current) {
                            asrWorkerRef.current.postMessage({ 
                                type: 'transcribe', 
                                payload: { 
                                    audio: audioData, 
                                    asrLanguage: targetLang.asrCode,
                                    promptLanguage: targetLang.code,
                                    isFinal: true
                                } 
                            });
                        } else {
                            throw new Error('ASR Worker is not initialized.');
                        }
                    } catch (err) {
                        console.error(err);
                        setTranslatedText('');
                        setIsTranscribing(false);
                        isReverseTranslateRef.current = false;
                        const message = err instanceof Error ? err.message : 'Transcription failed.';
                        showNotification(message, 'error');
                    }
                }, isRealtimeAsrEnabled ? async (audioBlob) => {
                    if (isAsrProcessingRef.current) return;
                    try {
                        isAsrProcessingRef.current = true;
                        const audioData = await processAudioForTranscription(audioBlob, { noiseSuppression: false, gain: 1.0 });
                        // Ignore fragments shorter than 0.8s (12800 samples at 16kHz) to avoid Whisper hallucinations on noise
                        if (audioData.length < 12800) {
                            isAsrProcessingRef.current = false;
                            return;
                        }
                        if (asrWorkerRef.current) {
                            asrWorkerRef.current.postMessage({ 
                                type: 'transcribe', 
                                payload: { 
                                    audio: audioData, 
                                    asrLanguage: targetLang.asrCode,
                                    promptLanguage: targetLang.code,
                                    isFinal: false
                                } 
                            });
                        } else {
                            isAsrProcessingRef.current = false;
                        }
                    } catch (err) {
                        console.error(err);
                        isAsrProcessingRef.current = false;
                    }
                } : undefined);
            } else if (isWebSpeechApiEnabled) {
                webSpeech.startRecognition(targetLang.code);
            } else if (isOfflineModeEnabled && offlineSupportAudio) {
                 if (isOfflineModelReady) {
                    setIsAwaitingAstTranslation(true);
                    setInputText(t('notifications.transcribing'));
                    setIsTranscribing(true);
                    handleStartRecording(async (audioBlob) => {
                       const audioData = await processAudioForTranscription(audioBlob, { noiseSuppression: false, gain: 1.0 }); 
                       const tempAudioContext = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 16000 });
                       const audioBuffer = tempAudioContext.createBuffer(1, audioData.length, 16000);
                       audioBuffer.copyToChannel(audioData, 0);
                       const pcmWavBlob = audioBufferToWav(audioBuffer);
                       await tempAudioContext.close();
                       
                       const worker = getOrCreateWorker();
                       worker.postMessage({ 
                           type: 'transcribe', 
                           payload: { 
                               audioData: pcmWavBlob, 
                               sourceLang: targetLang.code,
                               isStream: false
                           } 
                       });
                    }, undefined, offlineAudioRealtime ? async (audioBlob) => {
                       if (isAsrProcessingRef.current) return;
                       try {
                           isAsrProcessingRef.current = true;
                           const audioData = await processAudioForTranscription(audioBlob, { noiseSuppression: false, gain: 1.0 }); 
                           
                           const newAudioData = audioData.slice(lastProcessedSampleIndexRef.current);
                           lastProcessedSampleIndexRef.current = audioData.length;

                           if (newAudioData.length === 0) {
                               isAsrProcessingRef.current = false;
                               return;
                           }

                           const tempAudioContext = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 16000 });
                           const audioBuffer = tempAudioContext.createBuffer(1, newAudioData.length, 16000);
                           audioBuffer.copyToChannel(newAudioData, 0);
                           const pcmWavBlob = audioBufferToWav(audioBuffer);
                           await tempAudioContext.close();
                           
                           const worker = getOrCreateWorker();
                           worker.postMessage({ 
                               type: 'transcribe', 
                               payload: { 
                                   audioData: pcmWavBlob, 
                                   sourceLang: targetLang.code,
                                   isStream: true
                               } 
                           });
                       } catch (err) {
                           console.error(err);
                           isAsrProcessingRef.current = false;
                       }
                    } : undefined);
               } else {
                    showNotification(t('notifications.offlineModelNotReadyRecording'), 'error');
                    setIsAstRecording(false);
                    isReverseTranslateRef.current = false;
               }
            } else {
                 handleStartRecording(async (audioBlob) => {
                    setInputText(t('notifications.transcribingApi'));
                    const controller = new AbortController();
                    audioAbortControllerRef.current = controller;
                    try {
                        const processedBlob = await processAudioToWav(audioBlob, false, 1.0);
                        let transcribedText = '';
                        if (onlineProvider === 'openai') {
                            const langCode = targetLang.code.split('-')[0];
                            transcribedText = await transcribeAudioOpenAI(processedBlob, langCode, apiKey, openaiApiUrl, controller.signal);
                        } else {
                            const langName = targetLang.code;
                            transcribedText = await transcribeAudioGemini(processedBlob, langName, apiKey, textModelName, controller.signal);
                        }
                        setInputText(transcribedText);
                        if (transcribedText.trim()) {
                            performReverseTranslate(transcribedText);
                        }
                    } catch (err) {
                        if (err instanceof DOMException && err.name === 'AbortError') return;
                        const message = err instanceof Error ? err.message : 'Transcription failed.';
                        showNotification(message, 'error');
                        setInputText('');
                    } finally {
                        audioAbortControllerRef.current = null;
                        isReverseTranslateRef.current = false;
                    }
                });
            }
        }
    }, [isAstRecording, isRecording, sourceLang, targetLang, showNotification, t, isOfflineAsrEnabled, isRealtimeAsrEnabled, webSpeech, isNoiseCancellationEnabled, audioGainValue, isWebSpeechApiEnabled, onlineProvider, apiKey, openaiApiUrl, modelName, isOfflineModeEnabled, offlineSupportAudio, isOfflineModelReady, getOrCreateWorker, performReverseTranslate, i18n, isGeminiLiveModel, liveServiceRef]);

    const handleImageCaptured = useCallback(async (imageDataUrl: string) => {
        //console.log('[App] handleImageCaptured triggered. OCR Status:', ocrEngineStatus);
        setIsCameraOpen(false);
        setIsLoading(true);
        setInputText(t('notifications.processingImage'));
        setInputText('');
    
        // Await two animation frames AND a short timeout to securely flush the React unmount 
        // before starting heavy worker tasks (WebGPU compilation might freeze GPU/UI threads)
        await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
        await new Promise(resolve => setTimeout(resolve, 50));

        try {
            // Priority 1: Use local OCR if it's initialized and ready.
            if (ocrEngineStatus === 'ready') {
                const image = new Image();
                image.src = imageDataUrl;
                await new Promise<void>((resolve, reject) => {
                    image.onload = () => {
                        //console.log('[App] Local image loaded for OCR');
                        resolve();
                    };
                    image.onerror = (e) => {
                        //console.error('[App] Failed to load local image for OCR');
                        reject(e);
                    };
                });
    
                const recognitionData = await recognize(image, (chunk: string) => {
                    setInputText(prev => {
                        const trimmedChunk = chunk.trim();
                        if (!trimmedChunk) return prev;
                        if (prev === t('notifications.processingImage') || prev === '') {
                            return trimmedChunk + '\n';
                        }
                        return prev + trimmedChunk + '\n';
                    });
                });
                if (!recognitionData) {
                    throw new Error('OCR recognition returned no data.');
                }
                const extractedText = processOcrResult(recognitionData.result);
                setInputText(extractedText);
    
                if (extractedText.trim()) {
                    await performTranslate(extractedText);
                } else {
                    showNotification(t('notifications.noTextInImage'), 'info');
                    setInputText('');
                    setIsLoading(false);
                }
                return;
            }
            
            //console.log('[App] Falling back to non-local OCR pathways...');
    
            // Fallback to other methods
            if (isOfflineModeEnabled) {
                if (!isOfflineModelReady || offlineMaxNumImages < 1) {
                    throw new Error(t('notifications.offlineImageError'));
                }
                const imageBitmap = await new Promise<ImageBitmap>((resolve, reject) => {
                    const img = new Image();
                    img.onload = () => createImageBitmap(img).then(resolve).catch(reject);
                    img.onerror = () => reject(new Error('Failed to load image for bitmap.'));
                    img.src = imageDataUrl;
                });
    
                const worker = getOrCreateWorker();
                worker.postMessage({ type: 'extractText', payload: { imageBitmap } }, [imageBitmap]);
    
            } else { // Online mode
                if (!isOnline) throw new Error(t('notifications.offlineImageTranslateError'));
                
                let result: { sourceText: string, translatedText: string };
                const targetLangEn = targetLang.code;
                if (onlineProvider === 'openai') {
                    if (!apiKey) throw new Error("OpenAI API Key is not set.");
                    if (!openaiApiUrl) throw new Error("OpenAI API URL is not set.");
                    result = await translateImageOpenAI(imageDataUrl, targetLangEn, apiKey, textModelName, openaiApiUrl);
                } else {
                    if (!apiKey) throw new Error("Gemini API Key is not set.");
                    result = await translateImageGemini(imageDataUrl, targetLangEn, apiKey, textModelName);
                }
                
                setInputText(result.sourceText);
                setTranslatedText(result.translatedText);
    
                const newHistoryItem: TranslationHistoryItem = {
                    id: Date.now(), inputText: result.sourceText, translatedText: result.translatedText, sourceLang, targetLang,
                };
                setHistory(prevHistory => {
                    const updatedHistory = [newHistoryItem, ...prevHistory].slice(0, 50);
                    localStorage.setItem('translation-history', JSON.stringify(updatedHistory));
                    return updatedHistory;
                });
                setIsLoading(false);
            }
        } catch (err) {
            const errorMessage = err instanceof Error ? err.message : 'An unknown error occurred.';
            showNotification(t('notifications.imageProcessingFailed', { errorMessage }), 'error');
            setInputText(''); 
            setIsLoading(false);
        }
    }, [isOfflineModeEnabled, isOfflineModelReady, offlineMaxNumImages, isOnline, apiKey, modelName, targetLang, sourceLang, showNotification, onlineProvider, openaiApiUrl, t, i18n, getOrCreateWorker, ocrEngineStatus, recognize, performTranslate]);

    const handleSaveSettings = (
        newApiKey: string, 
        newModelName: string, 
        newHfApiKey: string, 
        newOfflineModel: string,
        newAsrModelId: string, 
        isOfflineEnabled: boolean,
        newIsOfflineAsrEnabled: boolean,
        newIsRealtimeAsrEnabled: boolean,
        newIsWebSpeechApiEnabled: boolean,
        newOnlineProvider: string,
        newOpenaiApiUrl: string,
        newIsTtsEnabled: boolean,
        newTtsVoiceURI: string,
        newTtsRate: number,
        newTtsPitch: number,
        newIsTwoStepJpCn: boolean,
        newOfflineMaxTokens: number,
        newOfflineTopK: number,
        newOfflineTemperature: number,
        newOfflineRandomSeed: number,
        newOfflineSupportAudio: boolean,
        newOfflineAudioRealtime: boolean,
        newOfflineMaxNumImages: number,
        newIsNoiseCancellationEnabled: boolean,
        newAudioGainValue: number,
        newSelectedOcrModel: keyof typeof OCR_MODELS,
        newIsOcrAutoInitEnabled: boolean,
        newAsrEngine: AsrEngineType,
        newAsrProfile: NemotronProfile,
        newAsrBeamWidth: NemotronBeamWidth
    ) => {
        setApiKey(newApiKey);
        setModelName(newModelName);
        setOnlineProvider(newOnlineProvider);
        setOpenaiApiUrl(newOpenaiApiUrl);
        setHuggingFaceApiKey(newHfApiKey);
        setOfflineModelName(newOfflineModel);
        setAsrModelId(newAsrModelId);
        setAsrEngine(newAsrEngine);
        setAsrProfile(newAsrProfile);
        setAsrBeamWidth(newAsrBeamWidth);
        setIsOfflineModeEnabled(isOfflineEnabled);
        setIsOfflineAsrEnabled(newIsOfflineAsrEnabled);
        setIsRealtimeAsrEnabled(newIsRealtimeAsrEnabled);
        setIsWebSpeechApiEnabled(newIsWebSpeechApiEnabled);
        setIsTwoStepJpCn(newIsTwoStepJpCn);
        setIsOfflineTtsEnabled(newIsTtsEnabled);
        setOfflineTtsVoiceURI(newTtsVoiceURI);
        setOfflineTtsRate(newTtsRate);
        setOfflineTtsPitch(newTtsPitch);
        setOfflineMaxTokens(newOfflineMaxTokens);
        setOfflineTopK(newOfflineTopK);
        setOfflineTemperature(newOfflineTemperature);
        setOfflineRandomSeed(newOfflineRandomSeed);
        setOfflineSupportAudio(newOfflineSupportAudio);
        setOfflineAudioRealtime(newOfflineAudioRealtime);
        setOfflineMaxNumImages(newOfflineMaxNumImages);
        setIsNoiseCancellationEnabled(newIsNoiseCancellationEnabled);
        setAudioGainValue(newAudioGainValue);
        setSelectedOcrModel(newSelectedOcrModel);
        setIsOcrAutoInitEnabled(newIsOcrAutoInitEnabled);
        
        localStorage.setItem('api-key', newApiKey);
        localStorage.setItem('model-name', newModelName);
        localStorage.setItem('online-provider', newOnlineProvider);
        localStorage.setItem('openai-api-url', newOpenaiApiUrl);
        localStorage.setItem('hf-api-key', newHfApiKey);
        localStorage.setItem('offline-model-name', newOfflineModel);
        localStorage.setItem('asr-model-id', newAsrModelId);
        localStorage.setItem('offline-mode-enabled', JSON.stringify(isOfflineEnabled));
        localStorage.setItem('is-offline-asr-enabled', JSON.stringify(newIsOfflineAsrEnabled));
        localStorage.setItem('is-realtime-asr-enabled', JSON.stringify(newIsRealtimeAsrEnabled));
        localStorage.setItem('is-web-speech-api-enabled', JSON.stringify(newIsWebSpeechApiEnabled));
        localStorage.setItem('is-two-step-jp-cn-enabled', JSON.stringify(newIsTwoStepJpCn));
        localStorage.setItem('tts-enabled', JSON.stringify(newIsTtsEnabled));
        localStorage.setItem('tts-voice-uri', newTtsVoiceURI);
        localStorage.setItem('tts-rate', JSON.stringify(newTtsRate));
        localStorage.setItem('tts-pitch', JSON.stringify(newTtsPitch));
        localStorage.setItem('offline-max-tokens', JSON.stringify(newOfflineMaxTokens));
        localStorage.setItem('offline-top-k', JSON.stringify(newOfflineTopK));
        localStorage.setItem('offline-temperature', JSON.stringify(newOfflineTemperature));
        localStorage.setItem('offline-random-seed', JSON.stringify(newOfflineRandomSeed));
        localStorage.setItem('offline-support-audio', JSON.stringify(newOfflineSupportAudio));
        localStorage.setItem('offline-audio-realtime', JSON.stringify(newOfflineAudioRealtime));
        localStorage.setItem('offline-max-num-images', JSON.stringify(newOfflineMaxNumImages));
        localStorage.setItem('is-noise-cancellation-enabled', JSON.stringify(newIsNoiseCancellationEnabled));
        localStorage.setItem('audio-gain-value', JSON.stringify(newAudioGainValue));
        localStorage.setItem('selected-ocr-model', newSelectedOcrModel);
        localStorage.setItem('is-ocr-auto-init-enabled', JSON.stringify(newIsOcrAutoInitEnabled));
        localStorage.setItem('asr-engine', newAsrEngine);
        localStorage.setItem('asr-profile', newAsrProfile);
        localStorage.setItem('asr-beam-width', JSON.stringify(newAsrBeamWidth));

        if (newAsrEngine === 'nemotron' && asrWorkerRef.current) {
            if (newAsrProfile !== asrProfile || newAsrBeamWidth !== asrBeamWidth) {
                setIsAsrInitializing(true);
                asrWorkerRef.current.postMessage({ type: 'load', payload: { asrProfile: newAsrProfile, asrBeamWidth: newAsrBeamWidth } });
            }
        } else if (newAsrEngine === 'transformers' && asrWorkerRef.current) {
            if (newAsrModelId !== asrModelId) {
                // If transformers model changed, maybe we don't auto-load here, 
                // because downloading takes time and there is a "Download" button for it.
                // But wait, the user didn't ask for that. We'll just handle nemotron.
            }
        }
    };

    const handleSelectHistory = (item: TranslationHistoryItem) => {
        setInputText(item.inputText);
        setTranslatedText(item.translatedText);
        setSourceLang(item.sourceLang);
        setTargetLang(item.targetLang);
        setIsHistoryOpen(false);
    };

    const handleClearHistory = () => {
        setHistory([]);
        localStorage.removeItem('translation-history');
    };

    const updateProgress = useCallback((modelName: string, progress: DownloadProgress) => {
        setDownloadProgress(prev => ({ ...prev, [modelName]: progress }));
    }, []);

    const handleStartDownload = useCallback((modelName: string, url: string, isTSModel?: boolean, dtype?: string | Record<string, string>) => {
        if (isTSModel && dtype) {
            downloadManager.startTSModelDownload(modelName, dtype, huggingFaceApiKey, (p) => updateProgress(modelName, p))
                .catch((err: any) => {
                    const message = err?.message || 'Download failed';
                    showNotification(message, 'error');
                });
        } else {
            downloadManager.startDownload(modelName, url, huggingFaceApiKey, (p) => updateProgress(modelName, p));
        }
    }, [huggingFaceApiKey, updateProgress, showNotification]);

    const handleResumeDownload = useCallback((modelName: string, url: string, isTSModel?: boolean, dtype?: string | Record<string, string>) => {
        if (isTSModel && dtype) {
             downloadManager.startTSModelDownload(modelName, dtype, huggingFaceApiKey, (p) => updateProgress(modelName, p))
                .catch((err: any) => {
                    const message = err?.message || 'Download failed';
                    showNotification(message, 'error');
                });
        } else {
             downloadManager.resumeDownload(modelName, url, huggingFaceApiKey, (p) => updateProgress(modelName, p));
        }
    }, [huggingFaceApiKey, updateProgress, showNotification]);
    
    const handlePauseDownload = useCallback((modelName: string) => {
        downloadManager.pauseDownload(modelName);
    }, []);

    const handleDeleteModel = useCallback(async (modelName: string) => {
        await downloadManager.deleteModel(modelName);
        updateProgress(modelName, { downloaded: 0, total: 0, percent: 0, status: 'not_started' });
    }, [updateProgress]);

    const handleDownloadAsrModel = useCallback(async (modelId: string) => {
        if (isAsrInitializing || !asrWorkerRef.current) return;
        
        setIsAsrInitializing(true);
        setAsrLoadingProgress({ file: '', progress: 0 });

        if (modelId === 'nemotron') {
            asrWorkerRef.current.postMessage({
                type: 'load',
                payload: { asrProfile, asrBeamWidth }
            });
            return;
        }

        if (modelId === 'qwen3') {
            asrWorkerRef.current.postMessage({
                type: 'load',
                payload: {}
            });
            return;
        }

        const model = ASR_MODELS.find(m => m.id === modelId);
        if (!model) {
            showNotification(`ASR model ${modelId} not found.`, 'error');
            setIsAsrInitializing(false);
            return;
        }
        
        asrWorkerRef.current.postMessage({
            type: 'load',
            payload: { modelId: model.id, quantization: model.quantization }
        });
    }, [isAsrInitializing, showNotification, asrProfile, asrBeamWidth]);

    const handleClearAsrCache = useCallback(async () => {
        try {
            if (asrWorkerRef.current) {
                asrWorkerRef.current.terminate();
                asrWorkerRef.current = null;
            }
            setIsAsrInitialized(false);
            await clearAsrCache();
            checkAllAsrCacheStatus().then(setAsrModelsCacheStatus);
            showNotification(t('notifications.asrModelDeleted'), 'success');
        } catch (err) {
            const message = err instanceof Error ? err.message : 'Unknown error';
            showNotification(message, 'error');
        }
    }, [checkAllAsrCacheStatus, showNotification, t]);

    return (
        <div className="bg-slate-100 h-full flex flex-col">
             <div className="w-full h-full max-w-6xl mx-auto flex-grow flex flex-col landscape:flex-row p-2 landscape:p-4 gap-4">
                <main className="flex-1 landscape:h-full flex flex-col min-h-0 min-w-0">
                     <TranslationOutput
                        translatedText={translatedText}
                        targetLang={targetLang}
                        setTargetLang={setTargetLang}
                        isLoading={isLoading}
                        onSpeak={handleSpeak}
                        onExpandText={() => setIsExpandedTextOpen(true)}
                        onOpenHistory={() => setIsHistoryOpen(true)}
                        onClearText={() => setTranslatedText('')}
                        isOfflineModeEnabled={isOfflineModeEnabled}
                        isOfflineModelInitializing={isOfflineModelInitializing}
                        isOfflineModelReady={isOfflineModelReady}
                        offlineModelName={offlineModelName}
                        isSpeaking={isSpeaking}
                        speakingGender={speakingGender}
                        onlineProvider={onlineProvider}
                        isOfflineTtsEnabled={isOfflineTtsEnabled}
                        isAstRecording={isAstRecording}
                        onToggleAstRecording={handleToggleAstRecording}
                        recordingCountdown={recordingCountdown}
                     />
                </main>
    
                <div className="flex-1 landscape:h-full flex flex-col min-h-0 min-w-0">
                    <TranslationInput
                        inputText={inputText}
                        setInputText={setInputText}
                        sourceLang={sourceLang}
                        setSourceLang={setSourceLang}
                        isLoading={isLoading || isOfflineModelInitializing || isAsrInitializing || ocrEngineStatus === 'initializing' || isTranscribing}
                        onTranslate={handleTranslate}
                        onCancel={handleCancelTranslation}
                        isRecording={isRecording}
                        onToggleRecording={handleToggleRecording}
                        onOpenCamera={() => setIsCameraOpen(true)}
                        onOpenSettings={() => setIsSettingsOpen(true)}
                        onSwapLanguages={handleSwapLanguages}
                        isOnline={isOnline}
                        isOfflineModeEnabled={isOfflineModeEnabled}
                        isOfflineModelReady={isOfflineModelReady}
                        recordingCountdown={recordingCountdown}
                    />
                </div>
            </div>
            
            {notification && (
                <div 
                    className={`fixed top-5 left-1/2 -translate-x-1/2 px-4 py-3 rounded z-20 shadow-lg text-white ${
                        notification.type === 'error' ? 'bg-red-500' : notification.type === 'success' ? 'bg-green-500' : 'bg-blue-500'
                    }`} 
                    role="alert"
                >
                    {notification.message}
                </div>
            )}

            {isCameraOpen && <CameraView onClose={() => setIsCameraOpen(false)} onImageCaptured={handleImageCaptured} imageFormat={(!isOfflineModeEnabled && onlineProvider === 'openai') ? 'image/jpeg' : 'image/webp'} />}
            
            {isSettingsOpen && (
                <SettingsModal 
                    isOpen={isSettingsOpen}
                    onClose={() => setIsSettingsOpen(false)}
                    onSave={handleSaveSettings}
                    currentApiKey={apiKey}
                    currentModelName={modelName}
                    currentOnlineProvider={onlineProvider}
                    currentOpenaiApiUrl={openaiApiUrl}
                    currentHuggingFaceApiKey={huggingFaceApiKey}
                    currentOfflineModelName={offlineModelName}
                    currentIsOfflineModeEnabled={isOfflineModeEnabled}
                    currentIsTwoStepJpCnEnabled={isTwoStepJpCn}
                    downloadProgress={downloadProgress}
                    onStartDownload={handleStartDownload}
                    onResumeDownload={handleResumeDownload}
                    onPauseDownload={handlePauseDownload}
                    onDeleteModel={handleDeleteModel}
                    isOfflineModelInitializing={isOfflineModelInitializing}
                    voices={voices}
                    targetLang={targetLang}
                    currentIsOfflineTtsEnabled={isOfflineTtsEnabled}
                    currentOfflineTtsVoiceURI={offlineTtsVoiceURI}
                    currentOfflineTtsRate={offlineTtsRate}
                    currentOfflineTtsPitch={offlineTtsPitch}
                    currentOfflineMaxTokens={offlineMaxTokens}
                    currentOfflineTopK={offlineTopK}
                    currentOfflineTemperature={offlineTemperature}
                    currentOfflineRandomSeed={offlineRandomSeed}
                    currentOfflineSupportAudio={offlineSupportAudio}
                    currentOfflineAudioRealtime={offlineAudioRealtime}
                    currentOfflineMaxNumImages={offlineMaxNumImages}
                    // ASR Props
                    currentIsOfflineAsrEnabled={isOfflineAsrEnabled}
                    currentIsRealtimeAsrEnabled={isRealtimeAsrEnabled}
                    currentIsWebSpeechApiEnabled={isWebSpeechApiEnabled}
                    currentAsrModelId={asrModelId}
                    currentAsrEngine={asrEngine}
                    currentAsrProfile={asrProfile}
                    currentAsrBeamWidth={asrBeamWidth}
                    currentIsNoiseCancellationEnabled={isNoiseCancellationEnabled}
                    currentAudioGainValue={audioGainValue}
                    asrModelsCacheStatus={asrModelsCacheStatus}
                    isAsrInitializing={isAsrInitializing}
                    asrLoadingProgress={asrLoadingProgress}
                    onDownloadAsrModel={handleDownloadAsrModel}
                    onClearAsrCache={handleClearAsrCache}
                    // OCR Props
                    ocrEngineStatus={ocrEngineStatus}
                    ocrEngineError={ocrEngineError}
                    onInitializeOcr={initializeOcr}
                    onClearOcrModel={async (modelKey) => {
                        await deleteOcrModelCache(modelKey);
                        showNotification(t('notifications.modelDeleted', { defaultValue: 'Model deleted successfully' }), 'success');
                    }}
                    onOcrModelChange={(model) => {
                        setSelectedOcrModel(model);
                        localStorage.setItem('selected-ocr-model', model);
                    }}
                    currentSelectedOcrModel={selectedOcrModel}
                    currentIsOcrAutoInitEnabled={isOcrAutoInitEnabled}
                    onClearSettings={() => {
                        setSourceLang(LANGUAGES[0]);
                        setTargetLang(LANGUAGES[6]);
                        localStorage.removeItem('source-lang');
                        localStorage.removeItem('target-lang');
                    }}
                />
            )}

            <HistoryModal
                isOpen={isHistoryOpen}
                onClose={() => setIsHistoryOpen(false)}
                history={history}
                onSelectHistory={handleSelectHistory}
                onClearHistory={handleClearHistory}
            />

            <ExpandedTextModal
                isOpen={isExpandedTextOpen}
                onClose={() => setIsExpandedTextOpen(false)}
                text={translatedText}
            />
        </div>
    );
};

export default App;
