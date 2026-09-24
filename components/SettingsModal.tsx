
import React, { useState, useEffect, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { XIcon, TrashIcon } from './icons';
import { DownloadProgress } from '../services/downloadManager';
import { OFFLINE_MODELS, OFFLINE_MODELS_TS, ASR_MODELS, ASR_MODELS_RUNTIME, OCR_MODELS, LANGUAGES } from '../constants';
import { filterVoicesForLanguage, testVoice, stopVoiceTest, normalizeWebSpeechLang } from '../utils/speechUtils';
import type { Language, OcrEngineStatus, OcrModelConfig, AsrEngineType, NemotronProfile, NemotronBeamWidth } from '../types';

interface SettingsModalProps {
    isOpen: boolean;
    onClose: () => void;
    onSave: (
        apiKey: string, 
        modelName: string, 
        huggingFaceApiKey: string, 
        offlineModelName: string, 
        asrModelId: string,
        isOfflineEnabled: boolean,
        isOfflineAsrEnabled: boolean,
        isRealtimeAsrEnabled: boolean,
        isWebSpeechApiEnabled: boolean,
        onlineProvider: string,
        openaiApiUrl: string,
        isOfflineTtsEnabled: boolean,
        offlineTtsVoiceURI: string,
        offlineTtsRate: number,
        offlineTtsPitch: number,
        isTwoStepJpCnEnabled: boolean,
        offlineMaxTokens: number,
        offlineTopK: number,
        offlineTemperature: number,
        offlineRandomSeed: number,
        offlineSupportAudio: boolean,
        offlineAudioRealtime: boolean,
        offlineMaxNumImages: number,
        isNoiseCancellationEnabled: boolean,
        audioGainValue: number,
        selectedOcrModel: keyof typeof OCR_MODELS,
        isOcrAutoInitEnabled: boolean,
        newAsrEngine: AsrEngineType,
        newAsrProfile: NemotronProfile,
        newAsrBeamWidth: NemotronBeamWidth
    ) => void;
    currentApiKey: string;
    currentModelName: string;
    currentOnlineProvider: string;
    currentOpenaiApiUrl: string;
    currentHuggingFaceApiKey: string;
    currentOfflineModelName: string;
    currentIsOfflineModeEnabled: boolean;
    currentIsTwoStepJpCnEnabled: boolean;
    downloadProgress: Record<string, DownloadProgress>;
    onStartDownload: (modelName: string, url: string, isTSModel?: boolean, dtype?: string | Record<string, string>) => void;
    onResumeDownload: (modelName: string, url: string, isTSModel?: boolean, dtype?: string | Record<string, string>) => void;
    onPauseDownload: (modelName: string) => void;
    onDeleteModel: (modelName: string) => void;
    isOfflineModelInitializing: boolean;
    voices: SpeechSynthesisVoice[];
    targetLang: Language;
    currentIsOfflineTtsEnabled: boolean;
    currentOfflineTtsVoiceURI: string;
    currentOfflineTtsRate: number;
    currentOfflineTtsPitch: number;
    currentOfflineMaxTokens: number;
    currentOfflineTopK: number;
    currentOfflineTemperature: number;
    currentOfflineRandomSeed: number;
    currentOfflineSupportAudio: boolean;
    currentOfflineAudioRealtime: boolean;
    currentOfflineMaxNumImages: number;
    // ASR Props
    currentIsOfflineAsrEnabled: boolean;
    currentIsRealtimeAsrEnabled: boolean;
    currentIsWebSpeechApiEnabled: boolean;
    currentAsrModelId: string;
    currentAsrEngine: AsrEngineType;
    currentAsrProfile: NemotronProfile;
    currentAsrBeamWidth: NemotronBeamWidth;
    currentIsNoiseCancellationEnabled: boolean;
    currentAudioGainValue: number;
    asrModelsCacheStatus: Record<string, boolean>;
    isAsrInitializing: boolean;
    asrLoadingProgress: { file: string; progress: number };
    onDownloadAsrModel: (modelId: string) => void;
    onClearAsrCache: () => void;
    // OCR Props
    ocrEngineStatus: OcrEngineStatus;
    ocrEngineError: string | null;
    onInitializeOcr: (modelConfig: OcrModelConfig) => Promise<void>;
    onOcrModelChange: (model: keyof typeof OCR_MODELS) => void;
    onClearOcrModel: (model: keyof typeof OCR_MODELS) => void;
    currentSelectedOcrModel: keyof typeof OCR_MODELS;
    currentIsOcrAutoInitEnabled: boolean;
    onClearSettings: () => void;
}

const SettingsModal: React.FC<SettingsModalProps> = ({ 
    isOpen, 
    onClose, 
    onSave, 
    currentApiKey,
    currentModelName,
    currentOnlineProvider,
    currentOpenaiApiUrl,
    currentHuggingFaceApiKey,
    currentOfflineModelName,
    currentIsOfflineModeEnabled,
    currentIsTwoStepJpCnEnabled,
    downloadProgress,
    onStartDownload,
    onResumeDownload,
    onPauseDownload,
    onDeleteModel,
    isOfflineModelInitializing,
    voices,
    targetLang,
    currentIsOfflineTtsEnabled,
    currentOfflineTtsVoiceURI,
    currentOfflineTtsRate,
    currentOfflineTtsPitch,
    currentOfflineMaxTokens,
    currentOfflineTopK,
    currentOfflineTemperature,
    currentOfflineRandomSeed,
    currentOfflineSupportAudio,
    currentOfflineAudioRealtime,
    currentOfflineMaxNumImages,
    currentIsOfflineAsrEnabled,
    currentIsRealtimeAsrEnabled,
    currentIsWebSpeechApiEnabled,
    currentAsrModelId,
    currentAsrEngine,
    currentAsrProfile,
    currentAsrBeamWidth,
    currentIsNoiseCancellationEnabled,
    currentAudioGainValue,
    asrModelsCacheStatus,
    isAsrInitializing,
    asrLoadingProgress,
    onDownloadAsrModel,
    onClearAsrCache,
    ocrEngineStatus,
    ocrEngineError,
    onInitializeOcr,
    onOcrModelChange,
    onClearOcrModel,
    currentSelectedOcrModel,
    currentIsOcrAutoInitEnabled,
    onClearSettings
}) => {
    const { t, i18n } = useTranslation();
    const [activeTab, setActiveTab] = useState('online');
    const [activeOfflineSubTab, setActiveOfflineSubTab] = useState('models');
    
    // Main Settings
    const [apiKey, setApiKey] = useState(currentApiKey);
    const [modelName, setModelName] = useState(currentModelName);
    const [onlineProvider, setOnlineProvider] = useState(currentOnlineProvider);
    const [openaiApiUrl, setOpenaiApiUrl] = useState(currentOpenaiApiUrl);
    const [huggingFaceApiKey, setHuggingFaceApiKey] = useState(currentHuggingFaceApiKey);
    const [offlineModelName, setOfflineModelName] = useState(currentOfflineModelName);
    const [isOfflineEnabled, setIsOfflineEnabled] = useState(currentIsOfflineModeEnabled);
    const [isTwoStepJpCnEnabled, setIsTwoStepJpCnEnabled] = useState(currentIsTwoStepJpCnEnabled);

    // ASR Settings
    const [isOfflineAsrEnabled, setIsOfflineAsrEnabled] = useState(currentIsOfflineAsrEnabled);
    const [isRealtimeAsrEnabled, setIsRealtimeAsrEnabled] = useState(currentIsRealtimeAsrEnabled);
    const [isWebSpeechApiEnabled, setIsWebSpeechApiEnabled] = useState(currentIsWebSpeechApiEnabled);
    const [asrModelId, setAsrModelId] = useState(currentAsrModelId);
    const [asrEngine, setAsrEngine] = useState(currentAsrEngine);
    const [asrProfile, setAsrProfile] = useState(currentAsrProfile);
    const [asrBeamWidth, setAsrBeamWidth] = useState(currentAsrBeamWidth);
    const [isNoiseCancellationEnabled, setIsNoiseCancellationEnabled] = useState(currentIsNoiseCancellationEnabled);
    const [audioGainValue, setAudioGainValue] = useState(currentAudioGainValue);

    // Offline TTS State
    const [isOfflineTtsEnabled, setIsOfflineTtsEnabled] = useState(currentIsOfflineTtsEnabled);
    const [offlineTtsVoiceURI, setOfflineTtsVoiceURI] = useState(currentOfflineTtsVoiceURI);
    const [offlineTtsRate, setOfflineTtsRate] = useState(currentOfflineTtsRate);
    const [offlineTtsPitch, setOfflineTtsPitch] = useState(currentOfflineTtsPitch);
    const [filteredVoices, setFilteredVoices] = useState<SpeechSynthesisVoice[]>([]);
    const [ttsLangCode, setTtsLangCode] = useState(targetLang?.code && targetLang.code !== 'auto' ? targetLang.code : 'zh-TW');
    const [isPlayingTestVoice, setIsPlayingTestVoice] = useState(false);

    // Offline Model Params State
    const [offlineMaxTokens, setOfflineMaxTokens] = useState(currentOfflineMaxTokens);
    const [offlineTopK, setOfflineTopK] = useState(currentOfflineTopK);
    const [offlineTemperature, setOfflineTemperature] = useState(currentOfflineTemperature);
    const [offlineRandomSeed, setOfflineRandomSeed] = useState(currentOfflineRandomSeed);
    const [offlineSupportAudio, setOfflineSupportAudio] = useState(currentOfflineSupportAudio);
    const [offlineAudioRealtime, setOfflineAudioRealtime] = useState(currentOfflineAudioRealtime);
    const [offlineMaxNumImages, setOfflineMaxNumImages] = useState(currentOfflineMaxNumImages);
    
    // OCR State
    const [selectedOcrModel, setSelectedOcrModel] = useState(currentSelectedOcrModel);
    const [isOcrAutoInitEnabled, setIsOcrAutoInitEnabled] = useState(currentIsOcrAutoInitEnabled);


    const prevIsOpenRef = useRef(false);

    useEffect(() => {
        if (isOpen && !prevIsOpenRef.current) {
            setApiKey(currentApiKey);
            setModelName(currentModelName);
            setOnlineProvider(currentOnlineProvider);
            setOpenaiApiUrl(currentOpenaiApiUrl);
            setHuggingFaceApiKey(currentHuggingFaceApiKey);
            setOfflineModelName(currentOfflineModelName);
            setIsOfflineEnabled(currentIsOfflineModeEnabled);
            setIsTwoStepJpCnEnabled(currentIsTwoStepJpCnEnabled);
            
            // TTS
            setIsOfflineTtsEnabled(currentIsOfflineTtsEnabled);
            setOfflineTtsVoiceURI(currentOfflineTtsVoiceURI);
            setOfflineTtsRate(currentOfflineTtsRate);
            setOfflineTtsPitch(currentOfflineTtsPitch);
            const initialLang = targetLang?.code && targetLang.code !== 'auto' ? targetLang.code : 'zh-TW';
            setTtsLangCode(initialLang);
            setIsPlayingTestVoice(false);
            
            // Params
            setOfflineMaxTokens(currentOfflineMaxTokens);
            setOfflineTopK(currentOfflineTopK);
            setOfflineTemperature(currentOfflineTemperature);
            setOfflineRandomSeed(currentOfflineRandomSeed);
            setOfflineSupportAudio(currentOfflineSupportAudio);
            setOfflineAudioRealtime(currentOfflineAudioRealtime);
            setOfflineMaxNumImages(currentOfflineMaxNumImages);

            // ASR
            setIsOfflineAsrEnabled(currentIsOfflineAsrEnabled);
            setIsRealtimeAsrEnabled(currentIsRealtimeAsrEnabled);
            setIsWebSpeechApiEnabled(currentIsWebSpeechApiEnabled);
            setAsrModelId(currentAsrModelId);
            setAsrEngine(currentAsrEngine);
            setAsrProfile(currentAsrProfile);
            setAsrBeamWidth(currentAsrBeamWidth);
            setIsNoiseCancellationEnabled(currentIsNoiseCancellationEnabled);
            setAudioGainValue(currentAudioGainValue);
            
            // OCR
            setSelectedOcrModel(currentSelectedOcrModel);
            setIsOcrAutoInitEnabled(currentIsOcrAutoInitEnabled);
        } else if (!isOpen && prevIsOpenRef.current) {
            stopVoiceTest();
            setIsPlayingTestVoice(false);
        }
        prevIsOpenRef.current = isOpen;
    }, [isOpen]);
    
    // Standardized Google Web Speech / BCP 47 voice matching
    useEffect(() => {
        if (isOpen && voices.length > 0) {
            const activeLang = ttsLangCode || (targetLang?.code && targetLang.code !== 'auto' ? targetLang.code : 'zh-TW');
            const matchedVoices = filterVoicesForLanguage(voices, activeLang);
            setFilteredVoices(matchedVoices);
            
            if (matchedVoices.length > 0) {
                // If current selected voice is valid in this filtered list, keep it; otherwise set to highest-scored voice
                if (!offlineTtsVoiceURI || !matchedVoices.some(v => v.voiceURI === offlineTtsVoiceURI)) {
                    setOfflineTtsVoiceURI(matchedVoices[0].voiceURI);
                }
            }
        }
    }, [isOpen, voices, ttsLangCode, targetLang]);

    if (!isOpen) return null;

    const handleClose = () => {
        stopVoiceTest();
        setIsPlayingTestVoice(false);
        onClose();
    };
    
    const handleSave = () => {
        stopVoiceTest();
        setIsPlayingTestVoice(false);
        onSave(
            apiKey, modelName, huggingFaceApiKey, offlineModelName, asrModelId, isOfflineEnabled, isOfflineAsrEnabled, isRealtimeAsrEnabled, isWebSpeechApiEnabled, onlineProvider, openaiApiUrl,
            isOfflineTtsEnabled, offlineTtsVoiceURI, offlineTtsRate, offlineTtsPitch, isTwoStepJpCnEnabled,
            offlineMaxTokens, offlineTopK, offlineTemperature, offlineRandomSeed, offlineSupportAudio, offlineAudioRealtime, offlineMaxNumImages,
            isNoiseCancellationEnabled, audioGainValue, selectedOcrModel, isOcrAutoInitEnabled, asrEngine, asrProfile, asrBeamWidth
        );
        onClose();
    };

    const handleClear = () => {
        setApiKey('');
        setModelName('gemini-3.5-flash-lite');
        setOnlineProvider('gemini');
        setOpenaiApiUrl('');
        setHuggingFaceApiKey('');
        setOfflineModelName('');
        setIsOfflineEnabled(false);
        setIsTwoStepJpCnEnabled(false);
        setIsOfflineTtsEnabled(false);
        setOfflineTtsVoiceURI('');
        setOfflineTtsRate(1);
        setOfflineTtsPitch(1);
        setOfflineMaxTokens(2048);
        setOfflineTopK(40);
        setOfflineTemperature(0.3);
        setOfflineRandomSeed(1);
        setOfflineSupportAudio(false);
        setOfflineAudioRealtime(false);
        setOfflineMaxNumImages(0);
        setSelectedOcrModel('PP_v6_small');
        setIsOcrAutoInitEnabled(false);
        
        // Clear ASR
        setIsOfflineAsrEnabled(false);
        setIsRealtimeAsrEnabled(false);
        setIsWebSpeechApiEnabled(true);
        setAsrModelId(ASR_MODELS[0].id);
        setAsrEngine('whisper');
        setAsrProfile('NORMAL');
        setAsrBeamWidth(1);
        setIsNoiseCancellationEnabled(false);
        setAudioGainValue(1.0);
        onClearAsrCache();

        OFFLINE_MODELS.forEach(model => model.value && onDeleteModel(model.value));
        onClearSettings();
        onSave('', 'gemini-3.5-flash-lite', '', '', ASR_MODELS[0].id, false, false, false, true, 'gemini', '', false, '', 1, 1, false, 2048, 40, 0.3, 1, false, false, 0, false, 0, 'PP_v6_small', false, 'whisper', 'NORMAL', 1);
    };

    const handleDownloadedModelSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
        setOfflineModelName(e.target.value);
    };

    const handleAsrModelSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
        setAsrModelId(e.target.value);
    };
    
    const handleLoadOcrModel = async () => {
        const modelConfig = {
            key: selectedOcrModel,
            ...OCR_MODELS[selectedOcrModel].paths,
        };
        await onInitializeOcr(modelConfig);
        onOcrModelChange(selectedOcrModel);
    };

    const TabButton: React.FC<{tabName: string; label: string}> = ({tabName, label}) => (
        <button
           onClick={() => setActiveTab(tabName)}
           className={`px-4 py-2 text-sm font-medium rounded-md focus:outline-none ${
               activeTab === tabName
                   ? 'bg-blue-100 text-blue-700'
                   : 'text-gray-500 hover:text-gray-700'
           }`}
           aria-selected={activeTab === tabName}
           role="tab"
       >
           {label}
       </button>
   );

   const SubTabButton: React.FC<{subTabName: string; label: string}> = ({subTabName, label}) => (
    <button
       onClick={() => setActiveOfflineSubTab(subTabName)}
       className={`w-full px-4 py-2 text-sm font-medium rounded-md focus:outline-none transition-colors ${
           activeOfflineSubTab === subTabName
               ? 'bg-white text-blue-700 shadow-sm'
               : 'text-gray-500 hover:bg-gray-200'
       }`}
       aria-selected={activeOfflineSubTab === subTabName}
       role="tab"
   >
       {label}
   </button>
);
   
    const formatBytes = (bytes: number, decimals = 2) => {
        if (!bytes || bytes === 0) return '0 Bytes';
        const k = 1024;
        const dm = decimals < 0 ? 0 : decimals;
        const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
    }
    
    const renderDownloadControls = (model: { name: string, value: string, url?: string, dtype?: string }) => {
        const progress = downloadProgress[model.value] || { status: 'not_started', percent: 0, downloaded: 0, total: 0 };
        const isInitializingThisModel = isOfflineModelInitializing && offlineModelName === model.value;

        const ActionButton: React.FC<{ onClick: () => void, text: string, className?: string, disabled?: boolean }> = ({ onClick, text, className = 'bg-blue-500 hover:bg-blue-600', disabled }) => (
            <button
                onClick={onClick}
                disabled={disabled}
                className={`text-white text-xs font-semibold py-1 px-3 rounded-md transition-colors disabled:bg-gray-300 disabled:cursor-not-allowed ${className}`}
            >
                {text}
            </button>
        );

        const isProcessing = isInitializingThisModel || progress.status === 'consolidating';
        const statusKey = isProcessing ? progress.status : progress.status;
        const statusText = t(`common.status.${statusKey}`);
        const isTSModel = !model.url;

        return (
             <div className="flex flex-col space-y-2 p-3 border border-gray-200 rounded-lg bg-gray-50">
                <div className="flex justify-between items-center">
                    <div className="flex items-center">
                        <input
                            type="radio"
                            id={`model-${model.value}`}
                            name="offline-model-selection"
                            value={model.value}
                            checked={offlineModelName === model.value}
                            onChange={handleDownloadedModelSelect}
                            disabled={progress.status !== 'completed' || isProcessing}
                            className="h-4 w-4 text-blue-600 border-gray-300 focus:ring-blue-500 disabled:cursor-not-allowed"
                        />
                        <label htmlFor={`model-${model.value}`} className="ml-2 text-sm font-medium text-gray-800">
                            {model.name} {isTSModel && <span className="text-xs text-blue-500"></span>}
                        </label>
                    </div>
                     <div className="flex items-center space-x-2">
                         {progress.status === 'not_started' && <ActionButton onClick={() => onStartDownload(model.value, model.url || '', isTSModel, model.dtype)} text={t('settings.download')} />}
                         {progress.status === 'downloading' && <ActionButton onClick={() => onPauseDownload(model.value)} text={t('settings.pause')} className="bg-yellow-500 hover:bg-yellow-600" />}
                         {progress.status === 'paused' && <ActionButton onClick={() => onResumeDownload(model.value, model.url || '', isTSModel, model.dtype)} text={t('settings.resume')} />}
                         {progress.status === 'error' && <ActionButton onClick={() => onResumeDownload(model.value, model.url || '', isTSModel, model.dtype)} text={t('settings.retry')} />}
                         {(progress.status !== 'not_started' && progress.status !== 'downloading' && progress.status !== 'consolidating') && (
                            <button onClick={() => onDeleteModel(model.value)} className="p-1.5 text-gray-500 hover:text-red-600 hover:bg-red-50 rounded-full" aria-label={t('settings.deleteModelAriaLabel', { modelName: model.name })}>
                                <TrashIcon className="w-4 h-4" />
                            </button>
                        )}
                    </div>
                </div>
                 {(progress.status !== 'not_started' || isProcessing) && (
                     <div>
                         <div className="w-full bg-gray-200 rounded-full h-2 relative overflow-hidden">
                             <div className={`h-2 rounded-full ${isProcessing ? 'bg-green-400 w-full animate-pulse' : 'bg-blue-500'}`} style={{ width: `${progress.percent}%` }}></div>
                         </div>
                         <div className="text-xs text-gray-500 mt-1 flex justify-between">
                             <span>{statusText}</span>
                             {progress.status === 'downloading' && !isProcessing && <span>{formatBytes(progress.downloaded)} / {formatBytes(progress.total)} ({Math.round(progress.percent)}%)</span>}
                         </div>
                     </div>
                 )}
                 {progress.status === 'error' && <p className="text-xs text-red-600 mt-1">{t('settings.statusError', { error: progress.error })}</p>}
             </div>
        );
    }

    const Slider: React.FC<{id: string, label: string, value: number, onChange: (e: React.ChangeEvent<HTMLInputElement>) => void, min: number, max: number, step: number}> = ({ id, label, value, onChange, min, max, step }) => (
        <div>
            <label htmlFor={id} className="flex justify-between items-center text-sm font-medium text-gray-700 mb-1">
                <span>{label}</span>
                <span className="text-gray-500 font-normal">{value}</span>
            </label>
            <input
                type="range"
                id={id}
                min={min}
                max={max}
                step={step}
                value={value}
                onChange={onChange}
                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
            />
        </div>
    );

    const ToggleSwitch: React.FC<{id: string, isEnabled: boolean, setIsEnabled: (enabled: boolean) => void, title: string, description: string, disabled?: boolean}> = ({ id, isEnabled, setIsEnabled, title, description, disabled }) => (
        <div className="flex items-center justify-between">
            <label htmlFor={id} className={`text-sm font-medium ${disabled ? 'text-gray-400' : 'text-gray-700'}`}>
                {title}
                <span className={`block text-xs ${disabled ? 'text-gray-300' : 'text-gray-500'}`}>{description}</span>
            </label>
            <button
                id={id}
                onClick={() => !disabled && setIsEnabled(!isEnabled)}
                disabled={disabled}
                className={`relative inline-flex items-center h-6 rounded-full w-11 transition-colors focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 ${isEnabled ? 'bg-blue-600' : 'bg-gray-200'} ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}
                role="switch"
                aria-checked={isEnabled}
            >
                <span className={`inline-block w-4 h-4 transform bg-white rounded-full transition-transform ${isEnabled ? 'translate-x-6' : 'translate-x-1'}`}/>
            </button>
        </div>
    );

    return (
        <div 
            className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center p-4"
            role="dialog"
            aria-modal="true"
            aria-labelledby="settings-title"
        >
            <div className="bg-white rounded-lg shadow-xl h-full w-full max-w-md overflow-auto">
                <div className="p-6 border-b border-gray-200">
                    <div className="flex justify-between items-center mb-4">
                        <h2 id="settings-title" className="text-xl font-semibold text-gray-800">{t('settings.title')}</h2>
                        <button onClick={handleClose} className="text-gray-400 hover:text-gray-600" aria-label={t('settings.closeAriaLabel')}>
                            <XIcon />
                        </button>
                    </div>
                    <div>
                        <label htmlFor="language-select" className="block text-sm font-medium text-gray-700 mb-1">{t('settings.languageLabel')}</label>
                        <select
                            id="language-select"
                            value={i18n.language}
                            onChange={(e) => i18n.changeLanguage(e.target.value)}
                            className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                        >
                            <option value="en">English</option>
                            <option value="zh-TW">繁體中文</option>
                            <option value="ja">日本語</option>
                        </select>
                    </div>
                </div>

                <div className="border-b border-gray-200">
                    <nav className="flex space-x-2 p-2 overflow-x-auto" role="tablist" aria-label="Settings tabs">
                       <TabButton tabName="online" label={t('settings.tabOnline')} />
                       <TabButton tabName="offline" label={t('settings.tabOffline')} />
                       <TabButton tabName="speech" label={t('settings.tabSpeech')} />
                       <TabButton tabName="tts" label={t('settings.tabTts')} />
                       <TabButton tabName="ocr" label={t('settings.tabOcr')} />
                    </nav>
                </div>

                <div className="p-6 space-y-6 min-h-[350px]">
                    {activeTab === 'online' && (
                        <div role="tabpanel" id="online-settings" aria-labelledby="online-tab" className="space-y-6">
                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-2">{t('settings.providerLabel')}</label>
                                <div className="flex space-x-4">
                                    <div className="flex items-center">
                                        <input id="provider-gemini" name="online-provider" type="radio" value="gemini" checked={onlineProvider === 'gemini'} onChange={(e) => setOnlineProvider(e.target.value)} className="h-4 w-4 text-blue-600 border-gray-300 focus:ring-blue-500" />
                                        <label htmlFor="provider-gemini" className="ml-2 block text-sm text-gray-900">{t('settings.providerGemini')}</label>
                                    </div>
                                    <div className="flex items-center">
                                        <input id="provider-openai" name="online-provider" type="radio" value="openai" checked={onlineProvider === 'openai'} onChange={(e) => setOnlineProvider(e.target.value)} className="h-4 w-4 text-blue-600 border-gray-300 focus:ring-blue-500" />
                                        <label htmlFor="provider-openai" className="ml-2 block text-sm text-gray-900">{t('settings.providerOpenAI')}</label>
                                    </div>
                                </div>
                            </div>
                            {onlineProvider === 'openai' && (
                                <div>
                                    <label htmlFor="openai-api-url" className="block text-sm font-medium text-gray-700 mb-1">
                                        {t('settings.openaiUrlLabel')}
                                    </label>
                                    <input
                                        type="text"
                                        id="openai-api-url"
                                        value={openaiApiUrl}
                                        onChange={(e) => setOpenaiApiUrl(e.target.value)}
                                        className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                                        placeholder={t('settings.openaiUrlPlaceholder')}
                                    />
                                </div>
                            )}
                            <div>
                                <label htmlFor="api-key" className="block text-sm font-medium text-gray-700 mb-1">
                                    {t('settings.apiKeyLabel')}
                                </label>
                                <input
                                    type="password"
                                    id="api-key"
                                    value={apiKey}
                                    onChange={(e) => setApiKey(e.target.value)}
                                    className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                                    placeholder={t('settings.apiKeyPlaceholder')}
                                />
                            </div>
                            <div>
                                <label htmlFor="model-name" className="block text-sm font-medium text-gray-700 mb-1">
                                    {t('settings.modelNameLabel')}
                                </label>
                                <input
                                    type="text"
                                    id="model-name"
                                    value={modelName}
                                    onChange={(e) => setModelName(e.target.value)}
                                    className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 bg-white"
                                    placeholder={onlineProvider === 'gemini' ? t('settings.modelNameGeminiPlaceholder') : t('settings.modelNameOpenAIPlaceholder')}
                                />
                                {onlineProvider === 'gemini' && (
                                    <div className="mt-2 flex flex-col gap-1.5">
                                        <div className="flex flex-wrap gap-2 items-center">
                                            <span className="text-xs text-gray-500">✨ 推薦文字/圖片模型：</span>
                                            <button 
                                                type="button"
                                                onClick={() => setModelName('gemini-3.5-flash-lite')}
                                                className="text-xs px-2 py-1 bg-blue-50 border border-blue-200 text-blue-700 rounded-md hover:bg-blue-100 transition whitespace-nowrap font-medium"
                                            >
                                                gemini-3.5-flash-lite
                                            </button>
                                        </div>
                                        <p className="text-[11px] text-gray-500 flex items-center gap-1 bg-indigo-50 border border-indigo-100 rounded p-1.5 mt-0.5">
                                            <span className="shrink-0 text-indigo-500">🚀</span>
                                            <span className="text-indigo-700 leading-normal">
                                                語音通話與麥克風按鈕已自動啟用 <b>gemini-3.5-live-translate</b> 雙向即時口譯，免去手動切換的繁瑣事宜！
                                            </span>
                                        </p>
                                    </div>
                                )}
                            </div>
                        </div>
                    )}
                    {activeTab === 'offline' && (
                        <div role="tabpanel" id="offline-settings" aria-labelledby="offline-tab" className="space-y-4">
                            <div className="flex space-x-1 p-1 bg-gray-100 rounded-lg">
                                <SubTabButton subTabName="models" label={t('settings.subTabModels')} />
                                <SubTabButton subTabName="params" label={t('settings.subTabParameters')} />
                            </div>
                            {activeOfflineSubTab === 'models' && (
                                <div className="space-y-6 pt-2">
                                    <div className="space-y-3">
                                        <label className="block text-sm font-medium text-gray-700">{t('settings.manageModelsLabel')}</label>
                                        <div className="space-y-4">
                                            <div>
                                                <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">MediaPipe Models</h4>
                                                <div className="space-y-2">
                                                    {OFFLINE_MODELS.filter(m => m.value).map(model => (
                                                        <div key={model.value}>
                                                            {renderDownloadControls(model as any)}
                                                        </div>
                                                    ))}
                                                </div>
                                            </div>
                                            <div>
                                                <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">Transformers.js Models</h4>
                                                <div className="space-y-2">
                                                    {OFFLINE_MODELS_TS.map(model => (
                                                        <div key={model.value}>
                                                            {renderDownloadControls(model)}
                                                        </div>
                                                    ))}
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                    <div className="space-y-4 pt-2">
                                        <ToggleSwitch 
                                            id="offline-toggle"
                                            isEnabled={isOfflineEnabled}
                                            setIsEnabled={setIsOfflineEnabled}
                                            title={t('settings.enableOfflineLabel')}
                                            description={t('settings.enableOfflineDescription')}
                                        />
                                        <ToggleSwitch 
                                            id="twostep-toggle"
                                            isEnabled={isTwoStepJpCnEnabled}
                                            setIsEnabled={setIsTwoStepJpCnEnabled}
                                            title={t('settings.enableTwoStepLabel')}
                                            description={t('settings.enableTwoStepDescription')}
                                        />
                                    </div>
                                </div>
                            )}
                            {activeOfflineSubTab === 'params' && (
                                <div className="space-y-6 pt-2">
                                    <p className="text-sm text-gray-500">{t('settings.offlineParamsDescription')}</p>
                                    <Slider id="max-tokens-slider" label={t('settings.maxTokensLabel')} value={offlineMaxTokens} onChange={e => setOfflineMaxTokens(parseInt(e.target.value, 10))} min={256} max={8192} step={256} />
                                    <Slider id="top-k-slider" label={t('settings.topKLabel')} value={offlineTopK} onChange={e => setOfflineTopK(parseInt(e.target.value, 10))} min={1} max={100} step={1} />
                                    <Slider id="temperature-slider" label={t('settings.temperatureLabel')} value={offlineTemperature} onChange={e => setOfflineTemperature(parseFloat(e.target.value))} min={0} max={1} step={0.05} />
                                    <div>
                                        <label htmlFor="random-seed" className="block text-sm font-medium text-gray-700 mb-1">{t('settings.randomSeedLabel')}</label>
                                        <input type="number" id="random-seed" value={offlineRandomSeed} onChange={e => setOfflineRandomSeed(parseInt(e.target.value, 10))} className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500" />
                                    </div>
                                    <ToggleSwitch id="max-num-images-toggle" isEnabled={offlineMaxNumImages === 1} setIsEnabled={(enabled) => setOfflineMaxNumImages(enabled ? 1 : 0)} title={t('settings.maxNumImagesLabel')} description={t('settings.maxNumImagesDescription')} />
                                    <ToggleSwitch 
                                        id="support-audio-toggle" 
                                        isEnabled={offlineSupportAudio} 
                                        setIsEnabled={(enabled) => {
                                            setOfflineSupportAudio(enabled);
                                            if (enabled) {
                                                setIsOfflineAsrEnabled(false);
                                            }
                                        }} 
                                        title={t('settings.enableGemmaAudioLabel')} 
                                        description={t('settings.enableGemmaAudioDescription')} 
                                        disabled={!isOfflineEnabled}
                                    />
                                    <ToggleSwitch 
                                        id="support-audio-realtime-toggle" 
                                        isEnabled={offlineAudioRealtime} 
                                        setIsEnabled={setOfflineAudioRealtime} 
                                        title={t('settings.enableGemmaAudioRealtimeLabel') || "Gemma 3N Audio Realtime"} 
                                        description={t('settings.enableGemmaAudioRealtimeDescription') || "Process audio in real-time using Gemma 3N."} 
                                        disabled={!offlineSupportAudio || !isOfflineEnabled}
                                    />
                                </div>
                            )}
                        </div>
                    )}
                    {activeTab === 'speech' && (
                         <div role="tabpanel" id="speech-settings" aria-labelledby="speech-tab" className="space-y-6">
                             <div className="space-y-4">
                                <label className="block text-sm font-medium -mb-2 text-gray-700">{t('settings.audioProcessingLabel')}</label>
                                 <ToggleSwitch
                                    id="noise-cancellation-toggle"
                                    isEnabled={isNoiseCancellationEnabled}
                                    setIsEnabled={setIsNoiseCancellationEnabled}
                                    title={t('settings.enableNoiseCancellationLabel')}
                                    description={t('settings.enableNoiseCancellationDescription')}
                                 />
                                 <div>
                                    <label htmlFor="gain-slider" className="flex justify-between items-center text-sm font-medium mb-1 text-gray-700">
                                        <span>{t('settings.audioGainLabel')}</span>
                                        <span className="font-normal text-gray-500">{audioGainValue.toFixed(1)}x</span>
                                    </label>
                                    <input
                                        type="range"
                                        id="gain-slider"
                                        min={0.5} max={5} step={0.1}
                                        value={audioGainValue}
                                        onChange={e => setAudioGainValue(parseFloat(e.target.value))}
                                        className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                                    />
                                 </div>
                            </div>
                            <div className="border-t border-gray-200"></div>

                            <div className="space-y-1">
                                <ToggleSwitch
                                    id="web-speech-toggle"
                                    isEnabled={isWebSpeechApiEnabled}
                                    setIsEnabled={(enabled) => {
                                        setIsWebSpeechApiEnabled(enabled);
                                    }}
                                    title={t('settings.enableWebSpeechLabel')}
                                    description={t('settings.enableWebSpeechDescription')}
                                />
                            </div>
                             <div className="border-t border-gray-200"></div>
                             
                             <ToggleSwitch
                                id="asr-toggle"
                                isEnabled={isOfflineAsrEnabled}
                                setIsEnabled={(enabled) => {
                                    setIsOfflineAsrEnabled(enabled);
                                    if (enabled) {
                                        setOfflineSupportAudio(false);
                                    }
                                }}
                                title={t('settings.enableOfflineAsrLabel')}
                                description={t('settings.enableOfflineAsrDescription')}
                             />
                             <ToggleSwitch
                                id="realtime-asr-toggle"
                                isEnabled={isRealtimeAsrEnabled}
                                setIsEnabled={setIsRealtimeAsrEnabled}
                                title={t('settings.enableRealtimeAsrLabel', '即時處理(realtime)')}
                                description={t('settings.enableRealtimeAsrDescription', '即時辨識並轉換成文字')}
                                disabled={!isOfflineAsrEnabled}
                             />
                            <div className={`space-y-4 transition-opacity ${!isOfflineAsrEnabled ? 'opacity-50' : ''}`}>
                                <div>
                                    <label className={`block text-sm font-medium ${!isOfflineAsrEnabled ? 'text-gray-400' : 'text-gray-700'}`}>ASR Engine</label>
                                    <select
                                        value={asrEngine}
                                        onChange={(e) => {
                                            const newEngine = e.target.value as AsrEngineType;
                                            setAsrEngine(newEngine);
                                            if (newEngine === 'nemotron') {
                                                setAsrModelId('nemotron');
                                            } else if (newEngine === 'qwen3') {
                                                setAsrModelId('qwen3');
                                            } else if (newEngine === 'whisper') {
                                                setAsrModelId(ASR_MODELS[0].id);
                                            }
                                        }}
                                        disabled={!isOfflineAsrEnabled}
                                        className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm rounded-md"
                                    >
                                        {ASR_MODELS_RUNTIME.map((m) => (
                                            <option key={m.id} value={m.engine}>
                                                {m.name}
                                            </option>
                                        ))}
                                    </select>
                                </div>
                                {asrEngine === 'whisper' && (
                                    <div className="space-y-3">
                                        <label className={`block text-sm font-medium ${!isOfflineAsrEnabled ? 'text-gray-400' : 'text-gray-700'}`}>{t('settings.asrModelLabel')}</label>
                                        {ASR_MODELS.map(model => {
                                            const isCached = asrModelsCacheStatus[model.id] || false;
                                            const isLoadingThisModel = isAsrInitializing && asrModelId === model.id;
                                            return (
                                                <div key={model.id} className="p-3 border border-gray-200 rounded-lg bg-gray-50 space-y-2">
                                                    <div className="flex items-center justify-between">
                                                        <div className="flex items-center">
                                                            <input
                                                                type="radio"
                                                                id={`asr-model-${model.id}`}
                                                                name="asr-model-selection"
                                                                value={model.id}
                                                                checked={asrModelId === model.id}
                                                                onChange={handleAsrModelSelect}
                                                                className="h-4 w-4 text-blue-600 border-gray-300 focus:ring-blue-500"
                                                                disabled={!isOfflineAsrEnabled}
                                                            />
                                                            <label htmlFor={`asr-model-${model.id}`} className={`ml-3 text-sm font-medium ${!isOfflineAsrEnabled ? 'text-gray-400' : 'text-gray-800'}`}>
                                                                {model.name} <span className="text-gray-500 font-normal">({model.size})</span>
                                                            </label>
                                                        </div>
                                                        <div className="flex items-center space-x-3">
                                                            {!isLoadingThisModel && (
                                                                isCached ? (
                                                                    <span className="text-sm font-medium text-green-600">{t('settings.modelCached') || 'Cached'}</span>
                                                                ) : (
                                                                    <button
                                                                        onClick={() => onDownloadAsrModel(model.id)}
                                                                        disabled={isAsrInitializing || !isOfflineAsrEnabled}
                                                                        className="text-sm font-medium text-blue-600 hover:text-blue-800 disabled:text-gray-400 disabled:cursor-not-allowed"
                                                                    >
                                                                        {t('settings.modelDownload') || 'Download'}
                                                                    </button>
                                                                )
                                                            )}
                                                        </div>
                                                    </div>
                                                    {isLoadingThisModel && (
                                                        <div className="pt-1 text-center text-sm text-blue-600">
                                                            {asrLoadingProgress.file} ({Math.round(asrLoadingProgress.progress)}%)
                                                        </div>
                                                    )}
                                                </div>
                                            );
                                        })}
                                    </div>
                                )}
                                {asrEngine === 'nemotron' && (
                                    <div className="space-y-4 p-4 border border-gray-200 rounded-lg bg-gray-50">
                                        <div className="flex items-center justify-between mb-2">
                                            <div className="text-sm font-medium text-gray-800">
                                                Nemotron-ASR Model <span className="text-gray-500 font-normal">(~863MB)</span>
                                            </div>
                                            <div className="flex items-center space-x-3">
                                                {!(isAsrInitializing && asrEngine === 'nemotron') && (
                                                    asrModelsCacheStatus['nemotron'] ? (
                                                        <span className="text-sm font-medium text-green-600">{t('settings.modelCached') || 'Cached'}</span>
                                                    ) : (
                                                        <button
                                                            onClick={() => onDownloadAsrModel('nemotron')}
                                                            disabled={isAsrInitializing || !isOfflineAsrEnabled}
                                                            className="text-sm font-medium text-blue-600 hover:text-blue-800 disabled:text-gray-400 disabled:cursor-not-allowed"
                                                        >
                                                            {t('settings.modelDownload') || 'Download'}
                                                        </button>
                                                    )
                                                )}
                                            </div>
                                        </div>
                                        {isAsrInitializing && asrEngine === 'nemotron' && (
                                            <div className="pt-1 text-center text-sm text-blue-600">
                                                {asrLoadingProgress.file} ({Math.round(asrLoadingProgress.progress)}%)
                                            </div>
                                        )}
                                        <div>
                                            <label className="block text-sm font-medium text-gray-700">Latency Profile (ms)</label>
                                            <select
                                                value={asrProfile}
                                                onChange={(e) => setAsrProfile(e.target.value as NemotronProfile)}
                                                disabled={!isOfflineAsrEnabled}
                                                className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm rounded-md"
                                            >
                                                <option value="TURBO">TURBO (80)</option>
                                                <option value="FAST">FAST (160)</option>
                                                <option value="BALANCED">BALANCED (320)</option>
                                                <option value="NORMAL">NORMAL (560)</option>
                                                <option value="HIGH">HIGH (1120)</option>
                                            </select>
                                        </div>
                                        <div>
                                            <label className="block text-sm font-medium text-gray-700">Beam Width</label>
                                            <select
                                                value={asrBeamWidth}
                                                onChange={(e) => setAsrBeamWidth(Number(e.target.value) as NemotronBeamWidth)}
                                                disabled={!isOfflineAsrEnabled}
                                                className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm rounded-md"
                                            >
                                                <option value="1">1 (Greedy)</option>
                                                <option value="2">2</option>
                                                <option value="3">3</option>
                                                <option value="4">4</option>
                                                <option value="5">5</option>
                                            </select>
                                        </div>
                                    </div>
                                )}
                                {asrEngine === 'qwen3' && (
                                    <div className="space-y-4 p-4 border border-gray-200 rounded-lg bg-gray-50">
                                        <div className="flex items-center justify-between mb-2">
                                            <div>
                                                <div className="text-sm font-medium text-gray-800">
                                                    Qwen3-ASR-0.6B-ONNX
                                                </div>
                                                <div className="text-xs text-gray-500">
                                                    Auto WebGPU shader-f16: q4f16 (~888MB) / q4 (~1.29GB)
                                                </div>
                                            </div>
                                            <div className="flex items-center space-x-3">
                                                {!(isAsrInitializing && asrEngine === 'qwen3') && (
                                                    asrModelsCacheStatus['qwen3'] ? (
                                                        <span className="text-sm font-medium text-green-600">{t('settings.modelCached') || 'Cached'}</span>
                                                    ) : (
                                                        <button
                                                            onClick={() => onDownloadAsrModel('qwen3')}
                                                            disabled={isAsrInitializing || !isOfflineAsrEnabled}
                                                            className="text-sm font-medium text-blue-600 hover:text-blue-800 disabled:text-gray-400 disabled:cursor-not-allowed"
                                                        >
                                                            {t('settings.modelDownload') || 'Download'}
                                                        </button>
                                                    )
                                                )}
                                            </div>
                                        </div>
                                        {isAsrInitializing && asrEngine === 'qwen3' && (
                                            <div className="space-y-1">
                                                <div className="pt-1 text-center text-sm text-blue-600 font-medium">
                                                    {asrLoadingProgress.file} ({Math.round(asrLoadingProgress.progress)}%)
                                                </div>
                                                <div className="w-full bg-gray-200 rounded-full h-2">
                                                    <div
                                                        className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                                                        style={{ width: `${Math.min(100, Math.max(0, asrLoadingProgress.progress))}%` }}
                                                    />
                                                </div>
                                            </div>
                                        )}
                                    </div>
                                )}
                                <div>
                                    <button
                                        onClick={onClearAsrCache}
                                        disabled={!isOfflineAsrEnabled}
                                        className="w-full flex items-center justify-center space-x-2 text-red-600 font-medium py-2 px-4 rounded-lg border border-red-200 hover:bg-red-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-red-500 transition-colors mt-2 disabled:opacity-50 disabled:cursor-not-allowed"
                                    >
                                        <TrashIcon className="h-5 w-5" />
                                        <span>{t('settings.clearAsrCacheButton')}</span>
                                    </button>
                                </div>
                            </div>
                        </div>
                    )}
                    {activeTab === 'tts' && (
                        <div role="tabpanel" id="tts-settings" aria-labelledby="tts-tab" className="space-y-6">
                             <ToggleSwitch
                                id="tts-toggle"
                                isEnabled={isOfflineTtsEnabled}
                                setIsEnabled={setIsOfflineTtsEnabled}
                                title={t('settings.enableCustomTtsLabel')}
                                description={t('settings.enableCustomTtsDescription')}
                             />

                             <div className="space-y-4">
                                <div>
                                    <label htmlFor="tts-lang-select" className="block text-sm font-medium text-gray-700 mb-1">
                                        {t('settings.ttsTargetLangLabel', 'TTS 語系篩選 (Voice Language Filter)')}
                                    </label>
                                    <select
                                        id="tts-lang-select"
                                        value={ttsLangCode}
                                        onChange={(e) => {
                                            const newLang = e.target.value;
                                            setTtsLangCode(newLang);
                                            const newVoices = filterVoicesForLanguage(voices, newLang);
                                            setFilteredVoices(newVoices);
                                            if (newVoices.length > 0) {
                                                setOfflineTtsVoiceURI(newVoices[0].voiceURI);
                                            }
                                        }}
                                        className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 bg-white text-sm"
                                    >
                                        {LANGUAGES.filter(l => l.code !== 'auto').map(lang => (
                                            <option key={lang.code} value={lang.code}>
                                                {t(lang.name)} ({normalizeWebSpeechLang(lang.code)})
                                            </option>
                                        ))}
                                    </select>
                                </div>

                                <div>
                                    <label htmlFor="voice-select" className="block text-sm font-medium text-gray-700 mb-1">
                                        {t('settings.voiceLabel')} ({filteredVoices.length})
                                    </label>
                                    <div className="flex gap-2 items-center">
                                        <select
                                            id="voice-select"
                                            value={offlineTtsVoiceURI}
                                            onChange={(e) => setOfflineTtsVoiceURI(e.target.value)}
                                            className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 disabled:bg-gray-100 text-sm"
                                            disabled={filteredVoices.length === 0}
                                        >
                                            {filteredVoices.length > 0 ? (
                                                filteredVoices.map(voice => (
                                                    <option key={voice.voiceURI} value={voice.voiceURI}>
                                                        {voice.name} ({voice.lang})
                                                    </option>
                                                ))
                                            ) : (
                                                <option value="">{t('settings.voicePlaceholder', { languageName: t(LANGUAGES.find(l => l.code === ttsLangCode)?.name || targetLang.name) })}</option>
                                            )}
                                        </select>
                                        <button
                                            type="button"
                                            onClick={() => {
                                                if (isPlayingTestVoice) {
                                                    stopVoiceTest();
                                                    setIsPlayingTestVoice(false);
                                                } else {
                                                    setIsPlayingTestVoice(true);
                                                    testVoice(
                                                        offlineTtsVoiceURI,
                                                        undefined,
                                                        offlineTtsRate,
                                                        offlineTtsPitch,
                                                        ttsLangCode,
                                                        () => setIsPlayingTestVoice(false),
                                                        () => setIsPlayingTestVoice(false)
                                                    );
                                                }
                                            }}
                                            disabled={!offlineTtsVoiceURI}
                                            className={`px-3 py-2 rounded-md text-xs font-medium whitespace-nowrap border transition-colors flex items-center gap-1 shrink-0 ${
                                                isPlayingTestVoice
                                                    ? 'bg-amber-100 text-amber-800 border-amber-300 hover:bg-amber-200'
                                                    : 'bg-blue-50 text-blue-700 border-blue-200 hover:bg-blue-100 disabled:opacity-50 disabled:cursor-not-allowed'
                                            }`}
                                        >
                                            {isPlayingTestVoice ? '⏹ Stop' : '▶ Play'}
                                        </button>
                                    </div>
                                </div>

                                <Slider id="rate-slider" label={t('settings.rateLabel')} value={offlineTtsRate} onChange={e => setOfflineTtsRate(parseFloat(e.target.value))} min={0.5} max={2} step={0.1} />
                                <Slider id="pitch-slider" label={t('settings.pitchLabel')} value={offlineTtsPitch} onChange={e => setOfflineTtsPitch(parseFloat(e.target.value))} min={0} max={2} step={0.1} />
                             </div>
                        </div>
                    )}
                    {activeTab === 'ocr' && (
                        <div role="tabpanel" id="ocr-settings" aria-labelledby="ocr-tab" className="space-y-6">
                            <ToggleSwitch
                                id="ocr-auto-init-toggle"
                                isEnabled={isOcrAutoInitEnabled}
                                setIsEnabled={setIsOcrAutoInitEnabled}
                                title={t('settings.enableOcrAutoInitLabel')}
                                description={t('settings.enableOcrAutoInitDescription')}
                            />
                            <div className="border-t pt-6 space-y-6">
                                <div>
                                    <label htmlFor="ocr-model-select" className="block text-sm font-medium text-gray-700 mb-1">{t('settings.ocrModelLabel')}</label>
                                    <div className="flex space-x-2">
                                        <select 
                                            id="ocr-model-select"
                                            value={selectedOcrModel} 
                                            onChange={(e) => setSelectedOcrModel(e.target.value as any)}
                                            className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                                        >
                                            {Object.entries(OCR_MODELS).map(([k, m]) => <option key={k} value={k}>{m.name} ({t(m.description)})</option>)}
                                        </select>
                                        <button 
                                            onClick={() => onClearOcrModel(selectedOcrModel)}
                                            className="p-2 text-red-500 hover:text-red-700 hover:bg-red-50 rounded-md transition-colors"
                                            title="Delete cached OCR model"
                                        >
                                            <TrashIcon className="w-5 h-5" />
                                        </button>
                                    </div>
                                </div>
                                <button 
                                    onClick={handleLoadOcrModel}
                                    disabled={ocrEngineStatus === 'initializing'}
                                    className="w-full bg-indigo-600 hover:bg-indigo-500 text-white py-2 rounded-lg text-sm font-medium transition disabled:bg-indigo-300 disabled:cursor-wait"
                                >
                                    {ocrEngineStatus === 'initializing' 
                                        ? t('common.status.initializing')
                                        : (ocrEngineStatus === 'ready' ? t('settings.switchOcr') : t('settings.initializeOcr'))
                                    }
                                </button>
                                {ocrEngineStatus === 'ready' && <p className="text-sm text-center text-green-600">OCR Engine Ready.</p>}
                                {ocrEngineStatus === 'error' && <p className="text-sm text-center text-red-600">OCR Engine failed to initialize: {ocrEngineError}</p>}
                            </div>
                        </div>
                    )}
                </div>
                <div className="p-6 bg-gray-50 rounded-b-lg flex justify-between items-center">
                     <button 
                        onClick={handleClear}
                        className="text-red-600 font-medium py-2 px-4 rounded-lg hover:bg-red-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-red-500 transition-colors"
                        aria-label={t('history.clearAriaLabel')}
                    >
                        {t('settings.clearSettingsButton')}
                    </button>
                    <button 
                        onClick={handleSave}
                        className="bg-blue-500 text-white font-bold py-2 px-4 rounded-lg hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 transition-colors"
                    >
                        {t('settings.saveSettingsButton')}
                    </button>
                </div>
            </div>
        </div>
    );
};

export default SettingsModal;
