
export interface Language {
    code: string;
    name: string;
    asrCode?: string;
}

export interface TranslationHistoryItem {
    id: number;
    inputText: string;
    translatedText: string;
    sourceLang: Language;
    targetLang: Language;
}

export interface AsrModel {
    id: string;
    name: string;
    quantization: Record<string, string>;
    size: string;
}

export interface AsrModelRuntime {
    id: AsrEngineType;
    modelId: string;
    engine: AsrEngineType;
    name: string;
    size: string;
    description: string;
}

export interface CustomOfflineModel {
    name: string;
    value: string;
    url?: string;
    size?: string;
}

// BoxType: [top-left, top-right, bottom-right, bottom-left] as [x, y]
export type BoxType = [[number, number], [number, number], [number, number], [number, number]];

export interface EsearchOCRItem {
    text: string;
    mean: number;
    box: BoxType;
    style: { bg: [number, number, number]; text: [number, number, number] };
}

export type ReadingDirPart = "lr" | "rl" | "tb" | "bt";

export interface EsearchOCROutput {
    src: EsearchOCRItem[];
    columns: {
        src: EsearchOCRItem[];
        outerBox: BoxType;
        parragraphs: {
            src: EsearchOCRItem[];
            parse: EsearchOCRItem;
        }[];
    }[];
    parragraphs: EsearchOCRItem[];
    readingDir: {
        inline: ReadingDirPart;
        block: ReadingDirPart;
    };
    angle: {
        reading: { inline: number; block: number };
        angle: number;
    };
}

export type OcrEngineStatus = 'uninitialized' | 'initializing' | 'ready' | 'error';

export type AsrEngineType = 'whisper' | 'nemotron' | 'qwen3';
export type NemotronProfile = 'TURBO' | 'FAST' | 'BALANCED' | 'NORMAL' | 'HIGH';
export type NemotronBeamWidth = 1 | 2 | 3 | 4 | 5;

export interface OcrModelConfig {
    key: string;
    detPath: string;
    recPath: string;
    dictPath?: string;
}
