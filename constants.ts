import type { Language, AsrModel, AsrModelRuntime } from './types';

export const LANGUAGES: Language[] = [
    { code: 'auto', name: 'languages.autoDetect' },
    { code: 'en-US', name: 'languages.english', asrCode: 'en' },
    { code: 'zh-TW', name: 'languages.chineseTraditional', asrCode: 'zh' },
    { code: 'zh-HK', name: 'languages.chineseTraditionalHK', asrCode: 'zh' },
    { code: 'zh-Hans', name: 'languages.chineseSimplified', asrCode: 'zh' },
    { code: 'es-ES', name: 'languages.spanish', asrCode: 'es' },
    { code: 'ja-JP', name: 'languages.japanese', asrCode: 'ja' },
    { code: 'fr-FR', name: 'languages.french', asrCode: 'fr' },
    { code: 'de-DE', name: 'languages.german', asrCode: 'de' },
    { code: 'ko-KR', name: 'languages.korean', asrCode: 'ko' },
    { code: 'ru-RU', name: 'languages.russian', asrCode: 'ru' },
    { code: 'it-IT', name: 'languages.italian', asrCode: 'it'  },
];

export const OFFLINE_MODELS = [
    { 
        name: 'Gemma-4-E2B-it (2.1 GB)', 
        value: 'gemma-4-E2B-it-web.litertlm', 
        url: 'https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/resolve/main/gemma-4-E2B-it-web.litertlm'
    },
	{ 
        name: 'Gemma-4-E4B-it (2.9 GB)', 
        value: 'gemma-4-E4B-it-web.litertlm', 
        url: 'https://huggingface.co/litert-community/gemma-4-E4B-it-litert-lm/resolve/main/gemma-4-E4B-it-web.litertlm'
    },
    { 
        name: 'Gemma-3n-E2B (3.04 GB)', 
        value: 'gemma-3n-E2B-it-int4-Web.litertlm', 
        url: 'https://huggingface.co/willopcbeta/Gemma-3n-Web/resolve/main/gemma-3n-E2B-it-int4-Web.litertlm'
    },
    { 
        name: 'Gemma-3n-E4B (4.28 GB)', 
        value: 'gemma-3n-E4B-it-int4-Web.litertlm', 
        url: 'https://huggingface.co/willopcbeta/Gemma-3n-Web/resolve/main/gemma-3n-E4B-it-int4-Web.litertlm'
    },
	{ 
        name: 'TranslateGemma-4B (3.62 GB)', 
        value: 'translategemma-4b-it-int8-web.task', 
        url: 'https://huggingface.co/willopcbeta/Gemma-3n-Web/resolve/main/translategemma-4b-it-int8-web.task'
    },
];

export const OFFLINE_MODELS_TS = [
  { 
        name: 'Gemma-4-E2B-it (3.8 GB)', 
        value: 'onnx-community/gemma-4-E2B-it-ONNX', 
        dtype: 'q4f16',
		generationMode: 'Gemma4ForConditionalGeneration'
    },
	{ 
		name: 'Gemma-4-E2B-it-qat (2.6 GB)',
		value: 'onnx-community/gemma-4-E2B-it-qat-mobile-ONNX',
		generationMode: 'Gemma4ForConditionalGeneration',
		dtype: {
			embed_tokens: "q2f16",
			vision_encoder: "fp16",
			decoder_model_merged: "q2f16",
			audio_encoder: "q2f16"
		},
  },
  { 
        name: 'Gemma-4-E4B-it (4.5 GB)', 
        value: 'onnx-community/gemma-4-E4B-it-ONNX', 
        dtype: 'q4',
		generationMode: 'Gemma4ForConditionalGeneration'
  },
  { 
        name: 'Gemma-4-E4B-it-qat (3.6 GB)', 
        value: 'onnx-community/gemma-4-E4B-it-qat-mobile-ONNX', 
		generationMode: 'Gemma4ForConditionalGeneration',
		dtype: {
			embed_tokens: "q2f16",
			vision_encoder: "fp16",
			decoder_model_merged: "q2f16",
			audio_encoder: "q2f16"
		},
    },
	{ 
        name: 'Qwen3.5-4B (3.11 GB)', 
        value: 'onnx-community/Qwen3.5-4B-ONNX-OPT', 
        dtype: 'q4f16',
        generationMode: 'Qwen3_5ForConditionalGeneration'
    },
];

export const ASR_MODELS: AsrModel[] = [
    {
        id: 'willopcbeta/lite-whisper-large-v3-turbo-ONNX',
        name: 'lite-whisper-large-v3-turbo',
        quantization: {
            encoder_model: 'q4',
            decoder_model_merged: 'q4',
        },
        size: '~590 MB'
    },
    {
        id: 'willopcbeta/unsloth-whisper-large-v3-turbo-ONNX',
        name: 'unsloth-whisper-large-v3-turbo',
        quantization: {
            encoder_model: 'q4',
            decoder_model_merged: 'q4',
        },
        size: '~590 MB'
    },
	{
        id: 'willopcbeta/unsloth-whisper-small-ONNX',
        name: 'unsloth-whisper-small',
        quantization: {
            encoder_model: 'q4',
            decoder_model_merged: 'q4',
        },
        size: '~290 MB'
    },
    {
        id: 'nico-martin/whisper-base-ONNX',
        name: 'Whisper Base',
        quantization: {
            encoder_model: 'f32',
            decoder_model_merged: 'f32',  
        },
        size: '~150 MB'
    }
];

export const ASR_MODELS_RUNTIME: AsrModelRuntime[] = [
    {
        id: 'whisper',
        modelId: 'whisper',
        engine: 'whisper',
        name: 'Transformers.js Whisper',
        size: '~150 MB - ~590 MB',
        description: 'Transformers.js WebGPU / WASM'
    },
    {
        id: 'nemotron',
        modelId: 'nemotron',
        engine: 'nemotron',
        name: 'Nemotron-ASR (onnxruntime-web)',
        size: '~863 MB',
        description: 'Fast Conformer CTC'
    },
    {
        id: 'qwen3',
        modelId: 'qwen3',
        engine: 'qwen3',
        name: 'Qwen3-ASR-0.6B (onnxruntime-web)',
        size: 'q4f16 (~888MB) / q4 (~1.29GB)',
        description: 'Auto WebGPU shader-f16'
    }
];

export const OCR_MODELS = {
  PP_v6_tiny: {
    name: "PP-OCRv6_tiny",
    description: "settings.ocrModelDescriptions.PP_v6_tiny",
    paths: {
      detPath: 'https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.2/onnx/PP-OCRv6/det/PP-OCRv6_det_tiny.onnx',
      recPath: 'https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.2/onnx/PP-OCRv6/rec/PP-OCRv6_rec_small.onnx',
    },
  },
  PP_v6_small: {
    name: "PP-OCRv6_small",
    description: "settings.ocrModelDescriptions.PP_v6_small",
    paths: {
      detPath: 'https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.2/onnx/PP-OCRv6/det/PP-OCRv6_det_small.onnx',
      recPath: 'https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.2/onnx/PP-OCRv6/rec/PP-OCRv6_rec_small.onnx',
    },
  },
  PP_v6_medium: {
    name: "PP-OCRv6_medium",
    description: "settings.ocrModelDescriptions.PP_v6_medium",
    paths: {
      detPath: 'https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.2/onnx/PP-OCRv6/det/PP-OCRv6_det_medium.onnx',
      recPath: 'https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.2/onnx/PP-OCRv6/rec/PP-OCRv6_rec_medium.onnx',
    },
  },
  ch_v5: {
    name: "ch_PP-OCRv5",
    description: "settings.ocrModelDescriptions.ch_v5",
    paths: {
      detPath: 'https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.8.0/onnx/PP-OCRv5/det/ch_PP-OCRv5_det_mobile.onnx',
      recPath: 'https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.8.0/onnx/PP-OCRv5/rec/ch_PP-OCRv5_rec_mobile.onnx',
      dictPath: 'https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.8.0/paddle/PP-OCRv5/rec/ch_PP-OCRv5_rec_mobile/ppocrv5_dict.txt',
    },
  },
  latin_v5: {
    name: "latin_PP-OCRv5",
    description: "settings.ocrModelDescriptions.latin_v5",
    paths: {
      detPath: 'https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.8.0/onnx/PP-OCRv5/det/ch_PP-OCRv5_det_mobile.onnx',
      recPath: 'https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.8.0/onnx/PP-OCRv5/rec/latin_PP-OCRv5_rec_mobile.onnx',
      dictPath: 'https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.8.0/paddle/PP-OCRv5/rec/latin_PP-OCRv5_rec_mobile/ppocrv5_latin_dict.txt',
    },
  },
  kr_v5: {
    name: "korean_PP-OCRv5",
    description: "settings.ocrModelDescriptions.kr_v5",
    paths: {
      detPath: 'https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.8.0/onnx/PP-OCRv5/det/ch_PP-OCRv5_det_mobile.onnx',
      recPath: 'https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.8.0/onnx/PP-OCRv5/rec/korean_PP-OCRv5_rec_mobile.onnx',
      dictPath: 'https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.8.0/paddle/PP-OCRv5/rec/korean_PP-OCRv5_rec_mobile/ppocrv5_korean_dict.txt',
    },
  },
};