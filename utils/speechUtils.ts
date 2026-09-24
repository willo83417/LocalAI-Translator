/**
 * Speech Recognition and Speech Synthesis (TTS) Utility Standards
 * Follows Google Web Speech API & W3C SpeechSynthesis BCP 47 specifications.
 */

// Exact Google Speech Recognition / Android SpeechRecognizer BCP 47 locale mapping
export const GOOGLE_SPEECH_MAP: Record<string, string> = {
    // Traditional Chinese (Taiwan)
    'zh-TW': 'cmn-Hant-TW',
    'zh-Hant': 'cmn-Hant-TW',
    'zh-Hant-TW': 'cmn-Hant-TW',
    'cmn-Hant-TW': 'cmn-Hant-TW',
    'cmn-TW': 'cmn-Hant-TW',

    // Traditional Chinese (Hong Kong - Cantonese)
    'zh-HK': 'yue-Hant-HK',
    'zh-Hant-HK': 'yue-Hant-HK',
    'cmn-Hans-HK': 'cmn-Hans-HK',
    'yue-Hant-HK': 'yue-Hant-HK',
    'yue-HK': 'yue-Hant-HK',
    'zh-YUE': 'yue-Hant-HK',

    // Simplified Chinese (Mainland China / Singapore)
    'zh-CN': 'cmn-Hans-CN',
    'zh-Hans': 'cmn-Hans-CN',
    'zh-Hans-CN': 'cmn-Hans-CN',
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
    'en-NZ': 'en-NZ',

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
    'fr-CA': 'fr-CA',

    // German
    'de': 'de-DE',
    'de-DE': 'de-DE',

    // Russian
    'ru': 'ru-RU',
    'ru-RU': 'ru-RU',

    // Italian
    'it': 'it-IT',
    'it-IT': 'it-IT',

    // Vietnamese, Thai, Indonesian, Portuguese, Dutch, Polish
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

/**
 * Normalizes any language code or alias into official Google Web Speech API BCP 47 locale.
 */
export function normalizeWebSpeechLang(rawLang: string): string {
    if (!rawLang || rawLang === 'auto' || rawLang === 'Auto Detect' || rawLang === 'autodetect') {
        const browserLang = (typeof navigator !== 'undefined' && (navigator.language || (navigator as any).userLanguage)) || 'cmn-Hant-TW';
        return normalizeWebSpeechLang(browserLang);
    }

    const trimmed = rawLang.trim();
    if (GOOGLE_SPEECH_MAP[trimmed]) return GOOGLE_SPEECH_MAP[trimmed];

    // Regex checks for Chinese variants
    if (/^zh[-_]Hant/i.test(trimmed) || /^zh[-_]TW/i.test(trimmed)) return 'cmn-Hant-TW';
    if (/^zh[-_]HK/i.test(trimmed) || /^yue/i.test(trimmed)) return 'yue-Hant-HK';
    if (/^zh[-_]Hans/i.test(trimmed) || /^zh[-_]CN/i.test(trimmed)) return 'cmn-Hans-CN';

    const prefix = trimmed.split(/[-_]/)[0].toLowerCase();
    return GOOGLE_SPEECH_MAP[prefix] || trimmed;
}

/**
 * Fallback alternative language in case a browser or engine specifically lacks cmn-Hant-TW
 */
export function getFallbackWebSpeechLang(currentLang: string): string {
    const fallbacks: Record<string, string> = {
        'cmn-Hant-TW': 'zh-TW',
        'zh-TW': 'cmn-Hant-TW',
        'cmn-Hans-CN': 'zh-CN',
        'zh-CN': 'cmn-Hans-CN',
        'yue-Hant-HK': 'zh-HK',
        'zh-HK': 'yue-Hant-HK',
    };
    return fallbacks[currentLang] || currentLang;
}

/**
 * Evaluates how well a SpeechSynthesisVoice matches a target language code.
 * Returns a score:
 *   >= 100: Exact dialect match (e.g. cmn-Hant-TW for zh-TW, or yue-Hant-HK for zh-HK)
 *   >= 50: Same language and regional dialect family
 *   >= 10: General language match (for non-Chinese languages)
 *   0: Incompatible / no match
 */
export function getVoiceMatchScore(voice: SpeechSynthesisVoice, targetLangCode: string): number {
    if (!voice || !targetLangCode) return 0;

    const vLang = (voice.lang || '').replace('_', '-').trim().toLowerCase();
    const vName = (voice.name || '').toLowerCase();
    const targetNorm = normalizeWebSpeechLang(targetLangCode).toLowerCase();
    const targetBase = targetLangCode.toLowerCase();

    // 1. Traditional Chinese (Taiwan): zh-TW / cmn-Hant-TW
    if (targetNorm === 'cmn-hant-tw' || targetBase === 'zh-tw' || targetBase === 'zh-hant') {
        // Must reject Cantonese (HK/yue) and Mainland China Simplified (CN/Hans)
        if (vLang.includes('hk') || vLang.startsWith('yue') || vName.includes('cantonese') || vName.includes('香港') || vName.includes('粵語') || vName.includes('粤语')) {
            return 0;
        }
        if (vLang.includes('cn') || vLang.includes('hans') || vName.includes('china') || vName.includes('大陆') || vName.includes('大陸') || vName.includes('普通话') || vName.includes('普通話')) {
            return 0;
        }

        // Direct Google Web Speech standard matches
        if (vLang === 'cmn-hant-tw' || vLang === 'cmn-tw') return 120;
        if (vLang === 'zh-tw' || vLang === 'zh-hant-tw') return 110;
        if (vName.includes('taiwan') || vName.includes('臺灣') || vName.includes('台灣') || vName.includes('國語') || vName.includes('mei-jia') || vName.includes('hanhan') || vName.includes('yating') || vName.includes('hsiaochen')) {
            return 100;
        }
        if (vLang === 'zh-hant') return 80;
        return 0;
    }

    // 2. Traditional Chinese (Hong Kong / Cantonese): zh-HK / yue-Hant-HK
    if (targetNorm === 'yue-hant-hk' || targetBase === 'zh-hk') {
        // Direct Cantonese matches
        if (vLang === 'yue-hant-hk' || vLang === 'yue-hk' || vLang === 'zh-yue') return 120;
        if (vLang === 'zh-hk' || vLang === 'zh-hant-hk') return 110;
        if (vName.includes('hong kong') || vName.includes('香港') || vName.includes('cantonese') || vName.includes('粵語') || vName.includes('粤语') || vName.includes('廣東話') || vName.includes('sin-ji') || vName.includes('hiumaan') || vName.includes('danny')) {
            return 100;
        }
        if (vLang.startsWith('yue')) return 90;
        return 0;
    }

    // 3. Simplified Chinese: zh-Hans / zh-CN
    if (targetNorm === 'cmn-hans-cn' || targetBase === 'zh-hans' || targetBase === 'zh-cn') {
        // Must reject Taiwan (TW/Hant) and Cantonese (HK/yue)
        if (vLang.includes('tw') || vLang.includes('hant') || vName.includes('taiwan') || vName.includes('臺灣') || vName.includes('台灣')) {
            return 0;
        }
        if (vLang.includes('hk') || vLang.startsWith('yue') || vName.includes('cantonese') || vName.includes('香港')) {
            return 0;
        }

        if (vLang === 'cmn-hans-cn' || vLang === 'cmn-cn') return 120;
        if (vLang === 'zh-cn' || vLang === 'zh-hans-cn' || vLang === 'zh-hans') return 110;
        if (vName.includes('china') || vName.includes('大陆') || vName.includes('大陸') || vName.includes('普通话') || vName.includes('普通話') || vName.includes('xiaoxiao') || vName.includes('yunxi') || vName.includes('tingting')) {
            return 100;
        }
        if (vLang === 'zh-sg') return 70;
        return 0;
    }

    // 4. Exact standard BCP 47 code match
    if (vLang === targetNorm) return 120;
    if (vLang === targetBase) return 110;

    // 5. Region/dialect match for other languages
    const targetRegion = targetBase.split('-')[1]?.toLowerCase();
    const vRegion = vLang.split('-')[1]?.toLowerCase();
    const targetLangPrefix = targetBase.split('-')[0].toLowerCase();
    const vLangPrefix = vLang.split('-')[0].toLowerCase();

    if (targetLangPrefix === vLangPrefix) {
        if (targetRegion && vRegion && targetRegion === vRegion) {
            return 100;
        }
        // Preferred base match
        return 50;
    }

    return 0;
}

/**
 * Filters and prioritizes SpeechSynthesisVoices for a given target language.
 * Ensures Google Web Speech API & system voices for zh-TW, zh-HK, zh-CN, etc. are properly captured.
 */
export function filterVoicesForLanguage(voices: SpeechSynthesisVoice[], targetLangCode: string): SpeechSynthesisVoice[] {
    if (!voices || voices.length === 0 || !targetLangCode) return [];

    const scored = voices
        .map(voice => ({ voice, score: getVoiceMatchScore(voice, targetLangCode) }))
        .filter(item => item.score > 0);

    // Sort by:
    // 1. Highest match score first
    // 2. Google / Natural / Premium / Local voices first
    // 3. Alphabetical voice name
    scored.sort((a, b) => {
        if (b.score !== a.score) {
            return b.score - a.score;
        }

        const aIsGoogleOrNatural = /google|natural|online/i.test(a.voice.name);
        const bIsGoogleOrNatural = /google|natural|online/i.test(b.voice.name);
        if (aIsGoogleOrNatural && !bIsGoogleOrNatural) return -1;
        if (!aIsGoogleOrNatural && bIsGoogleOrNatural) return 1;

        return a.voice.name.localeCompare(b.voice.name);
    });

    return scored.map(item => item.voice);
}

/**
 * Returns sample text for previewing TTS voices in a given language.
 */
export function getSampleSpeechText(langCode: string): string {
    const norm = normalizeWebSpeechLang(langCode).toLowerCase();
    const code = (langCode || '').toLowerCase();

    if (norm === 'cmn-hant-tw' || code.includes('tw')) {
        return '您好，這是一段語音朗讀測試，祝您今天順心愉快！';
    }
    if (norm === 'yue-hant-hk' || code.includes('hk')) {
        return '你好，呢個係語音朗讀測試，祝你今日開開心心！';
    }
    if (norm === 'cmn-hans-cn' || code.includes('hans') || code.includes('cn')) {
        return '您好，这是一段语音朗读测试，祝您今天顺心愉快！';
    }
    if (norm.startsWith('ja')) {
        return 'こんにちは、音声読み上げのテストです。今日も良い一日を！';
    }
    if (norm.startsWith('ko')) {
        return '안녕하세요, 음성 합성 테스트입니다. 오늘도 좋은 하루 되세요!';
    }
    if (norm.startsWith('es')) {
        return '¡Hola! Esta es una prueba de síntesis de voz. ¡Que tengas un excelente día!';
    }
    if (norm.startsWith('fr')) {
        return 'Bonjour, ceci est un test de synthèse vocale. Passez une excellente journée !';
    }
    if (norm.startsWith('de')) {
        return 'Hallo, dies ist ein Sprachsynthese-Test. Einen wunderschönen Tag noch!';
    }
    if (norm.startsWith('ru')) {
        return 'Здравствуйте, это тест синтеза речи. Желаем вам прекрасного дня!';
    }
    if (norm.startsWith('it')) {
        return 'Ciao, questo è un test di sintesi vocale. Ti auguro una splendida giornata!';
    }

    return 'Hello, this is a speech synthesis test. Have a wonderful and productive day!';
}

let activePreviewUtterance: SpeechSynthesisUtterance | null = null;

/**
 * Plays a quick test preview for a specific voice URI.
 */
export function testVoice(
    voiceURI: string,
    sampleText?: string,
    rate = 1,
    pitch = 1,
    langCode = 'zh-TW',
    onEnd?: () => void,
    onError?: (err: any) => void
): void {
    if (typeof window === 'undefined' || !('speechSynthesis' in window)) {
        if (onError) onError(new Error('SpeechSynthesis not supported'));
        return;
    }

    window.speechSynthesis.cancel();

    const voices = window.speechSynthesis.getVoices();
    const targetVoice = voices.find(v => v.voiceURI === voiceURI);
    const textToSpeak = sampleText || getSampleSpeechText(targetVoice?.lang || langCode);

    const utterance = new SpeechSynthesisUtterance(textToSpeak);
    if (targetVoice) {
        utterance.voice = targetVoice;
        utterance.lang = targetVoice.lang;
    } else {
        utterance.lang = normalizeWebSpeechLang(langCode);
    }

    utterance.rate = Math.max(0.5, Math.min(2, rate));
    utterance.pitch = Math.max(0, Math.min(2, pitch));

    utterance.onend = () => {
        activePreviewUtterance = null;
        if (onEnd) onEnd();
    };

    utterance.onerror = (e) => {
        activePreviewUtterance = null;
        if (onError) onError(e);
    };

    activePreviewUtterance = utterance;
    window.speechSynthesis.speak(utterance);
}

/**
 * Stops any active voice test preview.
 */
export function stopVoiceTest(): void {
    if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
        window.speechSynthesis.cancel();
        activePreviewUtterance = null;
    }
}
