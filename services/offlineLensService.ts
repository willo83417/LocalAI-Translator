import type { ImageLensBlock, ImageLensResult } from '../types';

/**
 * Parses batch translation output containing numbered items like:
 * [1] Translated text 1
 * [2] Translated text 2
 * Also falls back to "1. Translated text" or line-by-line alignment if needed.
 */
export function parseBatchNumberedTranslations(
    rawOutput: string,
    expectedCount: number
): Map<number, string> {
    const resultMap = new Map<number, string>();
    if (!rawOutput) return resultMap;

    const trimmed = rawOutput.trim();

    // Strategy 1: Regex with [index] bracket patterns
    const bracketRegex = /\[(\d+)\]\s*([\s\S]*?)(?=(?:\r?\n\s*\[\d+\]|$))/g;
    let match: RegExpExecArray | null;
    let foundWithBrackets = 0;

    while ((match = bracketRegex.exec(trimmed)) !== null) {
        const idx = parseInt(match[1], 10);
        const text = match[2].trim();
        if (!isNaN(idx) && text) {
            resultMap.set(idx, text);
            foundWithBrackets++;
        }
    }

    if (foundWithBrackets >= Math.min(expectedCount, 1)) {
        return resultMap;
    }

    // Strategy 2: Line by line scanning for "1. text" or "1) text" or "1: text" or "[1] text"
    const lines = trimmed.split(/\r?\n/).map(l => l.trim()).filter(Boolean);
    const lineNumRegex = /^\[?(\d+)\]?[\.\:\)\s\-]+(.*)$/;

    for (const line of lines) {
        const m = line.match(lineNumRegex);
        if (m) {
            const idx = parseInt(m[1], 10);
            const text = m[2].trim();
            if (!isNaN(idx) && text && !resultMap.has(idx)) {
                resultMap.set(idx, text);
            }
        }
    }

    if (resultMap.size >= Math.min(expectedCount, 1)) {
        return resultMap;
    }

    // Strategy 3: Direct positional fallback if line count matches
    if (lines.length === expectedCount) {
        lines.forEach((line, i) => {
            const cleanLine = line.replace(/^\[?\d+\]?[\.\:\)\s\-]+/, '').trim();
            resultMap.set(i + 1, cleanLine || line);
        });
    }

    return resultMap;
}

/**
 * Performs offline batch translation for AR Lens blocks using Gemma-4 / local LLM worker.
 */
export async function translateOfflineLensBlocks(
    blocks: ImageLensBlock[],
    sourceLangName: string,
    targetLangName: string,
    worker: Worker,
    signal?: AbortSignal
): Promise<{ sourceText: string; translatedText: string; blocks: ImageLensBlock[] }> {
    if (blocks.length === 0) {
        return { sourceText: '', translatedText: '', blocks: [] };
    }

    const batchItems = blocks.map((b, i) => `[${i + 1}] ${b.sourceText}`).join('\n');
    const prompt = `You are an expert OCR post-correction and translation engine proficient in ${sourceLangName} and ${targetLangName}.
Translate each numbered text block below from ${sourceLangName} into natural, accurate ${targetLangName}.
CRITICAL RULES:
- Output every translated item with its corresponding original number tag in the exact format: [index] translated_text
- Preserve the exact order and count from [1] to [${blocks.length}]. Keep each block on a single line.
- Automatically fix common OCR character lookalike errors in context before translating (for example, in Japanese menus/receipts, "ラ一メソ" or "ラーメソ" or "拉一女" is "ラーメン" (拉麵/Ramen), "チャーシュー" is "叉燒", "メンマ" is "筍乾", "ラムネ" is "彈珠汽水").
- Output ONLY the numbered translated lines. Do not add explanations, conversational filler, markdown fences, notes, or headers.

Blocks to translate:
${batchItems}
`;

    return new Promise((resolve, reject) => {
        let isSettled = false;

        const cleanup = () => {
            worker.removeEventListener('message', handleMessage);
            worker.removeEventListener('error', handleError);
            if (signal) {
                signal.removeEventListener('abort', handleAbort);
            }
        };

        const handleAbort = () => {
            if (isSettled) return;
            isSettled = true;
            cleanup();
            worker.postMessage({ type: 'cancel_task' });
            reject(new DOMException('Offline AR translation cancelled.', 'AbortError'));
        };

        if (signal?.aborted) {
            handleAbort();
            return;
        }

        const handleMessage = (event: MessageEvent) => {
            const { type, payload } = event.data || {};
            if (type === 'translation_full_done') {
                if (isSettled) return;
                isSettled = true;
                cleanup();

                const rawResult: string = payload?.result || '';
                const parsedTranslations = parseBatchNumberedTranslations(rawResult, blocks.length);

                const updatedBlocks: ImageLensBlock[] = blocks.map((b, i) => {
                    const translated = parsedTranslations.get(i + 1) || b.sourceText;
                    return {
                        ...b,
                        translatedText: translated
                    };
                });

                const fullSource = updatedBlocks.map(b => b.sourceText).join('\n\n');
                const fullTranslated = updatedBlocks.map(b => b.translatedText).join('\n\n');

                resolve({
                    sourceText: fullSource,
                    translatedText: fullTranslated,
                    blocks: updatedBlocks
                });
            } else if (type === 'translation_error' || type === 'error') {
                if (isSettled) return;
                isSettled = true;
                cleanup();
                reject(new Error(payload?.error || 'Offline model translation failed'));
            } else if (type === 'translation_cancelled') {
                if (isSettled) return;
                isSettled = true;
                cleanup();
                reject(new DOMException('Offline model translation cancelled', 'AbortError'));
            }
        };

        const handleError = (error: ErrorEvent) => {
            if (isSettled) return;
            isSettled = true;
            cleanup();
            reject(new Error(error.message || 'Worker encountered an error during translation'));
        };

        worker.addEventListener('message', handleMessage);
        worker.addEventListener('error', handleError);
        if (signal) {
            signal.addEventListener('abort', handleAbort);
        }

        // Post translation request to offline controller worker
        worker.postMessage({
            type: 'translate_full',
            payload: {
                text: prompt,
                sourceLang: sourceLangName,
                targetLang: targetLangName
            }
        });
    });
}
