import type { BoxType, EsearchOCROutput, EsearchOCRItem, ImageLensBlock } from '../types';

/**
 * Calculates the bounding box encompassing an array of 4-point OCR boxes.
 */
export function mergeBoxes(boxes: BoxType[]): BoxType {
    if (boxes.length === 0) {
        return [[0, 0], [0, 0], [0, 0], [0, 0]];
    }
    let minX = Infinity;
    let maxX = -Infinity;
    let minY = Infinity;
    let maxY = -Infinity;

    for (const box of boxes) {
        for (const [x, y] of box) {
            if (x < minX) minX = x;
            if (x > maxX) maxX = x;
            if (y < minY) minY = y;
            if (y > maxY) maxY = y;
        }
    }

    return [
        [minX, minY],
        [maxX, minY],
        [maxX, maxY],
        [minX, maxY]
    ];
}

/**
 * Converts a 4-point OCR pixel bounding box into normalized [ymin, xmin, ymax, xmax] (0..1000)
 */
export function boxToNormalized2d(
    box: BoxType,
    imageWidth: number,
    imageHeight: number
): [number, number, number, number] {
    if (!imageWidth || !imageHeight) {
        return [0, 0, 0, 0];
    }

    const xs = box.map(p => p[0]);
    const ys = box.map(p => p[1]);
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);

    const ymin = Math.max(0, Math.min(1000, Math.round((minY / imageHeight) * 1000)));
    const xmin = Math.max(0, Math.min(1000, Math.round((minX / imageWidth) * 1000)));
    const ymax = Math.max(0, Math.min(1000, Math.round((maxY / imageHeight) * 1000)));
    const xmax = Math.max(0, Math.min(1000, Math.round((maxX / imageWidth) * 1000)));

    return [ymin, xmin, ymax, xmax];
}

interface InternalRect {
    text: string;
    minX: number;
    maxX: number;
    minY: number;
    maxY: number;
    width: number;
    height: number;
    centerX: number;
    centerY: number;
    isVertical: boolean;
}

function itemToRect(item: EsearchOCRItem, isGlobalVertical: boolean): InternalRect {
    const xs = item.box.map(p => p[0]);
    const ys = item.box.map(p => p[1]);
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);
    const width = Math.max(1, maxX - minX);
    const height = Math.max(1, maxY - minY);
    const cleanLen = (item.text || '').replace(/\s+/g, '').length;

    // A box is vertical if the global reading direction is inline-top-to-bottom
    // AND the box is taller than wide, OR if an individual line has >1 character and is significantly taller than wide.
    const isVertical = isGlobalVertical
        ? height > width * 1.15
        : (cleanLen > 1 && height > width * 1.6);

    return {
        text: (item.text || '').trim(),
        minX,
        maxX,
        minY,
        maxY,
        width,
        height,
        centerX: (minX + maxX) / 2,
        centerY: (minY + maxY) / 2,
        isVertical
    };
}

function joinClusterTexts(a: string, b: string): string {
    if (!a) return b;
    if (!b) return a;
    const cjkOrPunct = /[\u3000-\u303f\u3040-\u309f\u30a0-\u30ff\uff00-\uffef\u4e00-\u9faf\u3400-\u4dbf]/;
    const lastA = a[a.length - 1];
    const firstB = b[0];
    if (cjkOrPunct.test(lastA) && cjkOrPunct.test(firstB)) {
        return `${a}${b}`;
    }
    return `${a} ${b}`;
}

/**
 * Groups individual OCR detections into clean, non-overlapping AR overlay blocks:
 * 1. Merges adjacent words within the same phrase on a horizontal line (small gap only),
 *    while strictly preserving multi-column menu boundaries and separate item/price tags.
 * 2. Merges tightly stacked multi-line prose paragraphs ONLY when they share the exact same
 *    column alignment and long-line prose structure, never merging menu items or price rows.
 * 3. Preserves vertical columns and groups multi-column vertical prose paragraphs cleanly.
 */
function buildSmartLensBlocksFromSrc(
    srcItems: EsearchOCRItem[],
    isGlobalVertical: boolean
): { text: string; box: BoxType; isVertical: boolean }[] {
    const validRects = srcItems
        .map(item => itemToRect(item, isGlobalVertical))
        .filter(r => {
            const cleaned = r.text.replace(/[\s\r\n\t_.,;:`~'"]+/g, '');
            return cleaned.length > 0;
        });

    if (validRects.length === 0) return [];

    const horizontalRects = validRects.filter(r => !r.isVertical);
    const verticalRects = validRects.filter(r => r.isVertical);

    const results: { text: string; box: BoxType; isVertical: boolean; sortKeyY: number; sortKeyX: number }[] = [];

    // --- Process Horizontal Rects ---
    if (horizontalRects.length > 0) {
        // Step 1: Sort top-to-bottom, then left-to-right
        horizontalRects.sort((a, b) => {
            if (Math.abs(a.centerY - b.centerY) < Math.min(a.height, b.height) * 0.45) {
                return a.minX - b.minX;
            }
            return a.centerY - b.centerY;
        });

        // Group items that sit on the same horizontal baseline
        const rows: InternalRect[][] = [];
        for (const rect of horizontalRects) {
            let matchedRow: InternalRect[] | null = null;
            for (let i = rows.length - 1; i >= 0; i--) {
                const row = rows[i];
                const lastInRow = row[row.length - 1];
                const minH = Math.min(lastInRow.height, rect.height);
                const overlapY = Math.min(lastInRow.maxY, rect.maxY) - Math.max(lastInRow.minY, rect.minY);

                if (Math.abs(rect.centerY - lastInRow.centerY) < minH * 0.42 && overlapY > minH * 0.45) {
                    matchedRow = row;
                    break;
                }
            }
            if (matchedRow) {
                matchedRow.push(rect);
            } else {
                rows.push([rect]);
            }
        }

        // Within each row, sort left-to-right and ONLY merge words belonging to the same phrase (tight word spacing).
        // Do NOT merge across menu columns or between distant prices and next-column labels!
        const mergedLineRects: InternalRect[] = [];
        for (const row of rows) {
            row.sort((a, b) => a.minX - b.minX);
            let current = { ...row[0] };
            for (let i = 1; i < row.length; i++) {
                const next = row[i];
                const gapX = next.minX - current.maxX;
                const avgH = (current.height + next.height) / 2;
                const heightRatio = Math.max(current.height, next.height) / Math.max(1, Math.min(current.height, next.height));

                // Never merge if the left block already ends with a price/currency (e.g., "850円" + "煮玉子")
                // or if the right block is a standalone price tag separated from the item name.
                const currentIsPrice = /[¥￥$€£]\s*\d+|\d+\s*[円元]/.test(current.text);
                const nextIsPrice = /^[¥￥$€£]?\s*\d[\d,.]*\s*[円元]?$/.test(next.text.trim());

                // Tight word gap threshold (0.65x line height) so words in "Hello World OCR Test" merge,
                // while separate columns or price columns stay in their exact own boxes.
                if (gapX >= -avgH * 0.3 && gapX <= avgH * 0.65 && heightRatio < 1.35 && !currentIsPrice && !nextIsPrice) {
                    current.text = joinClusterTexts(current.text, next.text);
                    current.minX = Math.min(current.minX, next.minX);
                    current.maxX = Math.max(current.maxX, next.maxX);
                    current.minY = Math.min(current.minY, next.minY);
                    current.maxY = Math.max(current.maxY, next.maxY);
                    current.width = current.maxX - current.minX;
                    current.height = current.maxY - current.minY;
                    current.centerX = (current.minX + current.maxX) / 2;
                    current.centerY = (current.minY + current.maxY) / 2;
                } else {
                    mergedLineRects.push(current);
                    current = { ...next };
                }
            }
            mergedLineRects.push(current);
        }

        // Step 2: Merge multi-line horizontal prose paragraphs ONLY when they clearly form a continuous text paragraph
        // (Left-aligned, similar width/font height, small vertical gap, and neither line contains prices)
        mergedLineRects.sort((a, b) => a.minY - b.minY || a.minX - b.minX);
        const used = new Array(mergedLineRects.length).fill(false);

        for (let i = 0; i < mergedLineRects.length; i++) {
            if (used[i]) continue;
            used[i] = true;
            const paraLines: InternalRect[] = [mergedLineRects[i]];

            let keepLooking = true;
            while (keepLooking) {
                keepLooking = false;
                const lastLine = paraLines[paraLines.length - 1];
                const firstLine = paraLines[0];

                for (let j = i + 1; j < mergedLineRects.length; j++) {
                    if (used[j]) continue;
                    const candidate = mergedLineRects[j];

                    const verticalGap = candidate.minY - lastLine.maxY;
                    const avgH = (lastLine.height + candidate.height) / 2;

                    if (verticalGap > avgH * 0.65) {
                        if (verticalGap > avgH * 1.5) break;
                        continue;
                    }
                    if (verticalGap < -avgH * 0.25) continue;

                    const heightRatio = Math.max(lastLine.height, candidate.height) / Math.max(1, Math.min(lastLine.height, candidate.height));
                    const leftDiff = Math.abs(candidate.minX - firstLine.minX);
                    const widthRatio = Math.max(lastLine.width, firstLine.width) / Math.max(1, Math.min(lastLine.width, firstLine.width));
                    const isLongProseLine = lastLine.text.length >= 14 && candidate.text.length >= 8 && lastLine.width > lastLine.height * 4.5;

                    // Never merge if either line contains a price or currency unit (protects menus and receipts)
                    const hasPriceOrMenuNumber = /[¥￥$€£]\s*\d+|\d+\s*[円元]|\d{3,}/.test(lastLine.text) || /[¥￥$€£]\s*\d+|\d+\s*[円元]|\d{3,}/.test(candidate.text);

                    if (
                        heightRatio < 1.25 &&
                        leftDiff <= avgH * 0.9 &&
                        widthRatio < 1.35 &&
                        isLongProseLine &&
                        !hasPriceOrMenuNumber &&
                        Math.abs(candidate.width - lastLine.width) <= firstLine.width * 0.45
                    ) {
                        used[j] = true;
                        paraLines.push(candidate);
                        keepLooking = true;
                        break;
                    }
                }
            }

            const minX = Math.min(...paraLines.map(l => l.minX));
            const maxX = Math.max(...paraLines.map(l => l.maxX));
            const minY = Math.min(...paraLines.map(l => l.minY));
            const maxY = Math.max(...paraLines.map(l => l.maxY));
            let combinedText = paraLines[0].text;
            for (let k = 1; k < paraLines.length; k++) {
                combinedText = joinClusterTexts(combinedText, paraLines[k].text);
            }

            results.push({
                text: combinedText,
                box: [
                    [minX, minY],
                    [maxX, minY],
                    [maxX, maxY],
                    [minX, maxY]
                ],
                isVertical: false,
                sortKeyY: minY,
                sortKeyX: minX
            });
        }
    }

    // --- Process Vertical Rects (Japanese / Traditional Chinese vertical writing) ---
    if (verticalRects.length > 0) {
        // Sort right-to-left (descending X), then top-to-bottom
        verticalRects.sort((a, b) => {
            if (Math.abs(a.centerX - b.centerX) < Math.min(a.width, b.width) * 0.5) {
                return a.minY - b.minY;
            }
            return b.centerX - a.centerX;
        });

        // Group multi-column vertical prose paragraphs when adjacent columns have matching top alignment and font width
        const usedV = new Array(verticalRects.length).fill(false);
        for (let i = 0; i < verticalRects.length; i++) {
            if (usedV[i]) continue;
            usedV[i] = true;
            const colGroup: InternalRect[] = [verticalRects[i]];

            let keepLooking = true;
            while (keepLooking) {
                keepLooking = false;
                const lastCol = colGroup[colGroup.length - 1];
                const firstCol = colGroup[0];

                for (let j = i + 1; j < verticalRects.length; j++) {
                    if (usedV[j]) continue;
                    const candidate = verticalRects[j];

                    // Candidate is to the left of lastCol
                    const gapX = lastCol.minX - candidate.maxX;
                    const avgW = (lastCol.width + candidate.width) / 2;

                    if (gapX > avgW * 1.35 || gapX < -avgW * 0.4) continue;

                    const widthRatio = Math.max(lastCol.width, candidate.width) / Math.max(1, Math.min(lastCol.width, candidate.width));
                    const topDiff = Math.abs(candidate.minY - firstCol.minY);
                    const isProseColumn = lastCol.text.length >= 6 && lastCol.height > lastCol.width * 3.5;

                    if (widthRatio < 1.35 && topDiff <= avgW * 1.8 && isProseColumn) {
                        usedV[j] = true;
                        colGroup.push(candidate);
                        keepLooking = true;
                        break;
                    }
                }
            }

            const minX = Math.min(...colGroup.map(c => c.minX));
            const maxX = Math.max(...colGroup.map(c => c.maxX));
            const minY = Math.min(...colGroup.map(c => c.minY));
            const maxY = Math.max(...colGroup.map(c => c.maxY));
            let combinedText = colGroup[0].text;
            for (let k = 1; k < colGroup.length; k++) {
                combinedText = joinClusterTexts(combinedText, colGroup[k].text);
            }

            results.push({
                text: combinedText,
                box: [
                    [minX, minY],
                    [maxX, minY],
                    [maxX, maxY],
                    [minX, maxY]
                ],
                isVertical: true,
                sortKeyY: minY,
                sortKeyX: -maxX
            });
        }
    }

    return results.map(({ text, box, isVertical }) => ({ text, box, isVertical }));
}

/**
 * Extracts structured paragraph or line blocks from EsearchOCROutput and transforms them
 * into ImageLensBlock objects with normalized coordinates for AR overlay rendering.
 */
export function convertOcrToLensBlocks(
    output: EsearchOCROutput,
    imageWidth: number,
    imageHeight: number
): ImageLensBlock[] {
    // IMPORTANT: In esearch-ocr, horizontal text has readingDir = { inline: 'lr', block: 'tb' }
    // while vertical text has readingDir = { inline: 'tb', block: 'rl' }.
    // Therefore, ONLY inline === 'tb' (or block === 'rl') indicates vertical text!
    const isGlobalVertical = output.readingDir?.inline === 'tb' || output.readingDir?.block === 'rl';

    if (output.src && output.src.length > 0) {
        const smartBlocks = buildSmartLensBlocksFromSrc(output.src, isGlobalVertical);
        if (smartBlocks.length > 0) {
            return smartBlocks.map(item => ({
                box_2d: boxToNormalized2d(item.box, imageWidth, imageHeight),
                sourceText: item.text,
                translatedText: '',
                isVertical: item.isVertical
            }));
        }
    }

    const rawBlocks: { text: string; box: BoxType }[] = [];
    if (output.parragraphs && output.parragraphs.length > 0) {
        for (const para of output.parragraphs) {
            const text = (para.text || '').trim();
            if (!text) continue;
            rawBlocks.push({ text, box: para.box });
        }
    }

    return rawBlocks
        .filter(b => b.text.replace(/[\s\r\n\t_.,;:`~'"]+/g, '').length > 0)
        .map(item => {
            const box_2d = boxToNormalized2d(item.box, imageWidth, imageHeight);
            const [ymin, xmin, ymax, xmax] = box_2d;
            const height = ymax - ymin;
            const width = xmax - xmin;
            const isVertical = isGlobalVertical
                ? height > width * 1.15
                : (item.text.trim().length > 1 && height > width * 1.6);

            return {
                box_2d,
                sourceText: item.text,
                translatedText: '',
                isVertical
            };
        });
}
