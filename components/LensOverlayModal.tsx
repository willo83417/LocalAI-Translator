import React, { useState, useRef, useEffect, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { 
    X, 
    Download, 
    Copy, 
    Check, 
    Eye, 
    EyeOff, 
    Volume2, 
    Sparkles, 
    ChevronDown, 
    ChevronUp,
    ZoomIn,
    ZoomOut,
    RotateCcw,
    Type,
    Camera
} from 'lucide-react';
import type { ImageLensResult } from '../types';

interface LensOverlayModalProps {
    isOpen: boolean;
    onClose: () => void;
    lensResult: ImageLensResult | null;
    isLoading?: boolean;
    onSpeak?: (text: string) => void;
    onCaptureAgain?: () => void;
}

const LensOverlayModal: React.FC<LensOverlayModalProps> = ({
    isOpen,
    onClose,
    lensResult,
    isLoading = false,
    onSpeak,
    onCaptureAgain
}) => {
    const { t } = useTranslation();
    const [showTranslated, setShowTranslated] = useState(true);
    // Track individual block toggles (key: index, value: true if showing original, false if showing translation)
    const [flippedBlocks, setFlippedBlocks] = useState<Record<number, boolean>>({});
    const [isCopied, setIsCopied] = useState(false);
    const [isDownloading, setIsDownloading] = useState(false);
    const [showTextSheet, setShowTextSheet] = useState(false);
    
    // Zoom & Pan state for receipts and documents
    const [zoom, setZoom] = useState(1);
    const [pan, setPan] = useState({ x: 0, y: 0 });
    const [isDragging, setIsDragging] = useState(false);
    const dragOriginRef = useRef({ x: 0, y: 0, panX: 0, panY: 0 });

    // Font size scaling to prevent crowded rows on long receipts
    const [fontSizeScale, setFontSizeScale] = useState<'compact' | 'normal' | 'large'>('normal');
    // Selected block for quick focus & pronunciation
    const [selectedBlockIdx, setSelectedBlockIdx] = useState<number | null>(null);

    const imageRef = useRef<HTMLImageElement>(null);

    useEffect(() => {
        if (isOpen) {
            setShowTranslated(true);
            setFlippedBlocks({});
            setIsCopied(false);
            setShowTextSheet(false);
            setZoom(1);
            setPan({ x: 0, y: 0 });
            setSelectedBlockIdx(null);
            setFontSizeScale('normal');
        }
    }, [isOpen, lensResult]);

    const handleZoomIn = useCallback(() => {
        setZoom(prev => Math.min(3.0, +(prev + 0.35).toFixed(2)));
    }, []);

    const handleZoomOut = useCallback(() => {
        setZoom(prev => {
            const next = Math.max(1.0, +(prev - 0.35).toFixed(2));
            if (next <= 1.0) setPan({ x: 0, y: 0 });
            return next;
        });
    }, []);

    const handleResetZoom = useCallback(() => {
        setZoom(1);
        setPan({ x: 0, y: 0 });
    }, []);

    // Mouse drag handlers for panning when zoomed in
    const handleMouseDown = (e: React.MouseEvent) => {
        if (zoom <= 1) return;
        setIsDragging(true);
        dragOriginRef.current = {
            x: e.clientX,
            y: e.clientY,
            panX: pan.x,
            panY: pan.y
        };
    };

    const handleMouseMove = (e: React.MouseEvent) => {
        if (!isDragging || zoom <= 1) return;
        const dx = e.clientX - dragOriginRef.current.x;
        const dy = e.clientY - dragOriginRef.current.y;
        setPan({
            x: dragOriginRef.current.panX + dx,
            y: dragOriginRef.current.panY + dy
        });
    };

    const handleMouseUp = () => {
        setIsDragging(false);
    };

    // Touch drag handlers for mobile devices
    const handleTouchStart = (e: React.TouchEvent) => {
        if (zoom <= 1 || e.touches.length !== 1) return;
        const touch = e.touches[0];
        setIsDragging(true);
        dragOriginRef.current = {
            x: touch.clientX,
            y: touch.clientY,
            panX: pan.x,
            panY: pan.y
        };
    };

    const handleTouchMove = (e: React.TouchEvent) => {
        if (!isDragging || zoom <= 1) return;
        const touch = e.touches[0];
        const dx = touch.clientX - dragOriginRef.current.x;
        const dy = touch.clientY - dragOriginRef.current.y;
        setPan({
            x: dragOriginRef.current.panX + dx,
            y: dragOriginRef.current.panY + dy
        });
    };

    const handleTouchEnd = () => {
        setIsDragging(false);
    };

    const handleWheel = (e: React.WheelEvent) => {
        if (e.ctrlKey || e.metaKey || e.altKey) {
            e.preventDefault();
            if (e.deltaY < 0) {
                handleZoomIn();
            } else {
                handleZoomOut();
            }
        }
    };

    if (!isOpen || !lensResult) return null;

    const { blocks = [], imageUrl, sourceText, translatedText } = lensResult;

    const handleCopyAll = async () => {
        const textToCopy = translatedText || sourceText;
        if (!textToCopy) return;
        try {
            await navigator.clipboard.writeText(textToCopy);
            setIsCopied(true);
            setTimeout(() => setIsCopied(false), 2000);
        } catch (err) {
            console.error('Failed to copy text:', err);
        }
    };

    const handleToggleBlock = (index: number) => {
        if (isLoading) return;
        setFlippedBlocks(prev => ({
            ...prev,
            [index]: !prev[index]
        }));
    };

    // Download combined image with rendered translated text using Canvas
    const handleDownloadImage = async () => {
        if (!imageUrl || isLoading) return;
        setIsDownloading(true);
        try {
            const img = new Image();
            img.crossOrigin = 'anonymous';
            await new Promise<void>((resolve, reject) => {
                img.onload = () => resolve();
                img.onerror = reject;
                img.src = imageUrl;
            });

            const canvas = document.createElement('canvas');
            canvas.width = img.naturalWidth || img.width;
            canvas.height = img.naturalHeight || img.height;
            const ctx = canvas.getContext('2d');
            if (!ctx) throw new Error('Could not create 2D canvas context');

            // 1. Draw original base image
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

            // 2. Overlay translated blocks if global showTranslated is true
            if (showTranslated && blocks.length > 0) {
                for (let i = 0; i < blocks.length; i++) {
                    const block = blocks[i];
                    const isFlipped = flippedBlocks[i];
                    const displayText = isFlipped ? block.sourceText : block.translatedText;
                    if (!displayText) continue;

                    const [ymin, xmin, ymax, xmax] = block.box_2d;
                    const x = (xmin / 1000) * canvas.width;
                    const y = (ymin / 1000) * canvas.height;
                    const w = Math.max(16, ((xmax - xmin) / 1000) * canvas.width);
                    const h = Math.max(11, ((ymax - ymin) / 1000) * canvas.height);

                    // Draw rounded semi-opaque backdrop
                    ctx.save();
                    ctx.fillStyle = 'rgba(255, 255, 255, 0.94)';
                    ctx.strokeStyle = 'rgba(200, 200, 200, 0.85)';
                    ctx.lineWidth = Math.max(1, canvas.width * 0.001);

                    const radius = Math.min(6, Math.min(w, h) / 3);
                    ctx.beginPath();
                    ctx.roundRect(x, y, w, h, radius);
                    ctx.fill();
                    ctx.stroke();

                    // Draw text inside the box
                    ctx.fillStyle = '#0f172a'; // Slate-900
                    ctx.textBaseline = 'middle';
                    ctx.textAlign = 'center';

                    // Estimate font size based on box height and text length
                    const fontSize = Math.max(9, Math.min(h * 0.72, (w / (displayText.length || 1)) * 1.5));
                    ctx.font = `bold ${fontSize}px sans-serif, system-ui`;

                    ctx.fillText(displayText, x + w / 2, y + h / 2, w - 4);
                    ctx.restore();
                }
            }

            // Export to PNG and trigger download
            const dataUrl = canvas.toDataURL('image/png', 0.95);
            const downloadLink = document.createElement('a');
            downloadLink.href = dataUrl;
            downloadLink.download = `lens-translation-${Date.now()}.png`;
            document.body.appendChild(downloadLink);
            downloadLink.click();
            document.body.removeChild(downloadLink);
        } catch (err) {
            console.error('Failed to export translated image:', err);
        } finally {
            setIsDownloading(false);
        }
    };

    return (
        <div 
            className="fixed inset-0 z-50 flex flex-col bg-slate-950/95 text-white overflow-hidden select-none animate-fade-in backdrop-blur-md"
            role="dialog"
            aria-modal="true"
        >
            <style>{`
                @keyframes lensScannerBeam {
                    0% { top: 0%; opacity: 0.8; }
                    50% { opacity: 1; }
                    100% { top: 98%; opacity: 0.8; }
                }
            `}</style>

            {/* Top Navigation Bar */}
            <div className="w-full flex justify-between items-center px-4 py-2.5 bg-black/60 border-b border-white/10 z-30">
                <div className="flex items-center gap-2">
                    <div className="p-1.5 rounded-lg bg-blue-600/30 text-blue-400 border border-blue-500/30">
                        <Sparkles className={`w-4 h-4 ${isLoading ? 'animate-spin text-amber-400' : ''}`} />
                    </div>
                    <div>
                        <h2 className="text-sm font-semibold text-white tracking-wide">
                            {t('lens.title') || '智慧鏡頭實景翻譯'}
                        </h2>
                        <p className="text-[11px] text-gray-400">
                            {isLoading ? (
                                <span className="text-blue-400 font-medium animate-pulse">
                                    {t('lens.translating') || '智慧鏡頭實景分析中...'}
                                </span>
                            ) : (
                                blocks.length > 0 
                                    ? (t('lens.tapHint') || '點擊方框可切換原文/譯文')
                                    : (t('lens.noBlocks') || '全文翻譯模式')
                            )}
                        </p>
                    </div>
                </div>

                <div className="flex items-center gap-2">
                    {/* Continuous Camera Capture Button (Directly opens CameraView for next photo/album) */}
                    {onCaptureAgain && (
                        <button
                            onClick={onCaptureAgain}
                            className="flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium border bg-emerald-600 hover:bg-emerald-500 border-emerald-400 text-white shadow-lg active:scale-95 transition-all"
                            title={t('lens.cameraTooltip') || '連續拍攝：拍攝下一張或從相簿選擇'}
                        >
                            <Camera className="w-3.5 h-3.5" />
                            <span className="hidden sm:inline">{t('lens.continuousCapture') || '連續拍攝'}</span>
                        </button>
                    )}

                    {/* Toggle show all original vs translated */}
                    {!isLoading && (
                        <button
                            onClick={() => setShowTranslated(!showTranslated)}
                            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium border backdrop-blur-md transition-all active:scale-95 ${
                                showTranslated 
                                    ? 'bg-blue-600 hover:bg-blue-500 border-blue-400 text-white shadow-lg' 
                                    : 'bg-white/10 hover:bg-white/20 border-white/20 text-white/90'
                            }`}
                            title={showTranslated ? (t('lens.showOriginal') || '顯示原文') : (t('lens.showTranslated') || '顯示譯文')}
                        >
                            {showTranslated ? <Eye className="w-3.5 h-3.5" /> : <EyeOff className="w-3.5 h-3.5" />}
                            <span className="hidden md:inline">{showTranslated ? (t('lens.showOriginal') || '檢視原圖') : (t('lens.showTranslated') || '顯示實景譯文')}</span>
                        </button>
                    )}

                    {/* Download Image Button */}
                    {!isLoading && (
                        <button
                            onClick={handleDownloadImage}
                            disabled={isDownloading}
                            className="p-2 rounded-full bg-white/10 hover:bg-white/20 border border-white/15 text-white active:scale-95 transition-all"
                            title={t('lens.downloadImage') || '儲存圖片'}
                        >
                            <Download className="w-4 h-4" />
                        </button>
                    )}

                    {/* Copy Text Button */}
                    {!isLoading && (
                        <button
                            onClick={handleCopyAll}
                            className={`p-2 rounded-full border transition-all active:scale-95 ${
                                isCopied 
                                    ? 'bg-green-600 border-green-500 text-white' 
                                    : 'bg-white/10 hover:bg-white/20 border-white/15 text-white'
                            }`}
                            title={isCopied ? (t('lens.copied') || '已複製') : (t('lens.copyText') || '複製翻譯文字')}
                        >
                            {isCopied ? <Check className="w-4 h-4" /> : <Copy className="w-4 h-4" />}
                        </button>
                    )}

                    {/* Close Button */}
                    <button
                        onClick={onClose}
                        className="p-2 rounded-full bg-white/10 hover:bg-red-500/80 hover:border-red-400 border border-white/15 text-white active:scale-95 transition-all ml-1"
                        aria-label={t('lens.close') || '關閉'}
                    >
                        <X className="w-4 h-4" />
                    </button>
                </div>
            </div>

            {/* Central Canvas / Image Overlay Area with Zoom & Pan */}
            <div 
                className="flex-1 relative w-full h-full flex items-center justify-center p-2 sm:p-4 overflow-hidden"
                onMouseDown={handleMouseDown}
                onMouseMove={handleMouseMove}
                onMouseUp={handleMouseUp}
                onTouchStart={handleTouchStart}
                onTouchMove={handleTouchMove}
                onTouchEnd={handleTouchEnd}
                onWheel={handleWheel}
                style={{ cursor: zoom > 1 ? (isDragging ? 'grabbing' : 'grab') : 'default' }}
            >
                {/* Floating Zoom & Font Size Control Bar */}
                {!isLoading && (
                    <div className="absolute top-4 right-4 z-40 flex items-center gap-1 bg-black/75 backdrop-blur-md px-2 py-1 rounded-full border border-white/20 shadow-2xl text-xs text-white">
                        <button 
                            onClick={handleZoomOut} 
                            disabled={zoom <= 1} 
                            className="p-1 hover:text-blue-400 disabled:opacity-30 transition-colors"
                            title={t('lens.zoomOut') || '縮小'}
                        >
                            <ZoomOut className="w-3.5 h-3.5" />
                        </button>
                        <button 
                            onClick={handleResetZoom}
                            className="px-1.5 py-0.5 font-mono text-[11px] hover:text-blue-400 transition-colors"
                            title={t('lens.zoomReset') || '重設大小'}
                        >
                            {Math.round(zoom * 100)}%
                        </button>
                        <button 
                            onClick={handleZoomIn} 
                            disabled={zoom >= 3.0} 
                            className="p-1 hover:text-blue-400 disabled:opacity-30 transition-colors"
                            title={t('lens.zoomIn') || '放大'}
                        >
                            <ZoomIn className="w-3.5 h-3.5" />
                        </button>

                        <span className="w-px h-3.5 bg-white/20 mx-1" />

                        {/* Font size toggle for dense receipts */}
                        <button
                            onClick={() => {
                                setFontSizeScale(prev => prev === 'compact' ? 'normal' : prev === 'normal' ? 'large' : 'compact');
                            }}
                            className="px-2 py-0.5 text-[11px] font-medium rounded hover:bg-white/10 transition-colors flex items-center gap-1"
                            title="切換字體大小（緊湊/標準/放大）"
                        >
                            <Type className="w-3 h-3 text-amber-400" />
                            <span>
                                {fontSizeScale === 'compact' 
                                    ? (t('lens.fontSizeCompact') || '字體：緊湊') 
                                    : fontSizeScale === 'large'
                                    ? (t('lens.fontSizeLarge') || '字體：放大')
                                    : (t('lens.fontSizeNormal') || '字體：標準')}
                            </span>
                        </button>
                    </div>
                )}

                {/* Scaled/Panned Container */}
                <div 
                    className="relative inline-block max-w-full max-h-[82vh] rounded-xl overflow-hidden shadow-2xl border border-white/10 bg-black/40 transition-transform duration-75 ease-out"
                    style={{
                        transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
                        transformOrigin: 'center center',
                        touchAction: zoom > 1 ? 'none' : 'auto'
                    }}
                >
                    <img
                        ref={imageRef}
                        src={imageUrl}
                        alt="Lens Target"
                        className="max-w-full max-h-[82vh] object-contain block select-none pointer-events-none"
                    />

                    {/* Google Lens Laser Scanning Beam during loading */}
                    {isLoading && (
                        <div className="absolute inset-0 pointer-events-none overflow-hidden z-20">
                            {/* Scanning laser beam */}
                            <div 
                                className="absolute left-0 right-0 h-1 bg-gradient-to-r from-transparent via-cyan-400 to-transparent shadow-[0_0_16px_4px_rgba(34,211,238,0.8)]"
                                style={{
                                    animation: 'lensScannerBeam 2s ease-in-out infinite alternate'
                                }}
                            />
                            {/* Central pulsating status badge */}
                            <div className="absolute inset-0 flex items-center justify-center bg-black/35 backdrop-blur-[2px]">
                                <div className="flex items-center gap-2.5 px-4 py-2 rounded-full bg-slate-900/90 border border-cyan-500/50 shadow-2xl text-xs text-cyan-200 animate-pulse">
                                    <Sparkles className="w-4 h-4 text-amber-400 animate-spin" />
                                    <span className="font-medium tracking-wide">
                                        {t('lens.translating') || '智慧鏡頭正在實景翻譯中...'}
                                    </span>
                                </div>
                            </div>
                        </div>
                    )}

                    {/* AR Overlays Layer (rendered directly matching image bounding boxes) */}
                    {!isLoading && showTranslated && blocks.map((block, idx) => {
                        const [ymin, xmin, ymax, xmax] = block.box_2d;
                        const topPct = (ymin / 10).toFixed(2);
                        const leftPct = (xmin / 10).toFixed(2);
                        const widthPct = Math.max(3.5, (xmax - xmin) / 10).toFixed(2);
                        // Prevent row ballooning on receipts: use realistic row height
                        const rawHeightPct = (ymax - ymin) / 10;
                        const heightPct = Math.max(2.5, rawHeightPct).toFixed(2);

                        const isFlipped = !!flippedBlocks[idx];
                        const isSelected = selectedBlockIdx === idx;
                        const text = isFlipped ? block.sourceText : block.translatedText;

                        const fontStyle = fontSizeScale === 'compact'
                            ? 'text-[clamp(7px,0.85vw,10px)] leading-tight'
                            : fontSizeScale === 'large'
                            ? 'text-[clamp(9px,1.3vw,15px)] leading-snug font-semibold'
                            : 'text-[clamp(7.5px,1.05vw,12px)] leading-tight';

                        return (
                            <div
                                key={idx}
                                onClick={(e) => {
                                    e.stopPropagation();
                                    setSelectedBlockIdx(isSelected ? null : idx);
                                    handleToggleBlock(idx);
                                }}
                                style={{
                                    top: `${topPct}%`,
                                    left: `${leftPct}%`,
                                    width: `${widthPct}%`,
                                    height: `${heightPct}%`,
                                }}
                                className={`absolute flex items-center justify-center px-1 py-0.5 rounded-[3px] cursor-pointer transition-all duration-150 border select-none overflow-hidden ${
                                    isSelected 
                                        ? 'z-40 ring-2 ring-blue-500 shadow-xl scale-[1.03] bg-blue-50 text-slate-900 border-blue-500' 
                                        : isFlipped 
                                        ? 'z-10 hover:z-30 bg-amber-400/95 text-black border-amber-500/80 font-medium hover:scale-[1.02] shadow-xs' 
                                        : 'z-10 hover:z-30 bg-white/95 text-slate-900 border-slate-300/80 hover:bg-white font-medium hover:scale-[1.02] shadow-xs'
                                }`}
                                title={`${isFlipped ? '原文' : '譯文'} • 點擊切換`}
                            >
                                <span 
                                    className={`text-center break-words select-text line-clamp-2 overflow-hidden ${fontStyle} font-sans`}
                                    style={{ wordBreak: 'break-word', overflowWrap: 'break-word' }}
                                >
                                    {text}
                                </span>
                            </div>
                        );
                    })}
                </div>

                {/* Selected Block Quick Detail / Speak Floating Panel */}
                {selectedBlockIdx !== null && blocks[selectedBlockIdx] && (
                    <div className="absolute bottom-6 left-1/2 -translate-x-1/2 w-[92%] max-w-md bg-slate-900/95 backdrop-blur-md border border-blue-500/50 rounded-xl p-3 shadow-2xl z-40 animate-fade-in flex flex-col gap-2">
                        <div className="flex justify-between items-center text-xs">
                            <span className="font-semibold text-blue-400 flex items-center gap-1.5">
                                <Sparkles className="w-3.5 h-3.5 text-amber-400" />
                                {t('lens.blockDetails') || '區塊詳情'}
                            </span>
                            <div className="flex items-center gap-2">
                                <button
                                    onClick={() => handleToggleBlock(selectedBlockIdx)}
                                    className="text-[11px] text-gray-300 hover:text-white px-2 py-0.5 rounded bg-white/10"
                                >
                                    {flippedBlocks[selectedBlockIdx] ? '切換為譯文' : '切換為原文'}
                                </button>
                                {onSpeak && (
                                    <button
                                        onClick={() => onSpeak(blocks[selectedBlockIdx].translatedText)}
                                        className="text-[11px] text-blue-400 hover:text-blue-300 px-2 py-0.5 rounded bg-blue-600/20 flex items-center gap-1"
                                    >
                                        <Volume2 className="w-3 h-3" />
                                        <span>{t('lens.speak') || '朗讀'}</span>
                                    </button>
                                )}
                                <button
                                    onClick={() => setSelectedBlockIdx(null)}
                                    className="text-gray-400 hover:text-white ml-1"
                                >
                                    <X className="w-3.5 h-3.5" />
                                </button>
                            </div>
                        </div>
                        <div className="text-xs space-y-1">
                            <div className="text-white font-medium">
                                <span className="text-gray-400 text-[10px] mr-1.5">[譯文]</span>
                                {blocks[selectedBlockIdx].translatedText}
                            </div>
                            <div className="text-gray-300">
                                <span className="text-gray-400 text-[10px] mr-1.5">[原文]</span>
                                {blocks[selectedBlockIdx].sourceText}
                            </div>
                        </div>
                    </div>
                )}
            </div>

            {/* Bottom Collapsible Sheet: Full Text & TTS */}
            {!isLoading && (
                <div className="w-full bg-slate-900/95 border-t border-white/10 z-20 flex flex-col transition-all">
                    {/* Header toggle bar for bottom text */}
                    <div 
                        onClick={() => setShowTextSheet(!showTextSheet)}
                        className="flex justify-between items-center px-4 py-2 cursor-pointer hover:bg-white/5 transition-colors"
                    >
                        <div className="flex items-center gap-2">
                            <span className="text-xs font-semibold text-blue-400 uppercase tracking-wider">
                                {t('lens.fullText') || '完整翻譯'}
                            </span>
                            {translatedText && (
                                <span className="text-xs text-gray-400 truncate max-w-xs sm:max-w-md hidden sm:inline">
                                    {translatedText.slice(0, 45)}...
                                </span>
                            )}
                        </div>
                        <div className="flex items-center gap-3">
                            {onSpeak && translatedText && (
                                <button
                                    onClick={(e) => {
                                        e.stopPropagation();
                                        onSpeak(translatedText);
                                    }}
                                    className="flex items-center gap-1 text-xs text-gray-300 hover:text-white px-2 py-1 rounded bg-white/10 hover:bg-white/20 transition-all"
                                    title={t('lens.speak') || '朗讀譯文'}
                                >
                                    <Volume2 className="w-3.5 h-3.5 text-blue-400" />
                                    <span>{t('lens.speak') || '朗讀'}</span>
                                </button>
                            )}
                            <button className="text-gray-400 hover:text-white">
                                {showTextSheet ? <ChevronDown className="w-4 h-4" /> : <ChevronUp className="w-4 h-4" />}
                            </button>
                        </div>
                    </div>

                    {/* Expanded text preview container */}
                    {showTextSheet && (
                        <div className="px-4 py-3 max-h-48 overflow-y-auto space-y-3 bg-black/40 border-t border-white/5 text-xs text-gray-200">
                            {translatedText && (
                                <div>
                                    <p className="text-[10px] text-gray-400 font-semibold uppercase tracking-wider mb-1">
                                        {t('lens.fullText') || '譯文'}
                                    </p>
                                    <p className="whitespace-pre-wrap leading-relaxed select-text font-medium text-white">
                                        {translatedText}
                                    </p>
                                </div>
                            )}
                            {sourceText && (
                                <div className="pt-2 border-t border-white/5">
                                    <p className="text-[10px] text-gray-400 font-semibold uppercase tracking-wider mb-1">
                                        {t('lens.originalText') || '原文'}
                                    </p>
                                    <p className="whitespace-pre-wrap leading-relaxed select-text text-gray-300">
                                        {sourceText}
                                    </p>
                                </div>
                            )}
                        </div>
                    )}
                </div>
            )}
        </div>
    );
};

export default LensOverlayModal;
