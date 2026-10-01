import React, { useRef, useState, useCallback, useEffect } from 'react';
import Webcam from 'react-webcam';
import { useTranslation } from 'react-i18next';
import { 
    X, 
    Zap, 
    ZapOff, 
    Image as ImageIcon, 
    RotateCw, 
    RotateCcw, 
    Scan, 
    Crop, 
    Check, 
    RefreshCw, 
    Smartphone,
    Sparkles,
    Move
} from 'lucide-react';

interface CameraViewProps {
    onClose: () => void;
    onImageCaptured: (imageDataUrl: string, enableLens?: boolean) => void;
    imageFormat?: 'image/webp' | 'image/jpeg';
}

type OrientationAngle = 0 | 90 | 180 | 270;
type OrientationMode = 'auto' | 0 | 90 | 270;
type GuideShape = 'none' | 'doc' | 'strip';

const CameraView: React.FC<CameraViewProps> = ({ onClose, onImageCaptured, imageFormat = 'image/webp' }) => {
    const { t } = useTranslation();
    const webcamRef = useRef<Webcam>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);
    const [stream, setStream] = useState<MediaStream | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [isCapturing, setIsCapturing] = useState(false);
    const isApplyingConstraints = useRef(false);

    // Orientation state
    const [screenAngle, setScreenAngle] = useState<OrientationAngle>(0);
    const [physicalAngle, setPhysicalAngle] = useState<OrientationAngle>(0);
    const physicalAngleRef = useRef<OrientationAngle>(0);
    const candidateAngleRef = useRef<OrientationAngle>(0);
    const candidateCountRef = useRef<number>(0);
    const [orientationMode, setOrientationMode] = useState<OrientationMode>('auto');

    // Guide shape state: default is 'none' (標準文件模式，沒有取景輔助)
    const [guideShape, setGuideShape] = useState<GuideShape>('none');
    const [guideNotification, setGuideNotification] = useState<string | null>(null);
    const guideNotificationTimer = useRef<number | null>(null);

    // Preview state
    const [previewImage, setPreviewImage] = useState<string | null>(null);
    const [rawCapturedImage, setRawCapturedImage] = useState<string | null>(null);
    const [cropMode, setCropMode] = useState<'full' | 'strip' | 'doc'>('full');
    const [imageDimensions, setImageDimensions] = useState<{ width: number; height: number } | null>(null);

    // Interactive Viewfinder Crop Box state (in percentages 0..100)
    const [cropBox, setCropBox] = useState<{ x: number; y: number; w: number; h: number }>({
        x: 12,
        y: 6,
        w: 76,
        h: 88,
    });
    const overlayRef = useRef<HTMLDivElement>(null);
    const dragStartRef = useRef<{
        clientX: number;
        clientY: number;
        boxX: number;
        boxY: number;
        boxW: number;
        boxH: number;
        mode: 'move' | 'nw' | 'ne' | 'se' | 'sw';
    } | null>(null);
    const [isDraggingBox, setIsDraggingBox] = useState(false);

    // Calculate default crop box based on shape and image aspect ratio
    const getDefaultCropBox = useCallback((mode: 'strip' | 'doc', w?: number, h?: number) => {
        const isWide = (w && h) ? w > h : false;
        if (mode === 'strip') {
            if (isWide) {
                return { x: 6, y: 22, w: 88, h: 56 };
            }
            // Tall receipt: 76% width, 88% height, centered horizontally
            return { x: 12, y: 6, w: 76, h: 88 };
        } else {
            // 'doc' standard A4/document
            return { x: 8, y: 10, w: 84, h: 80 };
        }
    }, []);

    // Google Lens overlay translation mode toggle (persisted, default enabled)
    const [isLensEnabled, setIsLensEnabled] = useState<boolean>(() => {
        try {
            const saved = localStorage.getItem('camera-lens-enabled');
            return saved !== null ? JSON.parse(saved) : true;
        } catch {
            return true;
        }
    });

    // Camera hardware controls state
    const [torchOn, setTorchOn] = useState(false);
    const [torchSupported, setTorchSupported] = useState(false);
    
    // Zoom control (Logarithmic)
    const [zoomPercent, setZoomPercent] = useState(0);
    const [zoomSupported, setZoomSupported] = useState(false);
    const [zoomLimits, setZoomLimits] = useState({ min: 1, max: 1 });

    // Helper: read current Screen Orientation angle
    const getScreenAngle = useCallback((): OrientationAngle => {
        if (window.screen?.orientation?.angle !== undefined) {
            const a = window.screen.orientation.angle % 360;
            return (a === 90 || a === 180 || a === 270) ? a as OrientationAngle : 0;
        }
        if (typeof window.orientation === 'number') {
            const a = ((window.orientation % 360) + 360) % 360;
            return (a === 90 || a === 180 || a === 270) ? a as OrientationAngle : 0;
        }
        return 0;
    }, []);

    // 1. Listen for Screen Orientation changes
    useEffect(() => {
        const updateScreenAngle = () => {
            setScreenAngle(getScreenAngle());
        };
        updateScreenAngle();

        if (window.screen?.orientation) {
            window.screen.orientation.addEventListener('change', updateScreenAngle);
        }
        window.addEventListener('orientationchange', updateScreenAngle);

        return () => {
            if (window.screen?.orientation) {
                window.screen.orientation.removeEventListener('change', updateScreenAngle);
            }
            window.removeEventListener('orientationchange', updateScreenAngle);
        };
    }, [getScreenAngle]);

    // 2. Listen for Physical Accelerometer Gravity tilt (DeviceMotionEvent)
    // Uses single-source acceleration vector with hysteresis and temporal debounce.
    // Completely solves 90° vs 270° jumping caused by DeviceOrientation gimbal-lock singularities!
    useEffect(() => {
        let lastUpdateTime = 0;

        const handleDeviceMotion = (e: DeviceMotionEvent) => {
            const now = Date.now();
            if (now - lastUpdateTime < 60) return; // ~16Hz sampling
            lastUpdateTime = now;

            const acc = e.accelerationIncludingGravity;
            if (!acc || acc.x === null || acc.y === null) return;
            const ax = acc.x;
            const ay = acc.y;
            
            // In-plane gravitational acceleration magnitude
            const r = Math.hypot(ax, ay);

            // When phone is nearly horizontal (e.g. held flat looking straight down at a desk),
            // in-plane gravity is too weak and noisy. Retain previous stable angle!
            if (r < 2.0) {
                return;
            }

            // In-plane angle in [0, 360):
            // ax = 0, ay > 0 (gravity down along bottom edge) -> 0° (Portrait)
            // ax > 0, ay = 0 (gravity down along right edge) -> 90° (Landscape-Right)
            // ax < 0, ay = 0 (gravity down along left edge) -> 270° (Landscape-Left)
            // ay < 0 -> 180° (Upside down)
            const angleRad = Math.atan2(ax, ay);
            const deg = ((angleRad * 180 / Math.PI) + 360) % 360;

            const current = physicalAngleRef.current;
            let target: OrientationAngle = current;

            // Hysteresis zones to prevent fluttering
            if (current === 0) {
                // Portrait -> Landscape requires clear tilt
                if (deg >= 55 && deg <= 125) {
                    target = 90;
                } else if (deg >= 235 && deg <= 305) {
                    target = 270;
                } else if (deg >= 150 && deg <= 210) {
                    target = 180;
                }
            } else if (current === 90) {
                // Landscape 90°: to flip to 270°, angle must swing all the way across to > 230°!
                if (deg < 35 || deg > 325) {
                    target = 0;
                } else if (deg >= 230 && deg <= 310) {
                    target = 270;
                } else if (deg >= 150 && deg <= 210) {
                    target = 180;
                }
            } else if (current === 270) {
                // Landscape 270°: to flip to 90°, angle must swing all the way across to < 130°!
                if (deg < 35 || deg > 325) {
                    target = 0;
                } else if (deg >= 50 && deg <= 130) {
                    target = 90;
                } else if (deg >= 150 && deg <= 210) {
                    target = 180;
                }
            } else if (current === 180) {
                if (deg < 35 || deg > 325) {
                    target = 0;
                } else if (deg >= 55 && deg <= 125) {
                    target = 90;
                } else if (deg >= 235 && deg <= 305) {
                    target = 270;
                }
            }

            // Temporal debouncing: candidate state must persist for at least 3 consecutive frames (~180ms)
            if (target !== current) {
                if (candidateAngleRef.current === target) {
                    candidateCountRef.current += 1;
                    if (candidateCountRef.current >= 3) {
                        physicalAngleRef.current = target;
                        setPhysicalAngle(target);
                        candidateCountRef.current = 0;
                    }
                } else {
                    candidateAngleRef.current = target;
                    candidateCountRef.current = 1;
                }
            } else {
                candidateCountRef.current = 0;
            }
        };

        window.addEventListener('devicemotion', handleDeviceMotion, { passive: true });

        return () => {
            window.removeEventListener('devicemotion', handleDeviceMotion);
        };
    }, []);

    // Calculate effective holding angle:
    // If user selected manual mode, use manual mode.
    // If auto: if browser screen itself is rotated (screenAngle !== 0), use screenAngle;
    // otherwise use stable gravity-based physicalAngle.
    const effectiveAngle: OrientationAngle = orientationMode === 'auto'
        ? (screenAngle !== 0 ? screenAngle : physicalAngle)
        : orientationMode;

    const isLandscape = effectiveAngle === 90 || effectiveAngle === 270;

    const getActualZoom = useCallback((percent: number, min: number, max: number) => {
        if (min === max) return min;
        return min * Math.pow(max / min, percent / 100);
    }, []);

    const getPercentFromZoom = useCallback((actual: number, min: number, max: number) => {
        if (min === max) return 0;
        return (Math.log(actual / min) / Math.log(max / min)) * 100;
    }, []);

    const handleUserMedia = useCallback((mediaStream: MediaStream) => {
        setStream(mediaStream);
        const track = mediaStream.getVideoTracks()[0];
        if (!track) return;

        const capabilities = (track.getCapabilities() as any) || {};
        if (capabilities.torch) {
            setTorchSupported(true);
        }
        if (capabilities.zoom) {
            setZoomSupported(true);
            setZoomLimits({ min: capabilities.zoom.min, max: capabilities.zoom.max });
            const currentZoom = (track.getSettings() as any).zoom || capabilities.zoom.min;
            setZoomPercent(getPercentFromZoom(currentZoom, capabilities.zoom.min, capabilities.zoom.max));
        }
    }, [getPercentFromZoom]);

    const handleZoomChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
        if (!stream || !zoomSupported || isApplyingConstraints.current) return;
        
        const newPercent = parseFloat(e.target.value);
        setZoomPercent(newPercent);
        
        const actualZoom = getActualZoom(newPercent, zoomLimits.min, zoomLimits.max);
        const track = stream.getVideoTracks()[0];
        
        try {
            isApplyingConstraints.current = true;
            await track.applyConstraints({
                advanced: [{ zoom: actualZoom }]
            } as any);
        } catch (err) {
            console.error('Failed to apply zoom:', err);
        } finally {
            isApplyingConstraints.current = false;
        }
    };
    
    const handleToggleTorch = async () => {
        if (!stream || !torchSupported || isApplyingConstraints.current) return;
        
        const newTorchState = !torchOn;
        const track = stream.getVideoTracks()[0];
        
        try {
            isApplyingConstraints.current = true;
            await track.applyConstraints({
                advanced: [{ torch: newTorchState }]
            } as any);
            setTorchOn(newTorchState);
        } catch (err) {
            console.error('Failed to apply torch:', err);
        } finally {
            isApplyingConstraints.current = false;
        }
    };

    // Cycle orientation mode: auto -> 90 -> 270 -> 0 -> auto
    const handleCycleOrientation = () => {
        if (orientationMode === 'auto') {
            setOrientationMode(90);
        } else if (orientationMode === 90) {
            setOrientationMode(270);
        } else if (orientationMode === 270) {
            setOrientationMode(0);
        } else {
            setOrientationMode('auto');
        }
    };

    // Cycle framing guide: none -> doc -> strip -> none
    const handleCycleGuide = () => {
        let nextGuide: GuideShape = 'none';
        let label = '';
        if (guideShape === 'none') {
            nextGuide = 'doc';
            label = t('camera.framingGuideDoc') || '標準文件框';
        } else if (guideShape === 'doc') {
            nextGuide = 'strip';
            label = t('camera.framingGuideStrip') || '長條 / 御神籤框';
        } else {
            nextGuide = 'none';
            label = t('camera.framingGuideNone') || '標準模式 (無取景輔助)';
        }
        setGuideShape(nextGuide);
        setGuideNotification(label);

        if (guideNotificationTimer.current) {
            window.clearTimeout(guideNotificationTimer.current);
        }
        guideNotificationTimer.current = window.setTimeout(() => {
            setGuideNotification(null);
        }, 1800);
    };

    // Native resolution capture with automatic orientation correction on canvas
    const handleCapture = useCallback(async () => {
        if (isCapturing || !webcamRef.current) return;

        const video = webcamRef.current.video;
        if (!video || !video.videoWidth || !video.videoHeight) {
            setError(t('camera.errorCapture', { message: 'Video stream not ready' }));
            return;
        }

        setIsCapturing(true);
        setError(null);
        
        try {
            const vw = video.videoWidth;
            const vh = video.videoHeight;

            // Calculate relative rotation angle between target orientation and current stream:
            const relativeAngle = (effectiveAngle - screenAngle + 360) % 360;
            const canvasRotation = (360 - relativeAngle) % 360;

            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            if (!ctx) throw new Error('Could not create canvas 2D context');

            if (canvasRotation === 90) {
                canvas.width = vh;
                canvas.height = vw;
                ctx.translate(canvas.width, 0);
                ctx.rotate(90 * Math.PI / 180);
                ctx.drawImage(video, 0, 0, vw, vh);
            } else if (canvasRotation === 270) {
                canvas.width = vh;
                canvas.height = vw;
                ctx.translate(0, canvas.height);
                ctx.rotate(270 * Math.PI / 180);
                ctx.drawImage(video, 0, 0, vw, vh);
            } else if (canvasRotation === 180) {
                canvas.width = vw;
                canvas.height = vh;
                ctx.translate(canvas.width, canvas.height);
                ctx.rotate(180 * Math.PI / 180);
                ctx.drawImage(video, 0, 0, vw, vh);
            } else {
                canvas.width = vw;
                canvas.height = vh;
                ctx.drawImage(video, 0, 0, vw, vh);
            }

            const fullDataUrl = canvas.toDataURL(imageFormat, 0.92);
            setRawCapturedImage(fullDataUrl);
            setPreviewImage(fullDataUrl);
            setImageDimensions({ width: canvas.width, height: canvas.height });

            // Apply framing guide if one was active in camera view
            if (guideShape !== 'none') {
                setCropMode(guideShape);
                setCropBox(getDefaultCropBox(guideShape, canvas.width, canvas.height));
            } else {
                setCropMode('full');
            }

        } catch (err) {
            const message = err instanceof Error ? err.message : 'Capture failed.';
            setError(t('camera.errorCapture', { message }));
        } finally {
            setIsCapturing(false);
        }
    }, [isCapturing, effectiveAngle, screenAngle, imageFormat, guideShape, getDefaultCropBox, t]);

    // Apply or switch framing crop mode in preview screen (interactive, non-destructive)
    const handleApplyCropMode = useCallback((mode: 'full' | 'strip' | 'doc') => {
        setCropMode(mode);
        if (mode !== 'full') {
            setCropBox(getDefaultCropBox(mode, imageDimensions?.width, imageDimensions?.height));
        }
    }, [getDefaultCropBox, imageDimensions]);

    // Interactive Drag & Resize Handlers for the Viewfinder Crop Box
    const handleCropPointerDown = (
        e: React.PointerEvent<HTMLDivElement>, 
        mode: 'move' | 'nw' | 'ne' | 'se' | 'sw'
    ) => {
        e.preventDefault();
        e.stopPropagation();
        try {
            (e.target as HTMLElement).setPointerCapture?.(e.pointerId);
        } catch {
            // Ignore pointer capture errors
        }
        setIsDraggingBox(true);
        dragStartRef.current = {
            clientX: e.clientX,
            clientY: e.clientY,
            boxX: cropBox.x,
            boxY: cropBox.y,
            boxW: cropBox.w,
            boxH: cropBox.h,
            mode
        };
    };

    const handleCropPointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
        if (!isDraggingBox || !dragStartRef.current || !overlayRef.current) return;
        const rect = overlayRef.current.getBoundingClientRect();
        if (rect.width === 0 || rect.height === 0) return;

        const { clientX, clientY, boxX, boxY, boxW, boxH, mode } = dragStartRef.current;
        const dx = ((e.clientX - clientX) / rect.width) * 100;
        const dy = ((e.clientY - clientY) / rect.height) * 100;

        setCropBox(() => {
            if (mode === 'move') {
                const newX = Math.max(0, Math.min(100 - boxW, +(boxX + dx).toFixed(2)));
                const newY = Math.max(0, Math.min(100 - boxH, +(boxY + dy).toFixed(2)));
                return { x: newX, y: newY, w: boxW, h: boxH };
            }
            let nextX = boxX;
            let nextY = boxY;
            let nextW = boxW;
            let nextH = boxH;

            if (mode === 'se') {
                nextW = Math.max(15, Math.min(100 - boxX, +(boxW + dx).toFixed(2)));
                nextH = Math.max(10, Math.min(100 - boxY, +(boxH + dy).toFixed(2)));
            } else if (mode === 'sw') {
                const rawX = boxX + dx;
                nextX = Math.max(0, Math.min(boxX + boxW - 15, +rawX.toFixed(2)));
                nextW = +(boxW + (boxX - nextX)).toFixed(2);
                nextH = Math.max(10, Math.min(100 - boxY, +(boxH + dy).toFixed(2)));
            } else if (mode === 'ne') {
                nextW = Math.max(15, Math.min(100 - boxX, +(boxW + dx).toFixed(2)));
                const rawY = boxY + dy;
                nextY = Math.max(0, Math.min(boxY + boxH - 10, +rawY.toFixed(2)));
                nextH = +(boxH + (boxY - nextY)).toFixed(2);
            } else if (mode === 'nw') {
                const rawX = boxX + dx;
                nextX = Math.max(0, Math.min(boxX + boxW - 15, +rawX.toFixed(2)));
                nextW = +(boxW + (boxX - nextX)).toFixed(2);
                const rawY = boxY + dy;
                nextY = Math.max(0, Math.min(boxY + boxH - 10, +rawY.toFixed(2)));
                nextH = +(boxH + (boxY - nextY)).toFixed(2);
            }

            return { x: nextX, y: nextY, w: nextW, h: nextH };
        });
    };

    const handleCropPointerUp = (e: React.PointerEvent<HTMLDivElement>) => {
        if (isDraggingBox) {
            try {
                (e.target as HTMLElement).releasePointerCapture?.(e.pointerId);
            } catch {
                // Ignore
            }
            setIsDraggingBox(false);
            dragStartRef.current = null;
        }
    };

    // Quick alignment helper for crop box
    const handleAlignCrop = (alignment: 'center' | 'top' | 'bottom' | 'fullwidth') => {
        setCropBox(prev => {
            if (alignment === 'center') {
                return {
                    ...prev,
                    x: Math.max(0, +((100 - prev.w) / 2).toFixed(2)),
                    y: Math.max(0, +((100 - prev.h) / 2).toFixed(2))
                };
            }
            if (alignment === 'top') {
                return { ...prev, y: 2 };
            }
            if (alignment === 'bottom') {
                return { ...prev, y: Math.max(0, +(100 - prev.h - 2).toFixed(2)) };
            }
            if (alignment === 'fullwidth') {
                return { ...prev, x: 4, w: 92 };
            }
            return prev;
        });
    };

    // Rotate the currently previewed image by 90 degrees (clockwise or counter-clockwise)
    const handleRotatePreview = useCallback((direction: 'cw' | 'ccw') => {
        const baseSrc = rawCapturedImage || previewImage;
        if (!baseSrc) return;

        const img = new Image();
        img.onload = () => {
            const canvas = document.createElement('canvas');
            canvas.width = img.height;
            canvas.height = img.width;
            const ctx = canvas.getContext('2d');
            if (!ctx) return;

            if (direction === 'cw') {
                ctx.translate(canvas.width, 0);
                ctx.rotate(90 * Math.PI / 180);
            } else {
                ctx.translate(0, canvas.height);
                ctx.rotate(-90 * Math.PI / 180);
            }

            ctx.drawImage(img, 0, 0);
            const rotatedDataUrl = canvas.toDataURL(imageFormat, 0.92);
            setRawCapturedImage(rotatedDataUrl);
            setPreviewImage(rotatedDataUrl);
            setImageDimensions({ width: canvas.width, height: canvas.height });

            if (cropMode !== 'full') {
                setCropBox(getDefaultCropBox(cropMode, canvas.width, canvas.height));
            }
        };
        img.src = baseSrc;
    }, [rawCapturedImage, previewImage, cropMode, imageFormat, getDefaultCropBox]);

    // File import from gallery
    const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (!file || isCapturing) return;
        setIsCapturing(true);
        try {
            const reader = new FileReader();
            reader.onload = (e) => {
                const imageDataUrl = e.target?.result as string;
                if (imageDataUrl) {
                    const img = new Image();
                    img.onload = () => {
                        let { width, height } = img;
                        const maxDimension = 1920; // High resolution for OCR
                        
                        if (width > maxDimension || height > maxDimension) {
                            if (width > height) {
                                height = Math.round((height * maxDimension) / width);
                                width = maxDimension;
                            } else {
                                width = Math.round((width * maxDimension) / height);
                                height = maxDimension;
                            }
                        }
                        
                        const canvas = document.createElement('canvas');
                        canvas.width = width;
                        canvas.height = height;
                        const ctx = canvas.getContext('2d');
                        if (ctx) {
                            ctx.drawImage(img, 0, 0, width, height);
                            try {
                                const dataUrl = canvas.toDataURL(imageFormat, 0.92);
                                setRawCapturedImage(dataUrl);
                                setPreviewImage(dataUrl);
                                setCropMode('full');
                                setImageDimensions({ width, height });
                            } catch {
                                const jpegDataUrl = canvas.toDataURL('image/jpeg', 0.92);
                                setRawCapturedImage(jpegDataUrl);
                                setPreviewImage(jpegDataUrl);
                                setCropMode('full');
                                setImageDimensions({ width, height });
                            }
                        } else {
                            setRawCapturedImage(imageDataUrl);
                            setPreviewImage(imageDataUrl);
                            setCropMode('full');
                            setImageDimensions({ width: img.width, height: img.height });
                        }
                        setIsCapturing(false);
                    };
                    img.onerror = () => {
                        setRawCapturedImage(imageDataUrl);
                        setPreviewImage(imageDataUrl);
                        setCropMode('full');
                        setIsCapturing(false);
                    };
                    img.src = imageDataUrl;
                } else {
                    setIsCapturing(false);
                }
            };
            reader.readAsDataURL(file);
        } catch {
            setError(t('camera.errorLoad', { message: 'File read error' }));
            setIsCapturing(false);
        }
    };

    const onImageCapturedRef = useRef(onImageCaptured);
    useEffect(() => {
        onImageCapturedRef.current = onImageCaptured;
    }, [onImageCaptured]);

    const handleConfirm = useCallback(() => {
        const baseSrc = rawCapturedImage || previewImage;
        if (!baseSrc) return;

        if (cropMode === 'full') {
            setTimeout(() => {
                onImageCapturedRef.current(baseSrc, isLensEnabled);
            }, 50);
            return;
        }

        // Accurately crop to user's positioned cropBox on physical resolution
        const img = new Image();
        img.onload = () => {
            const cropX = Math.max(0, Math.round((cropBox.x / 100) * img.width));
            const cropY = Math.max(0, Math.round((cropBox.y / 100) * img.height));
            const cropW = Math.max(10, Math.min(img.width - cropX, Math.round((cropBox.w / 100) * img.width)));
            const cropH = Math.max(10, Math.min(img.height - cropY, Math.round((cropBox.h / 100) * img.height)));

            const canvas = document.createElement('canvas');
            canvas.width = Math.max(1, cropW);
            canvas.height = Math.max(1, cropH);
            const ctx = canvas.getContext('2d');
            if (ctx) {
                ctx.drawImage(img, cropX, cropY, cropW, cropH, 0, 0, cropW, cropH);
                const croppedUrl = canvas.toDataURL(imageFormat, 0.92);
                setTimeout(() => {
                    onImageCapturedRef.current(croppedUrl, isLensEnabled);
                }, 50);
            } else {
                setTimeout(() => {
                    onImageCapturedRef.current(baseSrc, isLensEnabled);
                }, 50);
            }
        };
        img.onerror = () => {
            onImageCapturedRef.current(baseSrc, isLensEnabled);
        };
        img.src = baseSrc;
    }, [rawCapturedImage, previewImage, cropMode, cropBox, isLensEnabled, imageFormat]);

    const handleRetake = useCallback(() => {
        setPreviewImage(null);
        setRawCapturedImage(null);
        setCropMode('full');
        setImageDimensions(null);
    }, []);

    const videoConstraints: MediaTrackConstraints = {
        facingMode: 'environment',
        width: { ideal: 1920 },
        height: { ideal: 1080 }
    };

    return (
        <div className="fixed inset-0 bg-black z-50 flex flex-col items-center justify-center overflow-hidden select-none" role="dialog" aria-modal="true">
            <div className="relative w-full h-full flex items-center justify-center">
                {!previewImage ? (
                    <>
                        <Webcam
                            ref={webcamRef}
                            audio={false}
                            screenshotFormat={imageFormat}
                            screenshotQuality={0.92}
                            videoConstraints={videoConstraints}
                            onUserMedia={handleUserMedia}
                            onUserMediaError={() => setError(t('camera.errorAccess'))}
                            className="w-full h-full object-cover"
                        />
                        <input type="file" accept="image/*" ref={fileInputRef} onChange={handleFileChange} className="hidden" />

                        {/* Framing Guide Overlay (Only rendered if guideShape !== 'none') */}
                        {guideShape !== 'none' && (
                            <div className="absolute inset-0 flex items-center justify-center pointer-events-none z-10">
                                <div 
                                    className={`relative border-2 border-white/80 rounded-2xl shadow-[0_0_0_9999px_rgba(0,0,0,0.38)] transition-all duration-300 ${
                                        guideShape === 'strip' 
                                            ? (isLandscape ? 'w-[85%] h-[40%]' : 'w-[75%] h-[80%]') 
                                            : (isLandscape ? 'w-[75%] h-[65%]' : 'w-[85%] h-[60%]')
                                    }`}
                                >
                                    {/* Corner accents */}
                                    <div className="absolute -top-1 -left-1 w-5 h-5 border-t-4 border-l-4 border-blue-400 rounded-tl-md"></div>
                                    <div className="absolute -top-1 -right-1 w-5 h-5 border-t-4 border-r-4 border-blue-400 rounded-tr-md"></div>
                                    <div className="absolute -bottom-1 -left-1 w-5 h-5 border-b-4 border-l-4 border-blue-400 rounded-bl-md"></div>
                                    <div className="absolute -bottom-1 -right-1 w-5 h-5 border-b-4 border-r-4 border-blue-400 rounded-br-md"></div>
                                    
                                    {/* Center subtle label */}
                                    <div className="absolute top-2 left-1/2 -translate-x-1/2 px-2.5 py-0.5 rounded-full bg-black/60 backdrop-blur-md text-[11px] text-white/90 font-medium tracking-wide">
                                        {guideShape === 'strip' ? (t('camera.framingGuideStrip') || '長條 / 御神籤框') : (t('camera.framingGuideDoc') || '標準文件框')}
                                    </div>
                                </div>
                            </div>
                        )}

                        {/* Guide Mode Change Toast Notification */}
                        {guideNotification && (
                            <div className="absolute top-20 left-1/2 -translate-x-1/2 px-4 py-1.5 rounded-full bg-black/75 backdrop-blur-md text-white text-xs font-semibold border border-white/20 shadow-xl pointer-events-none z-30 animate-fade-in">
                                {guideNotification}
                            </div>
                        )}

                        {/* Viewfinder Controls Overlay */}
                        <div className="absolute inset-0 flex flex-col justify-between items-center p-4 z-20 pointer-events-none">
                            {/* Top Header Controls */}
                            <div className="w-full flex justify-between items-center pointer-events-auto">
                                {/* Orientation Indicator / Mode Selector Pill */}
                                <button 
                                    onClick={handleCycleOrientation}
                                    className="flex items-center gap-2 text-white text-xs font-semibold bg-black/50 backdrop-blur-md px-3.5 py-2 rounded-full shadow-lg border border-white/15 hover:bg-black/70 active:scale-95 transition-all"
                                    title={t('camera.orientationStatus') || '拍攝方向'}
                                >
                                    <Smartphone 
                                        className="w-4 h-4 transition-transform duration-300 text-blue-400"
                                        style={{ transform: `rotate(${effectiveAngle}deg)` }}
                                    />
                                    <span>
                                        {orientationMode === 'auto' ? (
                                            isLandscape 
                                                ? `${t('camera.orientationAuto') || '自動'}: ${effectiveAngle === 90 ? '橫向(90°)' : '橫向(270°)'}`
                                                : `${t('camera.orientationAuto') || '自動'}: 直向`
                                        ) : (
                                            orientationMode === 90 ? (t('camera.orientationLandscapeRight') || '橫向 90°') :
                                            orientationMode === 270 ? (t('camera.orientationLandscapeLeft') || '橫向 270°') :
                                            (t('camera.orientationPortrait') || '直向 0°')
                                        )}
                                    </span>
                                </button>

                                {/* Right action buttons: Guide toggle, Torch, Close */}
                                <div className="flex items-center gap-2.5">
                                    <button 
                                        onClick={handleCycleGuide}
                                        className={`rounded-full p-2.5 border backdrop-blur-md transition-all shadow-lg active:scale-95 ${
                                            guideShape !== 'none' 
                                                ? 'bg-blue-600/85 border-blue-400 text-white' 
                                                : 'bg-black/50 border-white/15 text-white/80 hover:bg-black/70'
                                        }`}
                                        title={
                                            guideShape === 'none' 
                                                ? (t('camera.framingGuideNone') || '標準模式 (無取景輔助)') 
                                                : guideShape === 'doc' 
                                                    ? (t('camera.framingGuideDoc') || '標準文件框') 
                                                    : (t('camera.framingGuideStrip') || '長條 / 御神籤框')
                                        }
                                    >
                                        <Scan className="w-5 h-5" />
                                    </button>

                                    {torchSupported && (
                                        <button 
                                            onClick={handleToggleTorch} 
                                            className={`rounded-full p-2.5 border backdrop-blur-md transition-all shadow-lg active:scale-95 ${
                                                torchOn 
                                                    ? 'bg-amber-400 border-amber-500 text-black' 
                                                    : 'bg-black/50 border-white/15 text-white/80 hover:bg-black/70'
                                            }`} 
                                            aria-label={torchOn ? t('camera.flashOnAriaLabel') : t('camera.flashOffAriaLabel')}
                                        >
                                            {torchOn ? <Zap className="w-5 h-5" /> : <ZapOff className="w-5 h-5" />}
                                        </button>
                                    )}

                                    <button 
                                        onClick={onClose} 
                                        className="text-white bg-black/50 backdrop-blur-md rounded-full p-2.5 border border-white/15 hover:bg-black/70 active:scale-95 transition-all shadow-lg" 
                                        aria-label={t('camera.closeAriaLabel')}
                                    >
                                        <X className="w-5 h-5"/>
                                    </button>
                                </div>
                            </div>

                            {/* Error Alert */}
                            {error && (
                                <div className="bg-red-600 text-white px-4 py-2 rounded-full text-xs font-semibold shadow-xl pointer-events-auto animate-bounce" role="alert">
                                    {error}
                                </div>
                            )}

                            {/* Bottom Controls Bar */}
                            <div className="w-full flex flex-col items-center pb-6 space-y-4 pointer-events-auto">
                                {/* Zoom Slider */}
                                {zoomSupported && (
                                    <div className="w-full max-w-xs flex flex-col items-center space-y-1.5">
                                        <div className="flex justify-between w-full px-2 text-[10px] text-white/80 font-bold uppercase tracking-widest">
                                            <span>{zoomLimits.min.toFixed(0)}x</span>
                                            <span className="bg-blue-600/90 px-2 py-0.5 rounded-full text-white text-xs font-mono shadow-sm">
                                                {getActualZoom(zoomPercent, zoomLimits.min, zoomLimits.max).toFixed(1)}x
                                            </span>
                                            <span>{zoomLimits.max.toFixed(0)}x</span>
                                        </div>
                                        <div className="w-full h-10 flex items-center px-3 bg-black/40 backdrop-blur-md rounded-full border border-white/15">
                                            <input
                                                type="range"
                                                min="0"
                                                max="100"
                                                step="1"
                                                value={zoomPercent}
                                                onChange={handleZoomChange}
                                                className="w-full h-1.5 bg-white/25 rounded-lg appearance-none cursor-pointer accent-blue-500"
                                                aria-label={t('camera.zoomAriaLabel')}
                                            />
                                        </div>
                                    </div>
                                )}

                                {/* Shutter Buttons Row */}
                                <div className="w-full flex justify-around items-center max-w-sm">
                                    {/* Gallery import button */}
                                    <button 
                                        onClick={() => fileInputRef.current?.click()} 
                                        className="p-3.5 bg-black/50 backdrop-blur-md rounded-full border border-white/20 text-white hover:bg-black/70 active:scale-95 transition-all shadow-lg" 
                                        aria-label={t('camera.galleryAriaLabel')}
                                        title={t('camera.galleryAriaLabel') || '從相簿匯入'}
                                    >
                                        <ImageIcon className="w-6 h-6" />
                                    </button>

                                    {/* Capture Shutter Button */}
                                    <button 
                                        onClick={handleCapture} 
                                        disabled={isCapturing}
                                        className="group relative w-20 h-20 flex items-center justify-center disabled:opacity-50"
                                        aria-label={t('camera.captureAriaLabel')}
                                    >
                                        <div className="absolute inset-0 bg-white/20 rounded-full animate-pulse group-hover:bg-white/30"></div>
                                        <div className="relative w-16 h-16 bg-white rounded-full shadow-2xl flex items-center justify-center transition-transform active:scale-90">
                                            {isCapturing ? (
                                                <div className="w-8 h-8 border-4 border-blue-600 border-t-transparent rounded-full animate-spin" role="status" aria-label={t('camera.processingAriaLabel')}></div>
                                            ) : (
                                                <div className="w-13 h-13 border-2 border-black/10 rounded-full flex items-center justify-center">
                                                    <div className="w-11 h-11 bg-white rounded-full"></div>
                                                </div>
                                            )}
                                        </div>
                                    </button>

                                    {/* Quick 90° Manual Rotate Shortcut Button */}
                                    <button 
                                        onClick={handleCycleOrientation}
                                        className="p-3.5 bg-black/50 backdrop-blur-md rounded-full border border-white/20 text-white hover:bg-black/70 active:scale-95 transition-all shadow-lg"
                                        title={t('camera.orientationStatus') || '切換拍攝角度'}
                                    >
                                        <RotateCw 
                                            className="w-6 h-6 transition-transform duration-300 text-white" 
                                            style={{ transform: `rotate(${effectiveAngle}deg)` }}
                                        />
                                    </button>
                                </div>
                            </div>
                        </div>
                    </>
                ) : (
                    /* Captured Photo Preview Screen */
                    <div className="absolute inset-0 flex flex-col items-center justify-between bg-black p-4 z-50">
                        {/* Preview Header: Information & Quick Rotation Buttons */}
                        <div className="w-full flex justify-between items-center z-20">
                            {/* Resolution & Orientation Tag */}
                            <div className="flex items-center gap-2 bg-black/60 backdrop-blur-md px-3 py-1.5 rounded-full border border-white/15 text-xs text-white/90">
                                {imageDimensions && (
                                    <span>
                                        {imageDimensions.width} × {imageDimensions.height}
                                        <span className="ml-1.5 text-blue-400 font-medium">
                                            ({imageDimensions.width >= imageDimensions.height ? '橫向' : '直向'})
                                        </span>
                                    </span>
                                )}
                            </div>

                            {/* Rotation Tools: Rotate CCW, Rotate CW */}
                            <div className="flex items-center gap-2">
                                <button
                                    onClick={() => handleRotatePreview('ccw')}
                                    className="p-2.5 bg-black/60 hover:bg-black/80 backdrop-blur-md text-white rounded-full border border-white/20 active:scale-95 transition-all shadow-lg flex items-center gap-1 text-xs"
                                    title={t('camera.rotateCcw') || '逆時針旋轉 90°'}
                                >
                                    <RotateCcw className="w-4 h-4" />
                                    <span className="hidden sm:inline">↺ 90°</span>
                                </button>
                                <button
                                    onClick={() => handleRotatePreview('cw')}
                                    className="p-2.5 bg-black/60 hover:bg-black/80 backdrop-blur-md text-white rounded-full border border-white/20 active:scale-95 transition-all shadow-lg flex items-center gap-1 text-xs"
                                    title={t('camera.rotateCw') || '順時針旋轉 90°'}
                                >
                                    <RotateCw className="w-4 h-4" />
                                    <span className="hidden sm:inline">↻ 90°</span>
                                </button>
                            </div>
                        </div>

                        {/* Image Preview Area with Interactive Viewfinder */}
                        <div className="relative flex-1 w-full flex items-center justify-center p-2 overflow-hidden">
                            <div className="relative inline-flex items-center justify-center max-w-full max-h-full select-none">
                                <img 
                                    src={rawCapturedImage || previewImage || ''} 
                                    alt="Captured Preview" 
                                    className="max-w-full max-h-[64vh] sm:max-h-[70vh] w-auto h-auto object-contain rounded-lg shadow-2xl block pointer-events-none" 
                                />

                                {/* Interactive Viewfinder Crop Overlay (when cropMode !== 'full') */}
                                {cropMode !== 'full' && (
                                    <div 
                                        ref={overlayRef}
                                        className="absolute inset-0 rounded-lg overflow-hidden select-none pointer-events-auto"
                                        style={{ touchAction: 'none' }}
                                        onPointerMove={handleCropPointerMove}
                                        onPointerUp={handleCropPointerUp}
                                        onPointerCancel={handleCropPointerUp}
                                    >
                                        {/* Crop Box with box-shadow masking outer region */}
                                        <div
                                            className="absolute border-2 border-blue-400 bg-transparent transition-[border-color] duration-150"
                                            style={{
                                                left: `${cropBox.x}%`,
                                                top: `${cropBox.y}%`,
                                                width: `${cropBox.w}%`,
                                                height: `${cropBox.h}%`,
                                                boxShadow: '0 0 0 9999px rgba(0, 0, 0, 0.65)',
                                                cursor: isDraggingBox ? 'grabbing' : 'grab',
                                            }}
                                            onPointerDown={(e) => handleCropPointerDown(e, 'move')}
                                        >
                                            {/* Corner Accent Brackets */}
                                            <div className="absolute -top-1 -left-1 w-3.5 h-3.5 border-t-3 border-l-3 border-cyan-400 pointer-events-none"></div>
                                            <div className="absolute -top-1 -right-1 w-3.5 h-3.5 border-t-3 border-r-3 border-cyan-400 pointer-events-none"></div>
                                            <div className="absolute -bottom-1 -left-1 w-3.5 h-3.5 border-b-3 border-l-3 border-cyan-400 pointer-events-none"></div>
                                            <div className="absolute -bottom-1 -right-1 w-3.5 h-3.5 border-b-3 border-r-3 border-cyan-400 pointer-events-none"></div>

                                            {/* Draggable Corner Resize Handles */}
                                            <div 
                                                className="absolute -top-3.5 -left-3.5 w-7 h-7 flex items-center justify-center cursor-nwse-resize z-20"
                                                onPointerDown={(e) => handleCropPointerDown(e, 'nw')}
                                            >
                                                <div className="w-2.5 h-2.5 bg-cyan-400 rounded-full shadow-md"></div>
                                            </div>
                                            <div 
                                                className="absolute -top-3.5 -right-3.5 w-7 h-7 flex items-center justify-center cursor-nesw-resize z-20"
                                                onPointerDown={(e) => handleCropPointerDown(e, 'ne')}
                                            >
                                                <div className="w-2.5 h-2.5 bg-cyan-400 rounded-full shadow-md"></div>
                                            </div>
                                            <div 
                                                className="absolute -bottom-3.5 -right-3.5 w-7 h-7 flex items-center justify-center cursor-nwse-resize z-20"
                                                onPointerDown={(e) => handleCropPointerDown(e, 'se')}
                                            >
                                                <div className="w-2.5 h-2.5 bg-cyan-400 rounded-full shadow-md"></div>
                                            </div>
                                            <div 
                                                className="absolute -bottom-3.5 -left-3.5 w-7 h-7 flex items-center justify-center cursor-nesw-resize z-20"
                                                onPointerDown={(e) => handleCropPointerDown(e, 'sw')}
                                            >
                                                <div className="w-2.5 h-2.5 bg-cyan-400 rounded-full shadow-md"></div>
                                            </div>

                                            {/* Central Drag Hint Pill */}
                                            <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                                                <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-black/65 backdrop-blur-sm border border-white/20 text-[11px] text-white/90 shadow-md">
                                                    <Move className="w-3 h-3 text-cyan-300" />
                                                    <span>{t('camera.dragHint') || '拖曳框格調整取景範圍'}</span>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                )}
                            </div>
                        </div>

                        {/* Bottom Actions Bar */}
                        <div className="w-full flex flex-col items-center gap-2.5 z-20 pb-2">
                            {/* Framing & Crop Mode Selector (Available for both camera capture & album import) */}
                            <div className="flex items-center gap-1.5 p-1 rounded-full bg-black/60 backdrop-blur-md border border-white/20 shadow-lg text-xs">
                                <span className="px-2 text-[11px] text-gray-300 font-medium flex items-center gap-1">
                                    <Crop className="w-3.5 h-3.5 text-blue-400" />
                                    <span className="hidden sm:inline">{t('camera.framingGuide') || '取景裁切'}:</span>
                                </span>
                                <button 
                                    type="button"
                                    onClick={() => handleApplyCropMode('full')}
                                    className={`px-3 py-1 rounded-full font-medium transition-all active:scale-95 ${
                                        cropMode === 'full' 
                                            ? 'bg-blue-600 text-white shadow-md' 
                                            : 'text-white/80 hover:text-white hover:bg-white/10'
                                    }`}
                                >
                                    {t('camera.cropFull') || '完整圖片'}
                                </button>
                                <button 
                                    type="button"
                                    onClick={() => handleApplyCropMode('strip')}
                                    className={`px-3 py-1 rounded-full font-medium transition-all active:scale-95 flex items-center gap-1 ${
                                        cropMode === 'strip' 
                                            ? 'bg-blue-600 text-white shadow-md' 
                                            : 'text-white/80 hover:text-white hover:bg-white/10'
                                    }`}
                                    title="適用於長條收據、籤詩、帳單，過濾周圍雜訊"
                                >
                                    <span>{t('camera.cropStrip') || '長條/收據框'}</span>
                                </button>
                                <button 
                                    type="button"
                                    onClick={() => handleApplyCropMode('doc')}
                                    className={`px-3 py-1 rounded-full font-medium transition-all active:scale-95 ${
                                        cropMode === 'doc' 
                                            ? 'bg-blue-600 text-white shadow-md' 
                                            : 'text-white/80 hover:text-white hover:bg-white/10'
                                    }`}
                                    title="適用於標準文件、證件、卡片"
                                >
                                    {t('camera.cropDoc') || '文件框'}
                                </button>
                            </div>

                            {/* Quick Align Shortcuts when crop frame is active */}
                            {cropMode !== 'full' && (
                                <div className="flex items-center gap-1.5 px-3 py-1 rounded-full bg-black/50 backdrop-blur-md border border-white/15 text-[11px] text-white/90">
                                    <span className="text-gray-400 mr-1">快速對齊:</span>
                                    <button
                                        type="button"
                                        onClick={() => handleAlignCrop('top')}
                                        className="px-2 py-0.5 rounded bg-white/10 hover:bg-white/20 active:scale-95 transition-all"
                                    >
                                        {t('camera.alignTop') || '置頂'}
                                    </button>
                                    <button
                                        type="button"
                                        onClick={() => handleAlignCrop('center')}
                                        className="px-2 py-0.5 rounded bg-white/10 hover:bg-white/20 active:scale-95 transition-all"
                                    >
                                        {t('camera.alignCenter') || '置中'}
                                    </button>
                                    <button
                                        type="button"
                                        onClick={() => handleAlignCrop('bottom')}
                                        className="px-2 py-0.5 rounded bg-white/10 hover:bg-white/20 active:scale-95 transition-all"
                                    >
                                        {t('camera.alignBottom') || '置底'}
                                    </button>
                                    <button
                                        type="button"
                                        onClick={() => handleAlignCrop('fullwidth')}
                                        className="px-2 py-0.5 rounded bg-white/10 hover:bg-white/20 active:scale-95 transition-all"
                                    >
                                        全寬
                                    </button>
                                </div>
                            )}

                            {/* Google Lens AR Overlay Translation Checkbox */}
                            <label className="flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-black/60 hover:bg-black/80 backdrop-blur-md text-xs font-medium text-white border border-white/20 cursor-pointer shadow-lg select-none transition-all active:scale-98">
                                <input
                                    type="checkbox"
                                    checked={isLensEnabled}
                                    onChange={(e) => {
                                        const nextVal = e.target.checked;
                                        setIsLensEnabled(nextVal);
                                        localStorage.setItem('camera-lens-enabled', JSON.stringify(nextVal));
                                    }}
                                    className="w-4 h-4 text-blue-600 rounded bg-gray-800 border-gray-500 focus:ring-blue-500 focus:ring-offset-0 cursor-pointer"
                                />
                                <Sparkles className="w-3.5 h-3.5 text-amber-400" />
                                <span>{t('camera.lensModeCheckbox') || '智慧鏡頭實景覆蓋翻譯'}</span>
                            </label>

                            {/* Retake / Confirm Buttons */}
                            <div className="w-full flex justify-center items-center gap-6 max-w-sm">
                                <button 
                                    onClick={handleRetake} 
                                    className="flex-1 flex items-center justify-center gap-2 py-3 bg-white/10 hover:bg-white/20 backdrop-blur-md text-white rounded-full border border-white/15 shadow-lg font-medium active:scale-95 transition-all text-sm"
                                >
                                    <RefreshCw className="w-4 h-4" />
                                    <span>{t('camera.retake') || '重拍'}</span>
                                </button>
                                <button 
                                    onClick={handleConfirm} 
                                    className="flex-1 flex items-center justify-center gap-2 py-3 bg-blue-600 hover:bg-blue-500 backdrop-blur-md text-white rounded-full shadow-xl font-medium active:scale-95 transition-all text-sm"
                                >
                                    <Check className="w-4 h-4" />
                                    <span>{t('camera.usePhoto') || '確認使用'}</span>
                                </button>
                            </div>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
};

export default CameraView;
