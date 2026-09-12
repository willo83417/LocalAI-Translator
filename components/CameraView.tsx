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
    Smartphone
} from 'lucide-react';

interface CameraViewProps {
    onClose: () => void;
    onImageCaptured: (imageDataUrl: string) => void;
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
    const [croppedPreviewImage, setCroppedPreviewImage] = useState<string | null>(null);
    const [isShowingCropped, setIsShowingCropped] = useState(false);
    const [imageDimensions, setImageDimensions] = useState<{ width: number; height: number } | null>(null);

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
    const handleCapture = useCallback(() => {
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

            // Compensate rotation to make the image upright:
            // When camera is rotated clockwise by alpha, world tilts counter-clockwise.
            // Correcting it requires rotating clockwise by (360 - alpha) % 360.
            // relativeAngle 90° -> canvasRotation 270° (produces upright image!)
            // relativeAngle 270° -> canvasRotation 90°
            // relativeAngle 180° -> canvasRotation 180°
            // relativeAngle 0° -> canvasRotation 0°
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

            // Generate cropped version if framing guide was enabled
            if (guideShape !== 'none') {
                const cropCanvas = document.createElement('canvas');
                const cropCtx = cropCanvas.getContext('2d');
                if (cropCtx) {
                    let cropWRatio = 0.85;
                    let cropHRatio = 0.45;
                    if (guideShape === 'strip') {
                        // For long strips / omikuji
                        if (canvas.width > canvas.height) {
                            cropWRatio = 0.90;
                            cropHRatio = 0.40;
                        } else {
                            cropWRatio = 0.70;
                            cropHRatio = 0.85;
                        }
                    } else {
                        // Standard document shape
                        cropWRatio = 0.80;
                        cropHRatio = 0.65;
                    }

                    const cropW = Math.round(canvas.width * cropWRatio);
                    const cropH = Math.round(canvas.height * cropHRatio);
                    const cropX = Math.round((canvas.width - cropW) / 2);
                    const cropY = Math.round((canvas.height - cropH) / 2);

                    cropCanvas.width = cropW;
                    cropCanvas.height = cropH;
                    cropCtx.drawImage(canvas, cropX, cropY, cropW, cropH, 0, 0, cropW, cropH);

                    const croppedDataUrl = cropCanvas.toDataURL(imageFormat, 0.92);
                    setCroppedPreviewImage(croppedDataUrl);
                    setIsShowingCropped(false);
                }
            } else {
                setCroppedPreviewImage(null);
                setIsShowingCropped(false);
            }

        } catch (err) {
            const message = err instanceof Error ? err.message : 'Capture failed.';
            setError(t('camera.errorCapture', { message }));
        } finally {
            setIsCapturing(false);
        }
    }, [isCapturing, effectiveAngle, screenAngle, imageFormat, guideShape, t]);

    // Rotate the currently previewed image by 90 degrees (clockwise or counter-clockwise)
    const handleRotatePreview = useCallback((direction: 'cw' | 'ccw') => {
        if (!previewImage) return;

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
            setPreviewImage(rotatedDataUrl);
            setImageDimensions({ width: canvas.width, height: canvas.height });
        };
        img.src = previewImage;
    }, [previewImage, imageFormat]);

    // Toggle between cropped and full image in preview
    const handleToggleCropView = useCallback(() => {
        if (!croppedPreviewImage || !rawCapturedImage) return;

        if (isShowingCropped) {
            setPreviewImage(rawCapturedImage);
            setIsShowingCropped(false);
            const img = new Image();
            img.onload = () => setImageDimensions({ width: img.width, height: img.height });
            img.src = rawCapturedImage;
        } else {
            setPreviewImage(croppedPreviewImage);
            setIsShowingCropped(true);
            const img = new Image();
            img.onload = () => setImageDimensions({ width: img.width, height: img.height });
            img.src = croppedPreviewImage;
        }
    }, [isShowingCropped, croppedPreviewImage, rawCapturedImage]);

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
                                setPreviewImage(dataUrl);
                                setRawCapturedImage(dataUrl);
                                setCroppedPreviewImage(null);
                                setIsShowingCropped(false);
                                setImageDimensions({ width, height });
                            } catch {
                                const jpegDataUrl = canvas.toDataURL('image/jpeg', 0.92);
                                setPreviewImage(jpegDataUrl);
                                setRawCapturedImage(jpegDataUrl);
                                setCroppedPreviewImage(null);
                                setIsShowingCropped(false);
                                setImageDimensions({ width, height });
                            }
                        } else {
                            setPreviewImage(imageDataUrl);
                            setRawCapturedImage(imageDataUrl);
                            setImageDimensions({ width: img.width, height: img.height });
                        }
                        setIsCapturing(false);
                    };
                    img.onerror = () => {
                        setPreviewImage(imageDataUrl);
                        setRawCapturedImage(imageDataUrl);
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
        if (previewImage) {
            setTimeout(() => {
                onImageCapturedRef.current(previewImage);
            }, 50);
        }
    }, [previewImage]);

    const handleRetake = useCallback(() => {
        setPreviewImage(null);
        setRawCapturedImage(null);
        setCroppedPreviewImage(null);
        setIsShowingCropped(false);
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

                        {/* Image Preview Area */}
                        <div className="relative flex-1 w-full flex items-center justify-center p-2 overflow-hidden">
                            <img 
                                src={previewImage} 
                                alt="Captured Preview" 
                                className="max-w-full max-h-full object-contain rounded-lg shadow-2xl transition-all duration-200" 
                            />
                        </div>

                        {/* Bottom Actions Bar */}
                        <div className="w-full flex flex-col items-center gap-3 z-20 pb-2">
                            {/* Optional Crop to Guide Toggle (only if captured with a guide) */}
                            {croppedPreviewImage && (
                                <button 
                                    onClick={handleToggleCropView}
                                    className="flex items-center gap-2 px-4 py-2 rounded-full bg-white/15 hover:bg-white/25 backdrop-blur-md text-xs font-medium text-white border border-white/20 active:scale-95 transition-all"
                                >
                                    <Crop className="w-3.5 h-3.5 text-blue-400" />
                                    <span>
                                        {isShowingCropped 
                                            ? (t('camera.showOriginal') || '切換為完整圖片') 
                                            : (t('camera.cropToFrame') || '切換為取景框裁切範圍')}
                                    </span>
                                </button>
                            )}

                            {/* Retake / Confirm Buttons */}
                            <div className="w-full flex justify-center items-center gap-6 max-w-sm">
                                <button 
                                    onClick={handleRetake} 
                                    className="flex-1 flex items-center justify-center gap-2 py-3.5 bg-white/10 hover:bg-white/20 backdrop-blur-md text-white rounded-full border border-white/15 shadow-lg font-medium active:scale-95 transition-all"
                                >
                                    <RefreshCw className="w-4 h-4" />
                                    <span>{t('camera.retake') || '重拍'}</span>
                                </button>
                                <button 
                                    onClick={handleConfirm} 
                                    className="flex-1 flex items-center justify-center gap-2 py-3.5 bg-blue-600 hover:bg-blue-500 backdrop-blur-md text-white rounded-full shadow-xl font-medium active:scale-95 transition-all"
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
