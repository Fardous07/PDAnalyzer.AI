// frontend/src/pages/AnalysisPage.jsx
import React, { useEffect, useMemo, useRef, useState, useCallback } from "react";
import { useNavigate, useParams } from "react-router-dom";
import {
  ArrowLeft,
  Award,
  BarChart3,
  ChevronDown,
  ChevronRight,
  ChevronUp,
  Copy,
  Download,
  FileText,
  Filter,
  Headphones,
  Layers,
  Lock,
  Maximize2,
  Minimize2,
  MessageSquare,
  Pause,
  Play,
  PlayCircle,
  Scale,
  Search,
  Shield,
  SkipBack,
  SkipForward,
  Sparkles,
  Target,
  User,
  Volume2,
  VolumeX,
  X,
  Zap,
  Check,
  AlertTriangle,
  CheckCircle,
  Info,
} from "lucide-react";

import apiClient, { getMediaUrl, getSpeech, getSpeechFull } from "../services/api";

const CENTRIST = "Centrist";
const LIB = "Libertarian";
const AUTH = "Authoritarian";

const normalizeFamily = (fam) => {
  const f = String(fam || "").trim();
  if (!f) return CENTRIST;
  if (f === LIB || f === AUTH || f === CENTRIST) return f;
  if (f === "Neutral") return CENTRIST;
  return CENTRIST;
};

const normalizeSubtype = (family, subtype) => {
  const fam = normalizeFamily(family);
  if (fam === CENTRIST) return null;
  const s = String(subtype || "").trim();
  return s || null;
};

const normalizeEvidenceItem = (item) => {
  if (!item || typeof item !== "object") return item;
  const fam = normalizeFamily(item.ideology_family);
  return {
    ...item,
    ideology_family: fam,
    ideology_subtype: normalizeSubtype(fam, item.ideology_subtype),
  };
};

const clamp = (n, min, max) => Math.max(min, Math.min(max, n));

const formatClock = (seconds) => {
  if (seconds === null || seconds === undefined || Number.isNaN(seconds)) return "0:00";
  const s = Math.max(0, Number(seconds));
  const mins = Math.floor(s / 60);
  const secs = Math.floor(s % 60);
  return `${mins}:${String(secs).padStart(2, "0")}`;
};

const formatTimeFull = (seconds) => {
  if (seconds === null || seconds === undefined || Number.isNaN(seconds)) return "0:00";
  const s = Math.max(0, Number(seconds));
  const hours = Math.floor(s / 3600);
  const mins = Math.floor((s % 3600) / 60);
  const secs = Math.floor(s % 60);
  if (hours > 0) return `${hours}:${String(mins).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
  return `${mins}:${String(secs).padStart(2, "0")}`;
};

const unwrap = (x) => {
  if (!x) return x;

  if (x?.data !== undefined && x?.status !== undefined) return unwrap(x.data);

  if (x && typeof x === "object" && "success" in x) {
    if (x.success === false) return x;
    if ("data" in x) return x.data;
    return x;
  }

  const ax = x?.data;
  if (ax && typeof ax === "object") {
    if (ax?.data !== undefined) return ax.data;
    return ax;
  }

  return x;
};

const unwrapArray = (x, keys = []) => {
  const v = unwrap(x);
  if (Array.isArray(v)) return v;
  if (v && typeof v === "object") {
    for (const k of keys) {
      const maybe = v?.[k];
      if (Array.isArray(maybe)) return maybe;
    }
  }
  return [];
};

const normalizeOverview = (analysisData, fallbackSpeech) => {
  const a = unwrap(analysisData) || {};
  const sl = a?.speech_level || null;
  const nestedScores = sl?.scores || {};

  const lib =
    Number(
      a?.libertarian_score ??
        nestedScores?.Libertarian ??
        fallbackSpeech?.analysis_summary?.scores?.Libertarian ??
        0
    ) || 0;

  const auth =
    Number(
      a?.authoritarian_score ??
        nestedScores?.Authoritarian ??
        fallbackSpeech?.analysis_summary?.scores?.Authoritarian ??
        0
    ) || 0;

  const centrist =
    Number(
      a?.centrist_score ??
        nestedScores?.Centrist ??
        nestedScores?.Neutral ??
        fallbackSpeech?.analysis_summary?.scores?.Centrist ??
        fallbackSpeech?.analysis_summary?.scores?.Neutral ??
        0
    ) || 0;

  const rawFam =
    a?.ideology_family ||
    sl?.dominant_family ||
    a?.dominant_family ||
    fallbackSpeech?.analysis_summary?.ideology_family ||
    CENTRIST;

  const ideology_family = normalizeFamily(rawFam);

  const rawSub =
    a?.ideology_subtype ||
    sl?.dominant_subtype ||
    a?.dominant_subtype ||
    fallbackSpeech?.analysis_summary?.ideology_subtype ||
    null;

  const ideology_subtype = normalizeSubtype(ideology_family, rawSub);

  return {
    ideology_family,
    ideology_subtype,
    libertarian_score: lib,
    authoritarian_score: auth,
    centrist_score: centrist,
    confidence_score:
      Number(a?.confidence_score ?? sl?.confidence_score ?? fallbackSpeech?.analysis_summary?.confidence_score ?? 0) || 0,
    marpor_codes: a?.marpor_codes || sl?.marpor_codes || fallbackSpeech?.analysis_summary?.marpor_codes || [],
    _analysis: a,
  };
};

const getIdeologyConfig = (ideology) => {
  const configs = {
    [LIB]: {
      color: "bg-emerald-100 text-emerald-800 border-emerald-200",
      gradient: "bg-gradient-to-r from-emerald-500 via-emerald-400 to-emerald-300",
      icon: Shield,
      iconColor: "text-emerald-500",
      bgGradient: "from-emerald-50/30 to-emerald-50/10",
      borderColor: "border-emerald-200",
      ringColor: "ring-emerald-100",
      description: "Emphasizes individual freedom, limited government, and personal autonomy.",
    },
    [AUTH]: {
      color: "bg-rose-100 text-rose-800 border-rose-200",
      gradient: "bg-gradient-to-r from-rose-500 via-rose-400 to-rose-300",
      icon: Lock,
      iconColor: "text-rose-500",
      bgGradient: "from-rose-50/30 to-rose-50/10",
      borderColor: "border-rose-200",
      ringColor: "ring-rose-100",
      description: "Supports strong centralized authority, social order, and collective discipline.",
    },
    [CENTRIST]: {
      color: "bg-blue-100 text-blue-800 border-blue-200",
      gradient: "bg-gradient-to-r from-blue-500 via-blue-400 to-blue-300",
      icon: Scale,
      iconColor: "text-blue-500",
      bgGradient: "from-blue-50/30 to-blue-50/10",
      borderColor: "border-blue-200",
      ringColor: "ring-blue-100",
      description: "Balanced or mixed ideological positioning; no strong dominant ideological evidence detected.",
    },
  };

  return (
    configs[normalizeFamily(ideology)] || {
      color: "bg-slate-100 text-slate-800 border-slate-200",
      gradient: "bg-gradient-to-r from-slate-500 via-slate-400 to-slate-300",
      icon: BarChart3,
      iconColor: "text-slate-500",
      bgGradient: "from-slate-50/30 to-slate-50/10",
      borderColor: "border-slate-200",
      ringColor: "ring-slate-100",
      description: "Political ideology classification",
    }
  );
};

const getSubtypesFromAnalysis = (overview, ideologyFamily, keyStatements = [], evidenceSections = []) => {
  const fam = normalizeFamily(ideologyFamily);
  if (fam === CENTRIST) return {};

  const present = new Set();

  [...(keyStatements || []), ...(evidenceSections || [])]
    .map(normalizeEvidenceItem)
    .forEach((x) => {
      if (!x) return;
      if (normalizeFamily(x.ideology_family) !== fam) return;
      const st = normalizeSubtype(fam, x.ideology_subtype);
      if (st) present.add(st);
    });

  if (present.size === 0) return {};

  const a = overview?._analysis || overview || {};
  const sl = a?.speech_level || {};
  const subscores = sl?.subscores || a?.subscores || null;
  const direct = subscores?.[fam];

  const raw =
    direct && typeof direct === "object" && !Array.isArray(direct)
      ? direct
      : sl?.subtype_breakdown || a?.subtype_breakdown || sl?.subtypes || a?.subtypes || {};

  if (!raw || typeof raw !== "object" || Array.isArray(raw)) return {};

  const filtered = {};
  for (const [k, v] of Object.entries(raw)) {
    if (present.has(k)) filtered[k] = v;
  }
  return filtered;
};

const LoadingScreen = () => (
  <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 flex items-center justify-center p-4">
    <div className="max-w-md w-full text-center">
      <div className="relative w-24 h-24 mx-auto mb-6">
        <div className="absolute inset-0 bg-gradient-to-r from-indigo-500 to-purple-500 rounded-full blur-xl opacity-20 animate-pulse" />
        <div className="absolute inset-4 border-4 border-transparent rounded-full border-t-indigo-500 border-r-purple-500 animate-spin" />
        <Sparkles className="absolute inset-0 m-auto w-8 h-8 text-indigo-400 animate-pulse" />
      </div>
      <div className="space-y-4">
        <h2 className="text-2xl font-bold bg-gradient-to-r from-white to-gray-300 bg-clip-text text-transparent">
          Loading Analysis
        </h2>
        <p className="text-gray-400 text-sm">Fetching transcript, media, and results...</p>
        <div className="h-1 bg-gray-800 rounded-full overflow-hidden">
          <div className="h-full bg-gradient-to-r from-indigo-500 to-purple-500 animate-shimmer w-1/2" />
        </div>
      </div>
    </div>
    <style>{`
      @keyframes shimmer {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(200%); }
      }
      .animate-shimmer {
        animation: shimmer 1.5s infinite;
      }
    `}</style>
  </div>
);

const ErrorScreen = ({ error, onBack }) => (
  <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 flex items-center justify-center p-4">
    <div className="max-w-md w-full bg-gradient-to-b from-gray-800/50 to-gray-900/50 backdrop-blur-xl rounded-2xl border border-gray-700/50 shadow-2xl p-8 text-center">
      <div className="w-20 h-20 bg-gradient-to-br from-red-900/20 to-rose-900/20 rounded-full flex items-center justify-center mx-auto mb-6 ring-4 ring-red-900/20">
        <div className="w-12 h-12 bg-gradient-to-br from-red-500 to-rose-500 rounded-full flex items-center justify-center">
          <X className="w-6 h-6 text-white" />
        </div>
      </div>
      <h2 className="text-2xl font-bold bg-gradient-to-r from-red-400 to-rose-400 bg-clip-text text-transparent mb-3">
        Analysis Failed
      </h2>
      <p className="text-gray-300 mb-2">We couldn't load the analysis for this speech.</p>
      <p className="text-gray-400 text-sm mb-8">{error}</p>
      <button
        onClick={onBack}
        className="px-8 py-3 bg-gradient-to-r from-gray-700 to-gray-800 hover:from-gray-600 hover:to-gray-700 text-white rounded-xl font-semibold transition-all duration-300 hover:scale-105 hover:shadow-xl border border-gray-600/50"
      >
        Return to Dashboard
      </button>
    </div>
  </div>
);

const AnalysisHeader = ({ speech, onBack }) => {
  const [isScrolled, setIsScrolled] = useState(false);

  useEffect(() => {
    const handleScroll = () => setIsScrolled(window.scrollY > 20);
    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  return (
    <div
      className={`sticky top-0 z-50 transition-all duration-300 ${
        isScrolled
          ? "bg-gradient-to-r from-gray-900/95 via-gray-900/95 to-gray-900/95 backdrop-blur-xl border-b border-gray-800/50 shadow-2xl"
          : "bg-gradient-to-r from-gray-900 via-gray-900 to-gray-900"
      }`}
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <button
              onClick={onBack}
              className="group flex items-center gap-2 text-gray-300 hover:text-white transition-all duration-300 px-3 py-2 rounded-xl hover:bg-gray-800/50"
            >
              <ArrowLeft className="w-5 h-5 group-hover:-translate-x-1 transition-transform" />
              <span className="font-semibold">Dashboard</span>
            </button>
            <div className="h-6 w-px bg-gray-700/50" />
            <div className="flex items-center gap-2">
              <FileText className="w-5 h-5 text-indigo-400" />
              <span className="text-sm text-gray-400">Analysis</span>
            </div>
            {speech?.speaker && (
              <>
                <div className="h-6 w-px bg-gray-700/50" />
                <div className="flex items-center gap-2 px-3 py-1.5 bg-gray-800/50 rounded-lg border border-gray-700/50">
                  <User className="w-4 h-4 text-gray-400" />
                  <span className="text-sm text-gray-300">{speech.speaker}</span>
                </div>
              </>
            )}
          </div>

          <div className="flex-1 mx-8" />

          <div className="flex items-center gap-6">
            {speech?.word_count ? (
              <div className="text-right">
                <p className="text-sm text-gray-400">Word Count</p>
                <p className="text-lg font-bold text-white">{speech.word_count.toLocaleString()}</p>
              </div>
            ) : null}

            {speech?.status ? (
              <div className="text-right">
                <p className="text-sm text-gray-400">Status</p>
                <div className="flex items-center gap-2">
                  <div
                    className={`w-2 h-2 rounded-full ${
                      speech.status === "completed"
                        ? "bg-emerald-500"
                        : speech.status === "processing"
                        ? "bg-yellow-500"
                        : "bg-gray-500"
                    }`}
                  />
                  <p className="text-sm font-semibold text-white capitalize">{speech.status}</p>
                </div>
              </div>
            ) : null}

            <div className="flex items-center gap-2">
              <button
                className="p-2 text-gray-400 hover:text-white hover:bg-gray-800/50 rounded-xl transition-all"
                type="button"
                title="Download (placeholder)"
              >
                <Download className="w-5 h-5" />
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

const ConfidenceTierBadge = ({ level }) => {
  const config = {
    high: {
      bg: "bg-gradient-to-r from-emerald-900/20 to-emerald-800/20",
      border: "border-emerald-700/50",
      text: "text-emerald-300",
      icon: CheckCircle,
    },
    medium: {
      bg: "bg-gradient-to-r from-yellow-900/20 to-yellow-800/20",
      border: "border-yellow-700/50",
      text: "text-yellow-300",
      icon: Info,
    },
    low: {
      bg: "bg-gradient-to-r from-rose-900/20 to-rose-800/20",
      border: "border-rose-700/50",
      text: "text-rose-300",
      icon: AlertTriangle,
    },
  };

  const tier = (level || "").toLowerCase();
  const currentConfig = config[tier] || config.medium;
  const Icon = currentConfig.icon;

  return (
    <div className={`flex items-center gap-2 px-3 py-2 rounded-lg border ${currentConfig.border} ${currentConfig.bg}`}>
      <Icon className={`w-4 h-4 ${currentConfig.text}`} />
      <span className={`text-sm font-bold ${currentConfig.text}`}>{tier ? tier.toUpperCase() : "MEDIUM"}</span>
    </div>
  );
};

const PrimaryClassificationCard = ({ overview }) => {
  const fam = normalizeFamily(overview?.ideology_family || CENTRIST);
  const config = getIdeologyConfig(fam);
  const Icon = config.icon;

  const scientificSummary =
    overview?._analysis?.scientific_summary ||
    overview?._analysis?.speech_level?.scientific_summary ||
    {};

  const overallConfidence = scientificSummary.overall_confidence || "medium";
  const confidenceScore = Math.round((overview?.confidence_score || 0) * 100);

  return (
    <div className="bg-gradient-to-br from-gray-800/50 to-gray-900/50 backdrop-blur-xl rounded-2xl border border-gray-700/50 p-6 hover:border-gray-600/50 transition-all hover:shadow-2xl">
      <div className="flex flex-col md:flex-row md:items-center justify-between mb-6 gap-4">
        <div className="flex items-center gap-4">
          <div className="p-3 bg-gradient-to-br from-gray-800 to-gray-900 rounded-xl border border-gray-700/50">
            <Award className="w-7 h-7 text-yellow-400" />
          </div>
          <div>
            <h2 className="text-xl font-bold text-white">Primary Classification</h2>
            <p className="text-gray-400">Dominant classification with evidence-weighted scoring</p>
          </div>
        </div>
        <div className="flex flex-col items-end gap-3">
          <div className="flex items-center gap-4">
            <div className="text-right">
              <div className="flex items-center gap-2 justify-end">
                <span className="text-sm text-gray-400">Scientific Confidence</span>
              </div>
              <div className="flex items-center gap-3 mt-1">
                <ConfidenceTierBadge level={overallConfidence} />
                <span className="text-md font-bold text-white">{confidenceScore}% Confidence</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div>
          <div className="flex flex-col sm:flex-row items-start sm:items-center gap-4 mb-6">
            <div className={`p-5 rounded-2xl ${config.bgGradient} border ${config.borderColor}`}>
              <Icon className="w-10 h-10 text-white" />
            </div>
            <div>
              <p className="text-sm text-gray-400 mb-1">Dominant Classification</p>
              <h2 className="text-3xl font-bold text-white mb-2">{fam}</h2>
              {overview?.ideology_subtype ? <p className="text-md text-gray-300">{overview.ideology_subtype}</p> : null}
              <p className="text-gray-400 mt-3 max-w-xl text-sm">{config.description}</p>
            </div>
          </div>
        </div>

        <div className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div className="bg-gradient-to-br from-emerald-900/20 to-emerald-800/20 p-4 rounded-xl border border-emerald-700/30">
              <div className="flex items-center gap-2 mb-2">
                <Shield className="w-4 h-4 text-emerald-400" />
                <span className="text-sm text-gray-300">Libertarian</span>
              </div>
              <div className="text-2xl font-bold text-white">{overview?.libertarian_score?.toFixed?.(1) || "0.0"}%</div>
              <div className="h-2 bg-gray-800/50 rounded-full overflow-hidden mt-2">
                <div
                  className="h-full bg-gradient-to-r from-emerald-500 to-emerald-400 rounded-full"
                  style={{ width: `${overview?.libertarian_score || 0}%` }}
                />
              </div>
            </div>

            <div className="bg-gradient-to-br from-rose-900/20 to-rose-800/20 p-4 rounded-xl border border-rose-700/30">
              <div className="flex items-center gap-2 mb-2">
                <Lock className="w-4 h-4 text-rose-400" />
                <span className="text-sm text-gray-300">Authoritarian</span>
              </div>
              <div className="text-2xl font-bold text-white">{overview?.authoritarian_score?.toFixed?.(1) || "0.0"}%</div>
              <div className="h-2 bg-gray-800/50 rounded-full overflow-hidden mt-2">
                <div
                  className="h-full bg-gradient-to-r from-rose-500 to-rose-400 rounded-full"
                  style={{ width: `${overview?.authoritarian_score || 0}%` }}
                />
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

const EnhancedMediaPlayer = ({
  mediaUrl,
  isVideo,
  keyStatements,
  transcriptText,
  onJumpToTime,
  currentTime,
  duration,
  isPlaying,
  onSetIsPlaying,
  mediaRef,
  onSetDuration,
  onSetCurrentTime,
}) => {
  const [volume, setVolume] = useState(1);
  const [isMuted, setIsMuted] = useState(false);
  const [playbackRate, setPlaybackRate] = useState(1);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const containerRef = useRef(null);
  const progressRef = useRef(null);
  const draggingRef = useRef(false);

  useEffect(() => {
    const onFsChange = () => setIsFullscreen(Boolean(document.fullscreenElement));
    document.addEventListener("fullscreenchange", onFsChange);
    return () => document.removeEventListener("fullscreenchange", onFsChange);
  }, []);

  const safePlay = async () => {
    const el = mediaRef.current;
    if (!el) return;
    try {
      await el.play();
      onSetIsPlaying(true);
    } catch {
      onSetIsPlaying(false);
    }
  };

  const togglePlay = async () => {
    const el = mediaRef.current;
    if (!el) return;
    if (isPlaying) {
      el.pause();
      onSetIsPlaying(false);
    } else {
      await safePlay();
    }
  };

  const handleTimeUpdate = () => {
    if (mediaRef.current) onSetCurrentTime(mediaRef.current.currentTime || 0);
  };

  const handleLoadedMetadata = () => {
    if (mediaRef.current) onSetDuration(Number(mediaRef.current.duration || 0));
    if (mediaRef.current) mediaRef.current.playbackRate = playbackRate;
  };

  const setTimeByClientX = (clientX) => {
    if (!progressRef.current || !mediaRef.current || !duration) return;
    const rect = progressRef.current.getBoundingClientRect();
    const pos = clamp((clientX - rect.left) / rect.width, 0, 1);
    const t = pos * duration;
    mediaRef.current.currentTime = t;
    onSetCurrentTime(t);
  };

  const onMouseDown = (e) => {
    draggingRef.current = true;
    setTimeByClientX(e.clientX);

    const onMove = (ev) => {
      if (!draggingRef.current) return;
      setTimeByClientX(ev.clientX);
    };
    const onUp = () => {
      draggingRef.current = false;
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };

    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
  };

  const skip = (secs) => {
    if (!mediaRef.current || !duration) return;
    const t = clamp((mediaRef.current.currentTime || 0) + secs, 0, duration);
    mediaRef.current.currentTime = t;
    onSetCurrentTime(t);
  };

  const toggleMute = () => {
    if (!mediaRef.current) return;
    const next = !isMuted;
    mediaRef.current.muted = next;
    setIsMuted(next);
  };

  const handleVolumeChange = (e) => {
    const v = Number(e.target.value);
    setVolume(v);
    if (mediaRef.current) {
      mediaRef.current.volume = v;
      if (v > 0 && isMuted) {
        mediaRef.current.muted = false;
        setIsMuted(false);
      }
    }
  };

  const changePlaybackRate = (rate) => {
    setPlaybackRate(rate);
    if (mediaRef.current) mediaRef.current.playbackRate = rate;
  };

  const toggleFullscreen = async () => {
    const node = containerRef.current;
    try {
      if (!document.fullscreenElement) {
        await node?.requestFullscreen?.();
      } else {
        await document.exitFullscreen?.();
      }
    } catch {
      // ignore
    }
  };

  const progressPct = duration > 0 ? (currentTime / duration) * 100 : 0;

  const markerFractions = useMemo(() => {
    const ks = Array.isArray(keyStatements) ? keyStatements : [];
    if (ks.length === 0) return [];
    const totalChars = transcriptText?.length || 0;

    if (duration > 0 && ks.some((k) => Number.isFinite(Number(k?.time_begin)))) {
      const fr = ks
        .map((k, idx) => {
          const t = Number(k?.time_begin);
          if (Number.isFinite(t) && t >= 0) return { idx, frac: clamp(t / duration, 0, 1) };
          return { idx, frac: (idx + 1) / (ks.length + 1) };
        })
        .sort((a, b) => a.frac - b.frac);

      const out = [];
      for (const m of fr) {
        if (!out.length || Math.abs(out[out.length - 1].frac - m.frac) > 0.01) out.push(m);
      }
      return out;
    }

    if (totalChars > 0 && ks.some((k) => Number.isFinite(Number(k?.start_char)))) {
      const fr = ks
        .map((k, idx) => {
          const sc = Number(k?.start_char);
          if (Number.isFinite(sc)) return { idx, frac: clamp(sc / totalChars, 0, 1) };
          return { idx, frac: (idx + 1) / (ks.length + 1) };
        })
        .sort((a, b) => a.frac - b.frac);

      const out = [];
      for (const m of fr) {
        if (!out.length || Math.abs(out[out.length - 1].frac - m.frac) > 0.01) out.push(m);
      }
      return out;
    }

    return ks.map((_, idx) => ({ idx, frac: (idx + 1) / (ks.length + 1) }));
  }, [keyStatements, transcriptText, duration]);

  return (
    <div
      ref={containerRef}
      className="bg-gradient-to-br from-gray-800/50 to-gray-900/50 backdrop-blur-xl rounded-2xl border border-gray-700/50 overflow-hidden hover:shadow-2xl transition-all"
    >
      <div className="relative aspect-video bg-gradient-to-br from-gray-900 to-black">
        {isVideo ? (
          <video
            ref={mediaRef}
            src={mediaUrl}
            className="w-full h-full object-contain"
            onTimeUpdate={handleTimeUpdate}
            onLoadedMetadata={handleLoadedMetadata}
            onEnded={() => onSetIsPlaying(false)}
            playsInline
            preload="metadata"
          />
        ) : (
          <>
            <audio
              ref={mediaRef}
              src={mediaUrl}
              onTimeUpdate={handleTimeUpdate}
              onLoadedMetadata={handleLoadedMetadata}
              onEnded={() => onSetIsPlaying(false)}
              className="hidden"
              preload="metadata"
            />
            <div className="absolute inset-0 flex items-center justify-center bg-gradient-to-br from-indigo-900/20 to-purple-900/20">
              <div className="text-center">
                <div className="w-32 h-32 bg-gradient-to-br from-indigo-500/20 to-purple-500/20 rounded-full flex items-center justify-center mx-auto mb-6 backdrop-blur-lg border border-indigo-500/30">
                  <Headphones className="w-16 h-16 text-indigo-400" />
                </div>
                <p className="text-xl font-bold text-white mb-2">Audio Track</p>
                <p className="text-gray-400">No video available for this media</p>
              </div>
            </div>
          </>
        )}

        {!isPlaying && (
          <button
            onClick={togglePlay}
            className="absolute inset-0 flex items-center justify-center bg-black/40 backdrop-blur-sm transition-all group"
            type="button"
          >
            <div className="p-8 bg-gradient-to-br from-indigo-600/90 to-purple-600/90 rounded-full group-hover:scale-110 transition-transform duration-300">
              <Play className="w-16 h-16 text-white ml-2" />
            </div>
          </button>
        )}

        <div className="absolute top-4 right-4 px-3 py-2 bg-black/60 backdrop-blur-sm rounded-lg border border-gray-700/50">
          <span className="text-white font-mono text-sm">
            {formatClock(currentTime)} / {formatClock(duration)}
          </span>
        </div>
      </div>

      <div className="p-6 bg-gradient-to-b from-gray-900/80 to-gray-900/95 border-t border-gray-700/50">
        <div className="mb-6">
          <div
            ref={progressRef}
            onMouseDown={onMouseDown}
            className="relative h-2.5 bg-gray-800/80 rounded-full cursor-pointer hover:h-3 transition-all group"
          >
            <div
              className="absolute h-full bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500 rounded-full transition-all duration-300"
              style={{ width: `${progressPct}%` }}
            >
              <div className="absolute right-0 top-1/2 -translate-y-1/2 w-4 h-4 bg-white rounded-full shadow-lg ring-4 ring-indigo-500/50" />
            </div>

            {duration > 0 &&
              markerFractions.map((m) => (
                <button
                  key={`${m.idx}-${m.frac}`}
                  onClick={(e) => {
                    e.stopPropagation();
                    onJumpToTime(m.frac * duration);
                  }}
                  className="absolute top-1/2 -translate-y-1/2 w-3 h-3 rounded-full bg-gradient-to-br from-red-500 to-rose-500 shadow-lg hover:scale-150 transition-transform hover:ring-2 hover:ring-white/50 cursor-pointer"
                  style={{ left: `calc(${m.frac * 100}% - 6px)` }}
                  title={`Key Statement ${m.idx + 1}`}
                  type="button"
                />
              ))}
          </div>

          <div className="flex justify-between mt-2 text-sm text-gray-400">
            <span>{formatTimeFull(currentTime)}</span>
            <span>{formatTimeFull(duration)}</span>
          </div>
        </div>

        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <button
              onClick={() => skip(-10)}
              className="p-3 bg-gray-800/50 hover:bg-gray-700/50 rounded-xl border border-gray-700/50 hover:border-gray-600/50 transition-all group"
              type="button"
            >
              <SkipBack className="w-5 h-5 text-gray-300 group-hover:text-white" />
            </button>

            <button
              onClick={togglePlay}
              className="p-4 bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-500 hover:to-purple-500 rounded-xl shadow-lg hover:shadow-xl transition-all transform hover:scale-105"
              type="button"
            >
              {isPlaying ? <Pause className="w-6 h-6 text-white" /> : <Play className="w-6 h-6 text-white ml-0.5" />}
            </button>

            <button
              onClick={() => skip(10)}
              className="p-3 bg-gray-800/50 hover:bg-gray-700/50 rounded-xl border border-gray-700/50 hover:border-gray-600/50 transition-all group"
              type="button"
            >
              <SkipForward className="w-5 h-5 text-gray-300 group-hover:text-white" />
            </button>

            <div className="flex items-center gap-2 ml-4">
              <button onClick={toggleMute} className="p-2 text-gray-300 hover:text-white transition-colors" type="button">
                {isMuted || volume === 0 ? <VolumeX className="w-5 h-5" /> : <Volume2 className="w-5 h-5" />}
              </button>
              <input
                type="range"
                min="0"
                max="1"
                step="0.01"
                value={volume}
                onChange={handleVolumeChange}
                className="w-32 accent-indigo-500"
              />
            </div>
          </div>

          <div className="flex items-center gap-3">
            <select
              value={playbackRate}
              onChange={(e) => changePlaybackRate(Number(e.target.value))}
              className="px-3 py-2 bg-gray-800/50 border border-gray-700/50 rounded-lg text-sm text-white hover:border-gray-600/50 transition-colors"
            >
              <option value="0.5">0.5x</option>
              <option value="0.75">0.75x</option>
              <option value="1">1x</option>
              <option value="1.25">1.25x</option>
              <option value="1.5">1.5x</option>
              <option value="2">2x</option>
            </select>

            <button
              onClick={toggleFullscreen}
              className="p-2.5 bg-gray-800/50 hover:bg-gray-700/50 rounded-xl border border-gray-700/50 hover:border-gray-600/50 transition-all"
              type="button"
              title={isFullscreen ? "Exit fullscreen" : "Fullscreen"}
            >
              {isFullscreen ? <Minimize2 className="w-5 h-5 text-gray-300" /> : <Maximize2 className="w-5 h-5 text-gray-300" />}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

const EnhancedTranscript = ({ speech, transcriptText, currentTime, duration, onJumpToTime }) => {
  const [expanded, setExpanded] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [copied, setCopied] = useState(false);
  const containerRef = useRef(null);
  const activeRef = useRef(null);

  const text = (transcriptText || "").trim();

  const paragraphs = useMemo(() => {
    if (!text) return [];
    const raw = text.split(/\n{2,}/g).map((p) => p.trim()).filter(Boolean);
    if (raw.length <= 1) return text.split(/\n/g).map((p) => p.trim()).filter(Boolean);
    return raw;
  }, [text]);

  const paraOffsets = useMemo(() => {
    let acc = 0;
    return paragraphs.map((p) => {
      const start = acc;
      acc += p.length + 2;
      return { start, end: acc };
    });
  }, [paragraphs]);

  const activeIndex = useMemo(() => {
    if (!duration || duration <= 0 || !text || paragraphs.length === 0) return -1;
    const frac = clamp(currentTime / duration, 0, 1);
    const targetChar = frac * text.length;
    const idx = paraOffsets.findIndex((o) => targetChar >= o.start && targetChar < o.end);
    return idx >= 0 ? idx : paragraphs.length - 1;
  }, [currentTime, duration, text, paragraphs.length, paraOffsets]);

  useEffect(() => {
    if (!expanded) return;
    if (activeRef.current && containerRef.current) {
      const el = activeRef.current;
      const parent = containerRef.current;
      const elTop = el.offsetTop;
      const elBottom = elTop + el.offsetHeight;
      const viewTop = parent.scrollTop;
      const viewBottom = viewTop + parent.clientHeight;

      if (elTop < viewTop + 24) parent.scrollTop = Math.max(0, elTop - 24);
      if (elBottom > viewBottom - 24) parent.scrollTop = elBottom - parent.clientHeight + 24;
    }
  }, [activeIndex, expanded]);

  const copyTranscript = async () => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // ignore
    }
  };

  const downloadTranscript = () => {
    const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${(speech?.title || "transcript").replace(/[^\w\-]+/g, "_")}.txt`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  };

  const wordCount = useMemo(() => (text ? text.split(/\s+/).filter(Boolean).length : 0), [text]);

  const filteredParagraphs = useMemo(() => {
    if (!searchQuery) return paragraphs;
    const q = searchQuery.toLowerCase();
    return paragraphs.filter((p) => p.toLowerCase().includes(q));
  }, [paragraphs, searchQuery]);

  return (
    <div className="bg-gradient-to-br from-gray-800/50 to-gray-900/50 backdrop-blur-xl rounded-2xl border border-gray-700/50 overflow-hidden hover:shadow-2xl transition-all h-full flex flex-col">
      <div className="p-6 border-b border-gray-700/50">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-3 bg-gradient-to-br from-gray-800 to-gray-900 rounded-xl border border-gray-700/50">
              <FileText className="w-6 h-6 text-indigo-400" />
            </div>
            <div>
              <h3 className="text-lg font-bold text-white">Transcript</h3>
              <p className="text-sm text-gray-400">
                {wordCount.toLocaleString()} words • {paragraphs.length} paragraphs
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={copyTranscript}
              disabled={!text}
              className="px-4 py-2.5 bg-gray-800/50 hover:bg-gray-700/50 disabled:opacity-50 disabled:cursor-not-allowed text-gray-300 hover:text-white rounded-xl border border-gray-700/50 hover:border-gray-600/50 transition-all flex items-center gap-2"
              type="button"
            >
              {copied ? (
                <>
                  <Check className="w-4 h-4 text-emerald-400" />
                  <span className="text-emerald-400">Copied!</span>
                </>
              ) : (
                <>
                  <Copy className="w-4 h-4" />
                  <span>Copy</span>
                </>
              )}
            </button>

            <button
              onClick={downloadTranscript}
              disabled={!text}
              className="px-4 py-2.5 bg-gradient-to-r from-indigo-600/80 to-purple-600/80 hover:from-indigo-500 hover:to-purple-500 text-white rounded-xl transition-all flex items-center gap-2"
              type="button"
            >
              <Download className="w-4 h-4" />
              Download
            </button>
          </div>
        </div>

        <div className="mt-4 relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-500" />
          <input
            type="text"
            placeholder="Search transcript..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-10 pr-4 py-3 bg-gray-900/50 border border-gray-700/50 rounded-xl text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 focus:border-transparent"
          />
        </div>
      </div>

      <div className="flex-1 overflow-hidden">
        {!text ? (
          <div className="flex items-center justify-center h-full p-8">
            <div className="text-center">
              <FileText className="w-16 h-16 text-gray-700 mx-auto mb-4" />
              <p className="text-gray-500 font-medium">No transcript available</p>
            </div>
          </div>
        ) : (
          <div ref={containerRef} className={`h-full overflow-y-auto p-6 ${expanded ? "" : "max-h-96"}`}>
            <div className="space-y-6">
              {filteredParagraphs.map((p, idx) => {
                const isActive = idx === activeIndex;
                return (
                  <div
                    key={idx}
                    ref={isActive ? activeRef : null}
                    className={`relative p-4 rounded-xl border transition-all duration-300 ${
                      isActive
                        ? "bg-gradient-to-r from-indigo-900/20 to-purple-900/20 border-indigo-500/30 ring-2 ring-indigo-500/20"
                        : "bg-gray-900/30 border-gray-700/30 hover:border-gray-600/50"
                    }`}
                  >
                    {isActive ? (
                      <div className="absolute -left-2 top-1/2 -translate-y-1/2 w-2 h-8 bg-gradient-to-b from-indigo-500 to-purple-500 rounded-full" />
                    ) : null}
                    <div className="flex items-start gap-3">
                      <span className="px-2 py-1 bg-gray-800/50 text-gray-400 text-xs font-mono rounded">
                        {String(idx + 1).padStart(3, "0")}
                      </span>
                      <p className="text-gray-300 leading-relaxed flex-1 whitespace-pre-wrap">{p}</p>
                      <button
                        onClick={() => {
                          if (!duration || duration <= 0) return;
                          const frac = (paraOffsets[idx]?.start || 0) / Math.max(1, text.length);
                          onJumpToTime(frac * duration);
                        }}
                        className="p-2 text-gray-500 hover:text-indigo-400 transition-colors"
                        type="button"
                        title="Jump to this paragraph"
                      >
                        <PlayCircle className="w-5 h-5" />
                      </button>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>

      <div className="p-4 border-t border-gray-700/50 bg-gray-900/30">
        <button
          onClick={() => setExpanded(!expanded)}
          className="w-full py-3 bg-gradient-to-r from-gray-800/50 to-gray-900/50 hover:from-gray-700/50 hover:to-gray-800/50 border border-gray-700/50 hover:border-gray-600/50 rounded-xl text-gray-300 hover:text-white transition-all flex items-center justify-center gap-2"
          type="button"
        >
          {expanded ? (
            <>
              <ChevronUp className="w-5 h-5" />
              <span>Collapse Transcript</span>
            </>
          ) : (
            <>
              <ChevronDown className="w-5 h-5" />
              <span>Expand Transcript</span>
            </>
          )}
        </button>
      </div>
    </div>
  );
};

const EnhancedIdeologyCard = ({
  title,
  score,
  color,
  gradient,
  icon: Icon,
  iconColor,
  bgGradient,
  borderColor,
  subtypes = {},
  description,
  onSubtypeClick,
  activeSubtype,
  keyStatements = [],
  evidenceSections = [],
  ideologyFamily,
}) => {
  const [isExpanded, setIsExpanded] = useState(false);

  const familyEvidenceCount = useMemo(() => {
    const fam = normalizeFamily(ideologyFamily);
    return (evidenceSections || []).map(normalizeEvidenceItem).filter((s) => s?.ideology_family === fam).length;
  }, [evidenceSections, ideologyFamily]);

  const familyKeyCount = useMemo(() => {
    const fam = normalizeFamily(ideologyFamily);
    return (keyStatements || []).map(normalizeEvidenceItem).filter((ks) => ks?.ideology_family === fam).length;
  }, [keyStatements, ideologyFamily]);

  const hasSubtypes = subtypes && typeof subtypes === "object" && Object.keys(subtypes).length > 0;
  const formattedScore = typeof score === "number" ? score.toFixed(1) : "0.0";

  return (
    <div
      className={`bg-gradient-to-br ${bgGradient} backdrop-blur-xl rounded-2xl border ${borderColor} overflow-hidden transition-all duration-500 ${
        isExpanded ? "shadow-2xl" : "shadow-xl hover:shadow-2xl"
      }`}
    >
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full p-6 text-left hover:bg-gradient-to-r hover:from-white/5 hover:to-transparent transition-all"
        type="button"
      >
        <div className="flex items-start justify-between gap-6">
          <div className="flex items-start gap-4 flex-shrink-0">
            <div className={`p-4 rounded-xl bg-gradient-to-br from-white/10 to-transparent border ${borderColor}`}>
              <Icon className={`w-8 h-8 ${iconColor}`} />
            </div>
            <div>
              <div className="flex items-center gap-3 mb-2">
                <h4 className="text-xl font-bold text-white">{title}</h4>
                <span className={`px-4 py-1.5 rounded-full text-sm font-bold border ${color} backdrop-blur-sm`}>
                  {formattedScore}%
                </span>
              </div>
              <p className="text-gray-300 text-sm max-w-md">{description}</p>
            </div>
          </div>

          <div className="flex-1">
            <div className="grid grid-cols-4 gap-4">
              <div className="text-center p-3 bg-gradient-to-b from-white/5 to-transparent rounded-xl border border-white/10">
                <div className="text-2xl font-bold text-white">{familyEvidenceCount}</div>
                <div className="text-xs text-gray-400 mt-1">Evidence Units</div>
              </div>
              <div className="text-center p-3 bg-gradient-to-b from-white/5 to-transparent rounded-xl border border-white/10">
                <div className="text-2xl font-bold text-white">{familyKeyCount}</div>
                <div className="text-xs text-gray-400 mt-1">Statements</div>
              </div>
              <div className="text-center p-3 bg-gradient-to-b from-white/5 to-transparent rounded-xl border border-white/10">
                <div className="text-2xl font-bold text-white">{hasSubtypes ? Object.keys(subtypes).length : "N/A"}</div>
                <div className="text-xs text-gray-400 mt-1">Subtypes</div>
              </div>
              <div className="text-center p-3 bg-gradient-to-b from-white/5 to-transparent rounded-xl border border-white/10">
                <div className="text-2xl font-bold text-white">{formattedScore}%</div>
                <div className="text-xs text-gray-400 mt-1">Total Score</div>
              </div>
            </div>
          </div>

          <div className="flex flex-col items-end gap-4 flex-shrink-0">
            <div className="w-32">
              <div className="h-2.5 bg-gray-800/50 rounded-full overflow-hidden">
                <div className={`h-full rounded-full ${gradient} transition-all duration-1000`} style={{ width: `${score}%` }} />
              </div>
              <div className="flex justify-between text-xs text-gray-400 mt-1">
                <span>0%</span>
                <span>Score</span>
                <span>100%</span>
              </div>
            </div>
            <div
              className={`p-2 rounded-lg bg-gradient-to-br from-white/10 to-transparent border ${borderColor} transition-transform duration-300 ${
                isExpanded ? "rotate-180" : ""
              }`}
            >
              <ChevronDown className="w-5 h-5 text-gray-300" />
            </div>
          </div>
        </div>
      </button>

      {isExpanded && (
        <div className="border-t border-white/10 bg-gradient-to-b from-transparent to-black/20 p-6">
          <div className="mb-6">
            <h5 className="text-lg font-semibold text-white mb-2">Subtypes Breakdown</h5>
            <p className="text-sm text-gray-400">Detailed analysis of ideological subtypes with evidence-based scoring</p>
          </div>

          {hasSubtypes ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {Object.entries(subtypes)
                .sort(([, a], [, b]) => {
                  const scoreA = typeof a === "object" ? a?.score || 0 : a;
                  const scoreB = typeof b === "object" ? b?.score || 0 : b;
                  return scoreB - scoreA;
                })
                .map(([subtypeName, data]) => {
                  const subtypeData = typeof data === "object" ? data : { score: data };
                  const subtypeScore = Number(subtypeData?.score ?? data ?? 0) || 0;
                  const isActive = activeSubtype === `${ideologyFamily}-${subtypeName}`;

                  return (
                    <button
                      key={subtypeName}
                      onClick={(e) => {
                        e.stopPropagation();
                        const fam = normalizeFamily(ideologyFamily);
                        const filtered = (keyStatements || [])
                          .map(normalizeEvidenceItem)
                          .filter((ks) => ks?.ideology_family === fam && ks?.ideology_subtype === subtypeName);
                        onSubtypeClick(`${fam}-${subtypeName}`, filtered);
                      }}
                      className={`p-4 rounded-xl border transition-all duration-300 ${
                        isActive
                          ? "bg-gradient-to-r from-white/10 to-transparent border-indigo-500/50 ring-2 ring-indigo-500/30"
                          : "bg-gradient-to-b from-white/5 to-transparent border-white/10 hover:border-white/20"
                      }`}
                      type="button"
                    >
                      <div className="flex items-center justify-between mb-3">
                        <span className="font-semibold text-white">{subtypeName}</span>
                        <span className="text-xl font-bold text-white">{subtypeScore}%</span>
                      </div>
                      <div className="h-2 bg-gray-800/50 rounded-full overflow-hidden">
                        <div className={`h-full ${gradient} rounded-full transition-all duration-1000`} style={{ width: `${subtypeScore}%` }} />
                      </div>
                    </button>
                  );
                })}
            </div>
          ) : (
            <div className="text-center py-8">
              <Scale className="w-12 h-12 text-gray-600 mx-auto mb-4" />
              <p className="text-gray-400">No subtypes detected for this ideology</p>
            </div>
          )}

          {familyKeyCount > 0 && (
            <div className="mt-6">
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  const fam = normalizeFamily(ideologyFamily);
                  const filtered = (keyStatements || []).map(normalizeEvidenceItem).filter((ks) => ks?.ideology_family === fam);
                  onSubtypeClick(fam, filtered);
                }}
                className="w-full py-4 bg-gradient-to-r from-indigo-600/20 to-purple-600/20 hover:from-indigo-500/30 hover:to-purple-500/30 border border-indigo-500/30 hover:border-indigo-400/50 rounded-xl transition-all flex items-center justify-center gap-3 group"
                type="button"
              >
                <MessageSquare className="w-5 h-5 text-indigo-400" />
                <span className="text-white font-semibold">Filter {familyKeyCount} Statements</span>
                <ChevronRight className="w-5 h-5 text-indigo-400 group-hover:translate-x-1 transition-transform" />
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

const EnhancedKeyHighlights = ({ statements, activeSubtype, showFiltered, onClearFilter, onJumpToTimeApprox }) => {
  const [expandedIndex, setExpandedIndex] = useState(null);
  const [showAll, setShowAll] = useState(false);

  const displayStatements = showAll ? statements : statements.slice(0, 4);

  return (
    <div className="bg-gradient-to-br from-gray-800/50 to-gray-900/50 backdrop-blur-xl rounded-2xl border border-gray-700/50 overflow-hidden hover:shadow-2xl transition-all">
      <div className="p-6 border-b border-gray-700/50">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-3 bg-gradient-to-br from-gray-800 to-gray-900 rounded-xl border border-gray-700/50">
              <MessageSquare className="w-6 h-6 text-indigo-400" />
            </div>
            <div>
              <h3 className="text-lg font-bold text-white">
                {showFiltered ? (
                  <>
                    Statements: <span className="text-indigo-400">{activeSubtype}</span>
                  </>
                ) : (
                  "Key Statements"
                )}
              </h3>
              <p className="text-sm text-gray-400">
                {statements.length} statement{statements.length !== 1 ? "s" : ""} • Click play to jump
              </p>
            </div>
          </div>

          {showFiltered && (
            <button
              onClick={onClearFilter}
              className="px-4 py-2.5 bg-gradient-to-r from-gray-800/50 to-gray-900/50 hover:from-gray-700/50 hover:to-gray-800/50 border border-gray-700/50 hover:border-gray-600/50 rounded-xl text-gray-300 hover:text-white transition-all flex items-center gap-2"
              type="button"
            >
              <Filter className="w-4 h-4" />
              Clear Filter
            </button>
          )}
        </div>
      </div>

      <div className="p-6">
        {statements.length === 0 ? (
          <div className="text-center py-12">
            <MessageSquare className="w-16 h-16 text-gray-700 mx-auto mb-4" />
            <p className="text-gray-500 font-medium">No highlights found</p>
          </div>
        ) : (
          <>
            <div className="space-y-4">
              {displayStatements.map((st, idx) => (
                <EnhancedHighlightCard
                  key={idx}
                  statement={st}
                  index={idx}
                  expandedIndex={expandedIndex}
                  setExpandedIndex={setExpandedIndex}
                  onJumpToTimeApprox={onJumpToTimeApprox}
                />
              ))}
            </div>

            {statements.length > 4 && (
              <div className="mt-6 text-center">
                <button
                  onClick={() => setShowAll(!showAll)}
                  className="px-6 py-3 bg-gradient-to-r from-gray-800/50 to-gray-900/50 hover:from-gray-700/50 hover:to-gray-800/50 border border-gray-700/50 hover:border-gray-600/50 rounded-xl text-gray-300 hover:text-white transition-all"
                  type="button"
                >
                  {showAll ? `Show Less (First 4)` : `Show All ${statements.length} Statements`}
                </button>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
};

const EnhancedHighlightCard = ({ statement, index, expandedIndex, setExpandedIndex, onJumpToTimeApprox }) => {
  const norm = normalizeEvidenceItem(statement);
  const fam = normalizeFamily(norm?.ideology_family);
  const config = getIdeologyConfig(fam);
  const Icon = config.icon;
  const isExpanded = expandedIndex === index;

  const conf = Number(norm?.confidence_score ?? norm?.confidence);
  const confPct = Number.isFinite(conf) ? Math.round(conf * 100) : null;

  return (
    <div
      className={`bg-gradient-to-br from-gray-900/50 to-gray-800/50 backdrop-blur-xl rounded-xl border ${config.borderColor} p-5 hover:shadow-xl transition-all duration-300 ${
        isExpanded ? "ring-2 ring-indigo-500/30" : ""
      }`}
    >
      <div className="flex items-start justify-between gap-4 mb-3">
        <div className="flex items-center gap-3 flex-wrap">
          <div className={`p-2.5 rounded-lg ${config.bgGradient} border ${config.borderColor}`}>
            <Icon className={`w-5 h-5 ${config.iconColor}`} />
          </div>
          <span className={`px-3 py-1 rounded-full text-xs font-bold border ${config.color}`}>{fam}</span>

          {fam !== CENTRIST && norm?.ideology_subtype ? (
            <span className="px-3 py-1 bg-indigo-900/30 text-indigo-300 rounded-full text-xs font-medium border border-indigo-700/50">
              {norm.ideology_subtype}
            </span>
          ) : null}

          {confPct !== null ? (
            <span className="px-3 py-1 bg-emerald-900/30 text-emerald-300 rounded-full text-xs font-bold border border-emerald-700/50">
              {confPct}% confidence
            </span>
          ) : null}
        </div>

        <button
          onClick={() => onJumpToTimeApprox(norm)}
          className="p-2.5 bg-gradient-to-br from-gray-800 to-gray-900 hover:from-gray-700 hover:to-gray-800 border border-gray-700/50 hover:border-gray-600/50 rounded-lg transition-all group"
          title="Jump"
          type="button"
        >
          <Play className="w-4 h-4 text-gray-400 group-hover:text-indigo-400" />
        </button>
      </div>

      <div className="mb-3">
        <p className="text-gray-200 text-sm leading-relaxed font-medium italic">"{norm?.text || ""}"</p>
      </div>

      {norm?.context_before || norm?.context_after ? (
        <div>
          <button
            onClick={() => setExpandedIndex(isExpanded ? null : index)}
            className="flex items-center gap-2 text-sm font-medium text-indigo-400 hover:text-indigo-300 transition-colors"
            type="button"
          >
            {isExpanded ? (
              <>
                <ChevronUp className="w-4 h-4" />
                <span>Hide Context</span>
              </>
            ) : (
              <>
                <ChevronDown className="w-4 h-4" />
                <span>Show Context</span>
              </>
            )}
          </button>

          {isExpanded && (
            <div className="mt-3 p-4 bg-gradient-to-b from-gray-900/50 to-transparent rounded-xl border border-gray-700/50 space-y-3">
              {norm?.context_before ? (
                <div>
                  <p className="text-xs font-semibold text-gray-400 mb-2">Before</p>
                  <p className="text-sm text-gray-300 whitespace-pre-wrap">{norm.context_before}</p>
                </div>
              ) : null}
              {norm?.context_after ? (
                <div>
                  <p className="text-xs font-semibold text-gray-400 mb-2">After</p>
                  <p className="text-sm text-gray-300 whitespace-pre-wrap">{norm.context_after}</p>
                </div>
              ) : null}
            </div>
          )}
        </div>
      ) : null}
    </div>
  );
};

const QuestionsGeneratorCard = ({ speechId }) => {
  const [questionType, setQuestionType] = useState("journalistic");
  const [numQuestions, setNumQuestions] = useState(5);
  const [generating, setGenerating] = useState(false);
  const [genError, setGenError] = useState("");
  const [generatedQuestions, setGeneratedQuestions] = useState([]);

  const doGenerate = async () => {
    setGenError("");
    setGenerating(true);

    try {
      const res = await apiClient.post("/api/analysis/questions/generate", {
        speech_id: Number(speechId),
        question_type: questionType,
        max_questions: clamp(Number(numQuestions) || 5, 1, 8),
        llm_provider: null,
        llm_model: null,
      });

      const payload = unwrap(res) || {};
      if (payload?.success === false) {
        throw new Error(payload?.error || payload?.message || "Question generation failed");
      }

      const qs = payload?.questions || payload?.data?.questions || [];
      setGeneratedQuestions(Array.isArray(qs) ? qs : []);
    } catch (e) {
      const msg =
        e?.response?.data?.detail ||
        e?.response?.data?.error ||
        e?.response?.data?.message ||
        e?.message ||
        "Question generation failed";
      setGenError(String(msg));
    } finally {
      setGenerating(false);
    }
  };

  const generated = Array.isArray(generatedQuestions) ? generatedQuestions : [];

  return (
    <div className="bg-gradient-to-br from-gray-800/50 to-gray-900/50 backdrop-blur-xl rounded-2xl border border-gray-700/50 overflow-hidden hover:shadow-2xl transition-all h-full flex flex-col">
      <div className="p-6 border-b border-gray-700/50">
        <div className="flex items-center gap-3">
          <div className="p-3 bg-gradient-to-br from-gray-800 to-gray-900 rounded-xl border border-gray-700/50">
            <Zap className="w-6 h-6 text-yellow-400" />
          </div>
          <div>
            <h3 className="text-lg font-bold text-white">Question Generator</h3>
            <p className="text-sm text-gray-400">Generate evidence-based interview questions</p>
          </div>
        </div>
      </div>

      <div className="p-6 flex-1 flex flex-col">
        <div className="space-y-4 mb-6">
          <div>
            <label className="block text-xs font-semibold text-gray-400 mb-2">Question Type</label>
            <div className="flex gap-2">
              <button
                onClick={() => setQuestionType("journalistic")}
                className={`flex-1 px-3 py-2.5 rounded-lg text-sm transition-all ${
                  questionType === "journalistic"
                    ? "bg-gradient-to-r from-indigo-600/80 to-purple-600/80 text-white"
                    : "bg-gray-900/50 text-gray-400 hover:text-white hover:bg-gray-800/50"
                }`}
                type="button"
              >
                Journalistic
              </button>
              <button
                onClick={() => setQuestionType("technical")}
                className={`flex-1 px-3 py-2.5 rounded-lg text-sm transition-all ${
                  questionType === "technical"
                    ? "bg-gradient-to-r from-indigo-600/80 to-purple-600/80 text-white"
                    : "bg-gray-900/50 text-gray-400 hover:text-white hover:bg-gray-800/50"
                }`}
                type="button"
              >
                Technical
              </button>
            </div>
          </div>

          <div>
            <label className="block text-xs font-semibold text-gray-400 mb-2">Number of Questions</label>
            <div className="flex gap-2">
              {[3, 5, 8].map((num) => (
                <button
                  key={num}
                  onClick={() => setNumQuestions(num)}
                  className={`flex-1 px-3 py-2.5 rounded-lg text-sm transition-all ${
                    numQuestions === num
                      ? "bg-gradient-to-r from-indigo-600/80 to-purple-600/80 text-white"
                      : "bg-gray-900/50 text-gray-400 hover:text-white hover:bg-gray-800/50"
                  }`}
                  type="button"
                >
                  {num} Questions
                </button>
              ))}
            </div>
          </div>
        </div>

        <button
          onClick={doGenerate}
          disabled={generating}
          className={`w-full py-3.5 rounded-xl bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-500 hover:to-purple-500 text-white font-semibold transition-all duration-300 ${
            generating ? "opacity-50 cursor-not-allowed" : "hover:shadow-lg"
          }`}
          type="button"
        >
          {generating ? (
            <div className="flex items-center justify-center gap-3">
              <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
              <span>Generating...</span>
            </div>
          ) : (
            <div className="flex items-center justify-center gap-2">
              <Sparkles className="w-4 h-4" />
              <span>Generate Questions</span>
            </div>
          )}
        </button>

        {genError ? (
          <div className="mt-4 p-3 bg-gradient-to-r from-red-900/20 to-rose-900/20 border border-red-700/30 rounded-lg">
            <div className="flex items-center gap-2">
              <X className="w-4 h-4 text-red-400 flex-shrink-0" />
              <span className="text-sm text-red-300">{genError}</span>
            </div>
          </div>
        ) : null}

        <div className="mt-6 flex-1 overflow-hidden">
          {generated.length > 0 ? (
            <div className="h-full flex flex-col">
              <h4 className="text-sm font-semibold text-white mb-3 flex items-center gap-2">
                <Sparkles className="w-4 h-4" />
                Generated Questions ({generated.length})
              </h4>
              <div className="space-y-2 flex-1 overflow-y-auto pr-2">
                {generated.map((q, idx) => (
                  <div
                    key={idx}
                    className="p-3 bg-gradient-to-r from-indigo-900/20 to-purple-900/20 border border-indigo-700/30 rounded-lg group hover:border-indigo-600/50 transition-colors"
                  >
                    <div className="flex items-start gap-2">
                      <div className="px-2 py-1 bg-gradient-to-br from-indigo-600 to-purple-600 rounded text-xs font-bold text-white">
                        Q{idx + 1}
                      </div>
                      <p className="text-gray-200 text-sm flex-1">{q}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <div className="text-center py-6">
              <Zap className="w-12 h-12 text-gray-700 mx-auto mb-3" />
              <p className="text-gray-500 font-medium">No questions generated yet</p>
              <p className="text-xs text-gray-600 mt-1">Select options above and click "Generate Questions"</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

const EnhancedSegmentsCard = ({ segments }) => {
  const [expandedIndex, setExpandedIndex] = useState(null);
  const [filterType, setFilterType] = useState("all");
  const [showAllSegments, setShowAllSegments] = useState(false);

  const filteredSegments = useMemo(() => {
    const arr = Array.isArray(segments) ? segments.map(normalizeEvidenceItem) : [];
    if (filterType === "all") return arr;
    const f = normalizeFamily(filterType);
    return arr.filter((s) => normalizeFamily(s?.ideology_family) === f);
  }, [segments, filterType]);

  const displaySegments = showAllSegments ? filteredSegments : filteredSegments.slice(0, 5);

  return (
    <div className="bg-gradient-to-br from-gray-800/50 to-gray-900/50 backdrop-blur-xl rounded-2xl border border-gray-700/50 overflow-hidden hover:shadow-2xl transition-all">
      <div className="p-6 border-b border-gray-700/50">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-3 bg-gradient-to-br from-gray-800 to-gray-900 rounded-xl border border-gray-700/50">
              <Layers className="w-6 h-6 text-indigo-400" />
            </div>
            <div>
              <h3 className="text-lg font-bold text-white">Speech Segments</h3>
              <p className="text-sm text-gray-400">{(segments || []).length} segments • Detailed analysis</p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            {["all", LIB, AUTH].map((t) => (
              <button
                key={t}
                onClick={() => setFilterType(t)}
                className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                  filterType === t
                    ? t === "all"
                      ? "bg-gradient-to-r from-indigo-600/80 to-purple-600/80 text-white"
                      : t === LIB
                      ? "bg-emerald-900/50 text-emerald-300 border border-emerald-700/50"
                      : "bg-rose-900/50 text-rose-300 border border-rose-700/50"
                    : "bg-gray-800/50 text-gray-400 hover:text-white hover:bg-gray-700/50"
                }`}
                type="button"
              >
                {t === "all" ? "All" : t}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="p-6">
        {displaySegments.length === 0 ? (
          <div className="text-center py-12">
            <Layers className="w-16 h-16 text-gray-700 mx-auto mb-4" />
            <p className="text-gray-500 font-medium">No segments found</p>
          </div>
        ) : (
          <div className="grid grid-cols-1 gap-4">
            {displaySegments.map((segment, idx) => (
              <EnhancedSegmentCard key={idx} segment={segment} index={idx} expandedIndex={expandedIndex} setExpandedIndex={setExpandedIndex} />
            ))}
          </div>
        )}

        {filteredSegments.length > 5 ? (
          <div className="mt-6 text-center">
            <button
              onClick={() => setShowAllSegments(!showAllSegments)}
              className="px-6 py-3 bg-gradient-to-r from-gray-800/50 to-gray-900/50 hover:from-gray-700/50 hover:to-gray-800/50 border border-gray-700/50 hover:border-gray-600/50 rounded-xl text-gray-300 hover:text-white transition-all"
              type="button"
            >
              {showAllSegments ? `Show Less (First 5)` : `Show All ${filteredSegments.length} Segments`}
            </button>
          </div>
        ) : null}
      </div>
    </div>
  );
};

const EnhancedSegmentCard = ({ segment, index, expandedIndex, setExpandedIndex }) => {
  const norm = normalizeEvidenceItem(segment);
  const fam = normalizeFamily(norm?.ideology_family);
  const config = getIdeologyConfig(fam);
  const Icon = config.icon;
  const isExpanded = expandedIndex === index;

  const text = norm?.full_text || norm?.text || norm?.content || "";
  const shouldTruncate = text && text.length > 280 && !isExpanded;

  const conf = Number(norm?.confidence_score ?? norm?.confidence);
  const confPct = Number.isFinite(conf) ? Math.round(conf * 100) : null;

  return (
    <div
      className={`bg-gradient-to-br from-gray-900/50 to-gray-800/50 backdrop-blur-xl rounded-xl border ${config.borderColor} p-5 hover:shadow-xl transition-all duration-300 ${
        isExpanded ? "ring-2 ring-indigo-500/30" : ""
      }`}
    >
      <div className="flex items-start justify-between gap-4 mb-4">
        <div className="flex items-center gap-3 flex-wrap">
          <div className="px-3 py-1.5 bg-gradient-to-br from-gray-800 to-gray-900 rounded-lg border border-gray-700/50">
            <span className="text-sm font-bold text-white">#{index + 1}</span>
          </div>

          <div className={`p-2 rounded-lg ${config.bgGradient} border ${config.borderColor}`}>
            <Icon className={`w-4 h-4 ${config.iconColor}`} />
          </div>

          <span className={`px-3 py-1 rounded-full text-xs font-bold border ${config.color}`}>{fam}</span>

          {fam !== CENTRIST && norm?.ideology_subtype ? (
            <span className="px-3 py-1 bg-indigo-900/30 text-indigo-300 rounded-full text-xs font-medium border border-indigo-700/50">
              {norm.ideology_subtype}
            </span>
          ) : null}

          {confPct !== null ? (
            <span className="px-3 py-1 bg-emerald-900/30 text-emerald-300 rounded-full text-xs font-bold border border-emerald-700/50">
              {confPct}% confidence
            </span>
          ) : null}
        </div>

        <button
          onClick={() => setExpandedIndex(isExpanded ? null : index)}
          className="p-2.5 bg-gradient-to-br from-gray-800 to-gray-900 hover:from-gray-700 hover:to-gray-800 border border-gray-700/50 hover:border-gray-600/50 rounded-lg transition-all"
          type="button"
        >
          {isExpanded ? <ChevronUp className="w-4 h-4 text-gray-400" /> : <ChevronDown className="w-4 h-4 text-gray-400" />}
        </button>
      </div>

      <div className={`transition-all duration-300 ${shouldTruncate ? "max-h-32 overflow-hidden relative" : ""}`}>
        <p className="text-gray-300 leading-relaxed whitespace-pre-wrap">{text || "No content available"}</p>
        {shouldTruncate ? <div className="absolute bottom-0 left-0 right-0 h-12 bg-gradient-to-t from-gray-900/50 to-transparent" /> : null}
      </div>

      <div className="flex items-center gap-4 mt-4 pt-4 border-t border-gray-700/30">
        {Number.isFinite(Number(norm?.signal_strength)) ? (
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 bg-gradient-to-r from-yellow-500 to-orange-500 rounded-full" />
            <span className="text-xs text-gray-400">Signal:</span>
            <span className="text-xs font-bold text-white">{Math.round(Number(norm.signal_strength))}%</span>
          </div>
        ) : null}

        {Number.isFinite(Number(norm?.evidence_count)) ? (
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 bg-gradient-to-r from-cyan-500 to-blue-500 rounded-full" />
            <span className="text-xs text-gray-400">Evidence:</span>
            <span className="text-xs font-bold text-white">{Number(norm.evidence_count)}</span>
          </div>
        ) : null}

        <div className="ml-auto text-xs text-gray-500">{(text || "").length} characters</div>
      </div>
    </div>
  );
};

const AnalysisPage = () => {
  const { id } = useParams();
  const navigate = useNavigate();

  const [speech, setSpeech] = useState(null);
  const speechRef = useRef(null);

  const [overview, setOverview] = useState(null);
  const [keyStatements, setKeyStatements] = useState([]);
  const [segments, setSegments] = useState([]);

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const [activeSubtype, setActiveSubtype] = useState(null);
  const [filteredStatements, setFilteredStatements] = useState([]);
  const [showFilteredStatements, setShowFilteredStatements] = useState(false);

  const mediaRef = useRef(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);

  const fetchedWithDurationRef = useRef(false);

  const transcriptText = speech?.text || "";

  const hasMedia = Boolean(speech?.media_url);
  const mediaUrl = hasMedia ? getMediaUrl(id, speech) : null;

  const isVideo = Boolean(
    speech?.media_url?.toLowerCase?.().includes(".mp4") ||
      speech?.media_url?.toLowerCase?.().includes(".webm") ||
      speech?.media_url?.toLowerCase?.().includes(".mov") ||
      speech?.media_url?.toLowerCase?.().includes(".mkv")
  );

  const handleSubtypeClick = (subtypeKey, statementsList) => {
    setActiveSubtype(subtypeKey);
    setFilteredStatements(Array.isArray(statementsList) ? statementsList : []);
    setShowFilteredStatements(true);
  };

  const clearFilteredStatements = () => {
    setShowFilteredStatements(false);
    setActiveSubtype(null);
    setFilteredStatements([]);
  };

  const displayStatements = showFilteredStatements ? filteredStatements : keyStatements;

  const jumpToTime = async (seconds) => {
    const el = mediaRef.current;
    if (!el) return;
    const t = clamp(Number(seconds) || 0, 0, duration || Number.MAX_SAFE_INTEGER);
    el.currentTime = t;
    setCurrentTime(t);
    try {
      await el.play?.();
      setIsPlaying(true);
    } catch {
      setIsPlaying(false);
    }
  };

  const jumpToTimeApproxFromStatement = (statement) => {
    if (!mediaRef.current || !duration || duration <= 0) return;

    const tb = Number(statement?.time_begin);
    if (Number.isFinite(tb) && tb >= 0) {
      jumpToTime(tb);
      return;
    }

    const text = transcriptText || "";
    const total = Math.max(1, text.length);

    const sc = Number(statement?.start_char);
    if (Number.isFinite(sc)) {
      const frac = clamp(sc / total, 0, 1);
      jumpToTime(frac * duration);
      return;
    }

    if (displayStatements.length > 0) {
      const idx = displayStatements.findIndex((s) => s === statement);
      if (idx >= 0) {
        const frac = (idx + 1) / (displayStatements.length + 1);
        jumpToTime(frac * duration);
        return;
      }
    }

    jumpToTime(duration / 2);
  };

  const fetchAnalysis = useCallback(
    async (dur = null) => {
      const params = {};
      if (dur && Number(dur) > 0) params.media_duration_seconds = Number(dur);

      const res = await apiClient.get(`/api/analysis/speech/${Number(id)}`, { params });
      const payload = unwrap(res) || {};

      if (payload?.success === false) {
        throw new Error(payload?.error || payload?.message || "Failed to load analysis");
      }

      const fallbackSpeech = speechRef.current || {};
      const ov = normalizeOverview(payload, fallbackSpeech);
      setOverview(ov);

      const ksRaw = unwrapArray(payload, ["key_statements", "key_segments", "highlights"]).map(normalizeEvidenceItem);
      const segRaw = unwrapArray(payload, ["segments", "sections", "scored_segments"]).map(normalizeEvidenceItem);

      const centristFromKS = ksRaw
        .filter((x) => normalizeFamily(x?.ideology_family) === CENTRIST)
        .map((x, i) => ({
          ...x,
          full_text: x.full_text || x.text || "",
          evidence_count: Number.isFinite(Number(x.evidence_count)) ? x.evidence_count : 1,
          signal_strength: Number.isFinite(Number(x.signal_strength)) ? x.signal_strength : 0,
          _source: "centrist_key_statement",
          _idx: i,
        }));

      const seen = new Set();
      const mergedSegments = [...segRaw, ...centristFromKS].filter((s) => {
        const t = String(s?.full_text || s?.text || "").trim();
        const tb = Number(s?.time_begin);
        const sc = Number(s?.start_char);
        const key = `${normalizeFamily(s?.ideology_family)}|${t}|${Number.isFinite(tb) ? tb : ""}|${Number.isFinite(sc) ? sc : ""}`;
        if (!t) return false;
        if (seen.has(key)) return false;
        seen.add(key);
        return true;
      });

      setKeyStatements(ksRaw);
      setSegments(mergedSegments);
    },
    [id]
  );

  useEffect(() => {
    let cancelled = false;

    const load = async () => {
      setLoading(true);
      setError("");
      fetchedWithDurationRef.current = false;

      try {
        const [speechRes, speechFullRes] = await Promise.all([
          getSpeech(id, { include_text: true, include_analysis: true }),
          getSpeechFull(id),
        ]);

        const speechData = unwrap(speechRes) || {};
        if (speechData?.success === false) {
          throw new Error(speechData?.error || speechData?.message || "Failed to load speech");
        }

        const speechFull = unwrap(speechFullRes) || {};
        if (speechFull?.success === false) {
          throw new Error(speechFull?.error || speechFull?.message || "Failed to load full speech");
        }

        const mergedSpeech = { ...(speechData || {}) };
        if (speechFull?.text && !mergedSpeech?.text) mergedSpeech.text = speechFull.text;

        if (!cancelled) {
          setSpeech(mergedSpeech);
          speechRef.current = mergedSpeech;
        }

        await fetchAnalysis(null);
      } catch (e) {
        if (!cancelled) setError(String(e?.message || "Failed to load analysis data"));
      } finally {
        if (!cancelled) setLoading(false);
      }
    };

    if (id) load();
    return () => {
      cancelled = true;
    };
  }, [id, fetchAnalysis]);

  useEffect(() => {
    if (!id) return;
    if (!duration || duration <= 0) return;
    if (fetchedWithDurationRef.current) return;
    if (!transcriptText || transcriptText.length < 20) return;

    (async () => {
      try {
        await fetchAnalysis(duration);
        fetchedWithDurationRef.current = true;
      } catch {
        // ignore
      }
    })();
  }, [duration, id, transcriptText, fetchAnalysis]);

  if (loading) return <LoadingScreen />;
  if (error) return <ErrorScreen error={error} onBack={() => navigate("/dashboard")} />;

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900">
      <AnalysisHeader speech={speech} onBack={() => navigate("/dashboard")} />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
        <PrimaryClassificationCard overview={overview} />

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {hasMedia && mediaUrl ? (
            <EnhancedMediaPlayer
              mediaUrl={mediaUrl}
              isVideo={isVideo}
              keyStatements={keyStatements}
              transcriptText={transcriptText}
              onJumpToTime={jumpToTime}
              currentTime={currentTime}
              duration={duration}
              isPlaying={isPlaying}
              onSetIsPlaying={setIsPlaying}
              mediaRef={mediaRef}
              onSetDuration={setDuration}
              onSetCurrentTime={setCurrentTime}
            />
          ) : (
            <div className="bg-gradient-to-br from-gray-800/50 to-gray-900/50 backdrop-blur-xl rounded-2xl border border-gray-700/50 p-8">
              <div className="text-center">
                <Headphones className="w-16 h-16 text-gray-700 mx-auto mb-4" />
                <h3 className="text-xl font-bold text-white mb-2">No Media Available</h3>
                <p className="text-gray-400">This speech doesn't have an associated media file.</p>
              </div>
            </div>
          )}

          <EnhancedTranscript
            speech={speech}
            transcriptText={transcriptText}
            currentTime={currentTime}
            duration={duration}
            onJumpToTime={jumpToTime}
          />
        </div>

        {overview && (
          <div className="bg-gradient-to-br from-gray-800/30 to-gray-900/30 backdrop-blur-xl rounded-2xl border border-gray-700/50 p-8">
            <div className="mb-8">
              <div className="flex items-center gap-3 mb-4">
                <div className="p-3 bg-gradient-to-br from-gray-800 to-gray-900 rounded-xl border border-gray-700/50">
                  <Target className="w-6 h-6 text-indigo-400" />
                </div>
                <div>
                  <h2 className="text-2xl font-bold text-white">Ideology Analysis</h2>
                  <p className="text-gray-400">Detail view focuses on ideological evidence.</p>
                </div>
              </div>
            </div>

            <div className="space-y-6">
              <EnhancedIdeologyCard
                title={LIB}
                score={overview.libertarian_score || 0}
                {...getIdeologyConfig(LIB)}
                subtypes={getSubtypesFromAnalysis(overview, LIB, keyStatements, segments)}
                onSubtypeClick={handleSubtypeClick}
                activeSubtype={activeSubtype}
                keyStatements={keyStatements}
                evidenceSections={segments}
                ideologyFamily={LIB}
                description={getIdeologyConfig(LIB).description}
              />

              <EnhancedIdeologyCard
                title={AUTH}
                score={overview.authoritarian_score || 0}
                {...getIdeologyConfig(AUTH)}
                subtypes={getSubtypesFromAnalysis(overview, AUTH, keyStatements, segments)}
                onSubtypeClick={handleSubtypeClick}
                activeSubtype={activeSubtype}
                keyStatements={keyStatements}
                evidenceSections={segments}
                ideologyFamily={AUTH}
                description={getIdeologyConfig(AUTH).description}
              />
            </div>

            {showFilteredStatements && (
              <div className="mt-8 p-4 bg-gradient-to-r from-indigo-900/20 to-purple-900/20 border border-indigo-700/30 rounded-xl">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <Filter className="w-5 h-5 text-indigo-400" />
                    <div>
                      <p className="text-white font-semibold">Active Filter: {activeSubtype}</p>
                      <p className="text-sm text-gray-400">{filteredStatements.length} highlights</p>
                    </div>
                  </div>
                  <button
                    onClick={clearFilteredStatements}
                    className="px-4 py-2.5 bg-gradient-to-r from-gray-800/50 to-gray-900/50 hover:from-gray-700/50 hover:to-gray-800/50 border border-gray-700/50 hover:border-gray-600/50 rounded-xl text-gray-300 hover:text-white transition-all"
                    type="button"
                  >
                    Clear Filter
                  </button>
                </div>
              </div>
            )}
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          <div className="lg:col-span-2">
            <EnhancedKeyHighlights
              statements={displayStatements}
              activeSubtype={activeSubtype}
              showFiltered={showFilteredStatements}
              onClearFilter={clearFilteredStatements}
              onJumpToTimeApprox={jumpToTimeApproxFromStatement}
            />
          </div>
          <div>
            <QuestionsGeneratorCard speechId={id} />
          </div>
        </div>

        <EnhancedSegmentsCard segments={segments} />

        <style>{`
          ::-webkit-scrollbar { width: 10px; height: 10px; }
          ::-webkit-scrollbar-track { background: rgba(30, 41, 59, 0.3); border-radius: 5px; }
          ::-webkit-scrollbar-thumb { background: linear-gradient(to bottom, #6366f1, #a855f7); border-radius: 5px; }
          ::-webkit-scrollbar-thumb:hover { background: linear-gradient(to bottom, #4f46e5, #9333ea); }
          input[type="range"] {
            -webkit-appearance: none;
            height: 6px;
            background: rgba(75, 85, 99, 0.5);
            border-radius: 3px;
            outline: none;
          }
          input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 18px;
            height: 18px;
            border-radius: 50%;
            background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
            cursor: pointer;
            border: 2px solid rgba(255, 255, 255, 0.8);
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
          }
        `}</style>
      </div>
    </div>
  );
};

export default AnalysisPage;
