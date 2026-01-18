
// frontend/src/pages/ComparisonPage.jsx
import React, { useEffect, useMemo, useState } from "react";
import {
  BarChart3,
  Users,
  RefreshCw,
  Calendar,
  Scale,
  TrendingUp,
  TrendingDown,
  Search,
  Filter,
  Info,
  AlertTriangle,
  Target,
  Sparkles,
  GitCompare,
  Clock,
  Globe,
  Activity,
  Cpu,
  Database,
  Shield,
  Lock,
  Eye,
  Maximize2,
  Download,
  Share2,
  X,
  CheckCircle,
  Loader2,
} from "lucide-react";

/**
 * Advanced Comparison Page
 * - Centrist-only policy (no "Neutral"); Centrist has no subtype.
 * - Icons/colors aligned with AnalysisPage:
 *   * Libertarian = Shield (emerald)
 *   * Authoritarian = Lock (rose)
 *   * Centrist = Scale (blue)
 */

const CENTRIST = "Centrist";
const LIB = "Libertarian";
const AUTH = "Authoritarian";

const clamp = (n, min, max) => Math.max(min, Math.min(max, n));
const pct = (x) => `${(Number(x) || 0).toFixed(1)}%`;

/** Date helpers */
const getDateValue = (p) => {
  const d = p?.date || p?.analyzed_at || p?.created_at || p?.updated_at || null;
  if (!d) return null;
  const dt = new Date(d);
  return Number.isNaN(dt.getTime()) ? null : dt;
};

/** String helpers - safe speaker/title extraction */
const getSpeaker = (p) =>
  (p?.speaker_name || p?.speaker || p?.speakerName || "Unknown speaker").trim();

const getTitle = (p) => (p?.title || p?.name || "Untitled").trim();

/** Family-safe normalize (just in case upstream UI feeds legacy data) */
const normalizeFamily = (fam) => {
  const f = String(fam || "").trim();
  if (f === LIB || f === AUTH || f === CENTRIST) return f;
  if (f === "Neutral") return CENTRIST;
  return CENTRIST;
};

/** Score extraction (Centrist policy) */
const getScores = (p) => {
  // Prefer analysis_summary.scores (server-provided)
  const as = p?.analysis_summary || {};
  const asScores = as?.scores || {};

  const lib =
    Number(
      asScores?.[LIB] ??
        p?.scores?.[LIB] ??
        p?.libertarian_score ??
        p?.overall_lib_score ??
        0
    ) || 0;

  const auth =
    Number(
      asScores?.[AUTH] ??
        p?.scores?.[AUTH] ??
        p?.authoritarian_score ??
        p?.overall_auth_score ??
        0
    ) || 0;

  // Centrist-only (no Neutral)
  const cen =
    Number(
      asScores?.[CENTRIST] ??
        p?.scores?.[CENTRIST] ??
        p?.centrist_score ??
        p?.overall_centrist_score ?? // optional legacy alias
        p?.overall_neutral_score ?? // last-ditch legacy
        0
    ) || 0;

  // Normalize to sum ~100 if needed
  const total = lib + auth + cen;
  if (total > 0 && Math.abs(total - 100) > 0.6) {
    return {
      lib: (lib / total) * 100,
      auth: (auth / total) * 100,
      cen: (cen / total) * 100,
    };
  }
  return { lib, auth, cen };
};

/** Confidence is 0..1 (backend typically), but normalize if 0..100 appears */
const getConfidence01 = (p) => {
  const as = p?.analysis_summary || {};
  const c = Number(p?.confidence_score ?? as?.confidence_score ?? 0);
  if (!Number.isFinite(c)) return 0;
  if (c > 1.01) return clamp(c / 100, 0, 1);
  return clamp(c, 0, 1);
};

/** Evidence count (segments) */
const getEvidenceCount = (p) => {
  const as = p?.analysis_summary || {};
  const n = Number(
    p?.evidence_segment_count ??
      p?.evidence_sentence_count ??
      as?.evidence_segment_count ??
      as?.evidence_sentence_count ??
      p?.segment_count ??
      0
  );
  return Number.isFinite(n) ? Math.max(0, Math.floor(n)) : 0;
};

/** 3D vector utilities for similarity */
const normalizeVec = (v) => {
  const x = Number(v[0]) || 0;
  const y = Number(v[1]) || 0;
  const z = Number(v[2]) || 0;
  const norm = Math.sqrt(x * x + y * y + z * z) || 1;
  return [x / norm, y / norm, z / norm];
};

const cosineSim = (a, b) => {
  const A = normalizeVec(a);
  const B = normalizeVec(b);
  return clamp(A[0] * B[0] + A[1] * B[1] + A[2] * B[2], -1, 1);
};

/** UI: Stat Card (colors aligned with AnalysisPage) */
const EnhancedStatCard = ({ title, value, subtitle, icon: Icon, color }) => {
  const colorClasses = {
    emerald: {
      text: "text-emerald-700",
      bg: "bg-emerald-50",
      border: "border-emerald-100",
      iconBg: "bg-emerald-100",
    },
    rose: {
      text: "text-rose-700",
      bg: "bg-rose-50",
      border: "border-rose-100",
      iconBg: "bg-rose-100",
    },
    blue: {
      text: "text-blue-700",
      bg: "bg-blue-50",
      border: "border-blue-100",
      iconBg: "bg-blue-100",
    },
    indigo: {
      text: "text-indigo-700",
      bg: "bg-indigo-50",
      border: "border-indigo-100",
      iconBg: "bg-indigo-100",
    },
  };
  const colors = colorClasses[color] || colorClasses.indigo;

  return (
    <div className={`rounded-2xl border ${colors.border} ${colors.bg} p-5 transition-all hover:shadow-md`}>
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <div className="flex items-center gap-3 mb-3">
            <div className={`p-2 rounded-xl ${colors.iconBg} ${colors.text}`}>
              {Icon && <Icon className="w-5 h-5" />}
            </div>
            <div className="text-sm font-semibold text-gray-700">{title}</div>
          </div>
          <div className={`text-3xl font-bold ${colors.text}`}>{value}</div>
          <div className="text-sm text-gray-500 mt-2">{subtitle}</div>
        </div>
      </div>
    </div>
  );
};

/** UI: Scatter - X = Lib - Auth, Y = Centrist */
const EnhancedScatterPlot = ({ items }) => {
  const W = 600;
  const H = 320;
  const pad = 45;

  const xMin = -100;
  const xMax = 100;
  const yMin = 0;
  const yMax = 100;

  const xTo = (v) => pad + ((v - xMin) / (xMax - xMin)) * (W - pad * 2);
  const yTo = (v) => pad + ((yMax - v) / (yMax - yMin)) * (H - pad * 2);

  const getQuadrant = (x, y) => {
    // Labels consistent with colors; "pragmatic" ~ higher Centrist
    if (x > 0 && y > 50) return { label: "Libertarian Pragmatic", color: "#10b981" }; // emerald-500
    if (x < 0 && y > 50) return { label: "Authoritarian Pragmatic", color: "#f43f5e" }; // rose-500
    if (x > 0 && y <= 50) return { label: "Libertarian Polarized", color: "#059669" }; // emerald-600
    if (x < 0 && y <= 50) return { label: "Authoritarian Polarized", color: "#e11d48" }; // rose-600
    return { label: CENTRIST, color: "#3b82f6" }; // blue-500
  };

  return (
    <div className="bg-white rounded-2xl border border-gray-200 p-6">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h3 className="font-semibold text-gray-900 flex items-center gap-2">
            <Globe className="w-5 h-5 text-indigo-600" />
            Ideological Positioning Map
          </h3>
          <p className="text-sm text-gray-500 mt-1">X = Libertarian − Authoritarian, Y = Centrist Score</p>
        </div>
      </div>

      <div className="relative">
        <svg width={W} height={H} className="rounded-lg border border-gray-100">
          {/* Grid */}
          <defs>
            <pattern id="gridPattern" width="40" height="40" patternUnits="userSpaceOnUse">
              <path d="M 40 0 L 0 0 0 40" fill="none" stroke="#f1f5f9" strokeWidth="1" />
            </pattern>
          </defs>

          <rect width={W} height={H} fill="url(#gridPattern)" />

          {/* Axes */}
          <line x1={pad} y1={H - pad} x2={W - pad} y2={H - pad} stroke="#cbd5e1" strokeWidth="1.5" />
          <line x1={pad} y1={pad} x2={pad} y2={H - pad} stroke="#cbd5e1" strokeWidth="1.5" />
          <line x1={xTo(0)} y1={pad} x2={xTo(0)} y2={H - pad} stroke="#94a3b8" strokeWidth="2" strokeDasharray="4 4" />

          {/* Labels */}
          <text x={W / 2} y={H - 10} fontSize="11" fill="#6b7280" textAnchor="middle">
            Libertarian − Authoritarian Balance
          </text>
          <text
            x={15}
            y={H / 2}
            fontSize="11"
            fill="#6b7280"
            textAnchor="middle"
            transform={`rotate(-90, 15, ${H / 2})`}
          >
            Centrist Score (%)
          </text>

          {/* Quadrant labels */}
          <text x={xTo(75)} y={yTo(85)} fontSize="10" fill="#059669" fontWeight="500">
            Libertarian
          </text>
          <text x={xTo(-75)} y={yTo(85)} fontSize="10" fill="#e11d48" fontWeight="500">
            Authoritarian
          </text>
          <text x={W / 2} y={yTo(15)} fontSize="10" fill="#475569" fontWeight="500">
            Polarized
          </text>
          <text x={W / 2} y={yTo(85)} fontSize="10" fill="#475569" fontWeight="500">
            Pragmatic
          </text>

          {/* Data points */}
          {items.map((it, idx) => {
            const r = clamp(5 + (it.evidence ? Math.sqrt(it.evidence) / 3 : 0), 5, 15);
            const opacity = clamp(0.5 + it.conf * 0.5, 0.5, 1);
            const quadrant = getQuadrant(it.x, it.y);

            return (
              <g key={idx}>
                <circle
                  cx={xTo(it.x)}
                  cy={yTo(it.y)}
                  r={r}
                  fill={quadrant.color}
                  opacity={opacity}
                  stroke="white"
                  strokeWidth="2"
                  className="transition-all duration-200 hover:opacity-100 cursor-pointer"
                />
                <title>
                  {it.title}
                  {"\n"}Speaker: {it.speaker}
                  {"\n"}Position: {quadrant.label}
                  {"\n"}X: {it.x.toFixed(1)} (Lib-Auth)
                  {"\n"}Y: {it.y.toFixed(1)}% Centrist
                  {"\n"}Confidence: {(it.conf * 100).toFixed(0)}%
                  {"\n"}Evidence: {it.evidence || "N/A"} segments
                </title>
              </g>
            );
          })}
        </svg>

        {/* Legend */}
        <div className="mt-6 p-4 bg-gray-50 rounded-xl">
          <div className="flex flex-wrap items-center gap-4">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-emerald-500" />
              <span className="text-sm text-gray-700">Libertarian Zones</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-rose-500" />
              <span className="text-sm text-gray-700">Authoritarian Zones</span>
            </div>
            <div className="flex items-center gap-2">
              <Scale className="w-4 h-4 text-blue-500" />
              <span className="text-sm text-gray-700">Y-axis = {CENTRIST}</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded-full bg-indigo-500/30" />
              <span className="text-sm text-gray-700">Point size ≈ Evidence</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded-full bg-indigo-500/70" />
              <span className="text-sm text-gray-700">Opacity ≈ Confidence</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

/** UI: Trend lines over time (Lib/Auth/Centrist) */
const EnhancedTrendChart = ({ series, speaker }) => {
  const W = 600;
  const H = 280;
  const pad = 40;

  const maxX = Math.max(series.length - 1, 1);
  const yMin = 0;
  const yMax = 100;

  const xTo = (i) => pad + (i / maxX) * (W - pad * 2);
  const yTo = (v) => pad + ((yMax - v) / (yMax - yMin)) * (H - pad * 2);

  const pathFor = (key) => {
    return series.map((s, i) => `${i === 0 ? "M" : "L"} ${xTo(i)} ${yTo(s[key])}`).join(" ");
  };

  const firstDate = series[0]?.date?.toLocaleDateString() || "";
  const lastDate = series[series.length - 1]?.date?.toLocaleDateString() || "";

  // Calculate trends
  const libChange = series.length >= 2 ? series[series.length - 1].lib - series[0].lib : 0;
  const authChange = series.length >= 2 ? series[series.length - 1].auth - series[0].auth : 0;

  return (
    <div className="bg-white rounded-2xl border border-gray-200 p-6">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h3 className="font-semibold text-gray-900 flex items-center gap-2">
            <Activity className="w-5 h-5 text-indigo-600" />
            Ideological Evolution {speaker && <span className="font-normal text-gray-600">• {speaker}</span>}
          </h3>
          <p className="text-sm text-gray-500 mt-1">Track ideological composition over time</p>
        </div>
        {series.length >= 2 && (
          <div
            className={`px-3 py-1.5 rounded-full text-sm font-medium ${
              Math.abs(libChange) > Math.abs(authChange)
                ? libChange > 0
                  ? "bg-emerald-100 text-emerald-800"
                  : "bg-rose-100 text-rose-800"
                : authChange > 0
                ? "bg-rose-100 text-rose-800"
                : "bg-emerald-100 text-emerald-800"
            }`}
          >
            {Math.abs(libChange) > Math.abs(authChange)
              ? libChange > 0
                ? "Net Libertarian Shift"
                : "Net Authoritarian Shift"
              : authChange > 0
              ? "Net Authoritarian Shift"
              : "Net Libertarian Shift"}
          </div>
        )}
      </div>

      <div className="relative">
        <svg width={W} height={H} className="rounded-lg border border-gray-100">
          {/* Grid lines */}
          {[0, 25, 50, 75, 100].map((v) => (
            <g key={v}>
              <line x1={pad} y1={yTo(v)} x2={W - pad} y2={yTo(v)} stroke="#f1f5f9" strokeWidth="1" />
              <text x={pad - 8} y={yTo(v) + 4} fontSize="10" fill="#94a3b8" textAnchor="end">
                {v}%
              </text>
            </g>
          ))}

          {/* Trend lines (colors aligned: emerald, rose, blue) */}
          <path d={pathFor("lib")} fill="none" stroke="#10b981" strokeWidth="3" strokeLinecap="round" />
          <path d={pathFor("auth")} fill="none" stroke="#f43f5e" strokeWidth="3" strokeLinecap="round" />
          <path d={pathFor("cen")} fill="none" stroke="#3b82f6" strokeWidth="3" strokeLinecap="round" />

          {/* Data points */}
          {series.map((s, i) => (
            <g key={i}>
              <circle cx={xTo(i)} cy={yTo(s.lib)} r="4" fill="#10b981" stroke="white" strokeWidth="2" />
              <circle cx={xTo(i)} cy={yTo(s.auth)} r="4" fill="#f43f5e" stroke="white" strokeWidth="2" />
              <circle cx={xTo(i)} cy={yTo(s.cen)} r="4" fill="#3b82f6" stroke="white" strokeWidth="2" />
              {/* Date labels */}
              {i === 0 && (
                <text x={xTo(i)} y={H - 15} fontSize="10" fill="#6b7280" textAnchor="start">
                  {firstDate}
                </text>
              )}
              {i === series.length - 1 && (
                <text x={xTo(i)} y={H - 15} fontSize="10" fill="#6b7280" textAnchor="end">
                  {lastDate}
                </text>
              )}
            </g>
          ))}
        </svg>

        {/* Legend */}
        <div className="flex items-center gap-4 mt-4">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-emerald-500" />
            <span className="text-sm text-gray-700">Libertarian</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-rose-500" />
            <span className="text-sm text-gray-700">Authoritarian</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-blue-500" />
            <span className="text-sm text-gray-700">Centrist</span>
          </div>
        </div>
      </div>
    </div>
  );
};

/** UI: Speech card in selection list */
const EnhancedSpeechCard = ({ speech, isSelected, onClick, index }) => {
  const { lib, auth, cen } = getScores(speech);
  const conf = getConfidence01(speech);
  const ev = getEvidenceCount(speech);
  const dt = getDateValue(speech);
  const speaker = getSpeaker(speech);
  const title = getTitle(speech);

  const speechId = speech?.id;
  const displayId = typeof speechId === "string" ? speechId.substring(0, 6) : String(speechId || "").substring(0, 6);

  return (
    <div
      onClick={onClick}
      className={`relative rounded-xl border p-4 transition-all cursor-pointer ${
        isSelected
          ? "border-indigo-500 bg-gradient-to-r from-indigo-50/50 to-white ring-2 ring-indigo-500/20 ring-offset-1"
          : "border-gray-200 bg-white hover:border-gray-300 hover:shadow-sm"
      }`}
    >
      {/* Selection indicator */}
      {isSelected && (
        <div className="absolute -top-2 -right-2 w-6 h-6 bg-indigo-500 rounded-full flex items-center justify-center">
          <CheckCircle className="w-4 h-4 text-white" />
        </div>
      )}

      <div className="flex items-start justify-between mb-3">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <div className={`w-2 h-2 rounded-full ${isSelected ? "bg-indigo-500" : "bg-gray-300"}`} />
            <h4 className="font-semibold text-sm text-gray-900 truncate">{title}</h4>
          </div>
          <p className="text-xs text-gray-500 truncate">{speaker}</p>
        </div>

        <div className="flex flex-col items-end gap-1">
          <div
            className={`text-xs px-2 py-1 rounded-full ${
              conf >= 0.8 ? "bg-emerald-100 text-emerald-800" : conf >= 0.6 ? "bg-amber-100 text-amber-800" : "bg-rose-100 text-rose-800"
            }`}
          >
            {Math.round(conf * 100)}% conf
          </div>
          {dt && (
            <div className="text-xs text-gray-500 flex items-center gap-1">
              <Calendar className="w-3 h-3" />
              {dt.toLocaleDateString()}
            </div>
          )}
        </div>
      </div>

      {/* Progress bars */}
      <div className="space-y-2 mb-4">
        <div>
          <div className="flex justify-between text-xs text-gray-600 mb-1">
            <span className="inline-flex items-center gap-1">
              <Shield className="w-3.5 h-3.5 text-emerald-500" />
              Libertarian
            </span>
            <span className="font-semibold">{lib.toFixed(1)}%</span>
          </div>
          <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-emerald-500 to-emerald-400 rounded-full transition-all duration-500"
              style={{ width: `${lib}%` }}
            />
          </div>
        </div>

        <div>
          <div className="flex justify-between text-xs text-gray-600 mb-1">
            <span className="inline-flex items-center gap-1">
              <Lock className="w-3.5 h-3.5 text-rose-500" />
              Authoritarian
            </span>
            <span className="font-semibold">{auth.toFixed(1)}%</span>
          </div>
          <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-rose-500 to-rose-400 rounded-full transition-all duration-500"
              style={{ width: `${auth}%` }}
            />
          </div>
        </div>

        <div>
          <div className="flex justify-between text-xs text-gray-600 mb-1">
            <span className="inline-flex items-center gap-1">
              <Scale className="w-3.5 h-3.5 text-blue-500" />
              Centrist
            </span>
            <span className="font-semibold">{cen.toFixed(1)}%</span>
          </div>
          <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-blue-500 to-blue-400 rounded-full transition-all duration-500"
              style={{ width: `${cen}%` }}
            />
          </div>
        </div>
      </div>

      {/* Footer */}
      <div className="flex items-center justify-between text-xs">
        <div className="flex items-center gap-2">
          <div className="flex items-center gap-1 text-gray-500">
            <Database className="w-3 h-3" />
            <span>{ev > 0 ? `${ev} seg` : "No evidence"}</span>
          </div>
          <span className="text-gray-300">•</span>
          <span className="text-gray-500">ID: {displayId || "N/A"}</span>
        </div>
        <div className="text-gray-400">#{index + 1}</div>
      </div>
    </div>
  );
};

/** Main Page */
const ComparisonPage = ({ projects, refreshProjects, loading }) => {
  // Only completed/has_analysis items
  const doneSpeeches = useMemo(() => {
    const arr = Array.isArray(projects) ? projects : [];
    return arr.filter((p) => {
      const st = String(p?.status || "").toLowerCase();
      const hasAnalysis = Boolean(p?.analysis_summary) || Boolean(p?.has_analysis);
      return st === "completed" || st === "done" || hasAnalysis;
    });
  }, [projects]);

  const MAX_SELECTED = 8;
  const [selectedIds, setSelectedIds] = useState([]);

  useEffect(() => {
    if (selectedIds.length === 0 && doneSpeeches.length > 0) {
      setSelectedIds(doneSpeeches.slice(0, 4).map((p) => p.id));
    }
  }, [doneSpeeches, selectedIds.length]);

  const [mode, setMode] = useState("cross"); // "cross" or "trend"
  const [query, setQuery] = useState("");
  const [speakerFilter, setSpeakerFilter] = useState("All speakers");
  const [expandedView, setExpandedView] = useState(false);

  const speakers = useMemo(() => {
    const set = new Set(doneSpeeches.map(getSpeaker));
    return ["All speakers", ...Array.from(set).sort((a, b) => a.localeCompare(b))];
  }, [doneSpeeches]);

  const filteredList = useMemo(() => {
    const q = (query || "").trim().toLowerCase();
    return doneSpeeches.filter((p) => {
      const sp = getSpeaker(p);
      if (speakerFilter !== "All speakers" && sp !== speakerFilter) return false;
      if (!q) return true;
      const title = getTitle(p).toLowerCase();
      const speaker = sp.toLowerCase();
      return title.includes(q) || speaker.includes(q);
    });
  }, [doneSpeeches, query, speakerFilter]);

  const toggleSelection = (id) => {
    setSelectedIds((prev) => {
      if (prev.includes(id)) return prev.filter((x) => x !== id);
      if (prev.length >= MAX_SELECTED) return prev;
      return [...prev, id];
    });
  };

  const selectedRaw = useMemo(
    () => doneSpeeches.filter((p) => selectedIds.includes(p.id)),
    [doneSpeeches, selectedIds]
  );

  const selected = useMemo(() => {
    const items = [...selectedRaw];
    if (mode === "trend") {
      items.sort((a, b) => {
        const da = getDateValue(a);
        const db = getDateValue(b);
        if (!da && !db) return 0;
        if (!da) return 1;
        if (!db) return -1;
        return da - db;
      });
    }
    return items;
  }, [selectedRaw, mode]);

  // Derived items for visualization
  const vizItems = useMemo(() => {
    return selected.map((p) => {
      const s = getScores(p);
      const dt = getDateValue(p);
      const conf = getConfidence01(p);
      const ev = getEvidenceCount(p);
      const speaker = getSpeaker(p);
      const title = getTitle(p);

      const x = s.lib - s.auth;
      const y = s.cen;

      return {
        id: String(p?.id || ""),
        title,
        short: title.length > 14 ? title.slice(0, 14) + "…" : title,
        speaker,
        date: dt,
        lib: s.lib,
        auth: s.auth,
        cen: s.cen,
        conf,
        evidence: ev,
        x,
        y,
        vec: [s.lib, s.auth, s.cen],
      };
    });
  }, [selected]);

  const trendItems = useMemo(() => {
    if (mode !== "trend") return [];
    if (vizItems.length === 0) return [];

    if (speakerFilter !== "All speakers") {
      return vizItems.filter((x) => x.speaker === speakerFilter && x.date);
    }

    // pick speaker with most items by default
    const counts = new Map();
    for (const it of vizItems) {
      counts.set(it.speaker, (counts.get(it.speaker) || 0) + 1);
    }
    let bestSpeaker = vizItems[0]?.speaker || "";
    let bestCount = -1;
    for (const [sp, c] of counts.entries()) {
      if (c > bestCount) {
        bestCount = c;
        bestSpeaker = sp;
      }
    }
    return vizItems.filter((x) => x.speaker === bestSpeaker && x.date);
  }, [mode, vizItems, speakerFilter]);

  const simpleStats = useMemo(() => {
    if (selected.length === 0) return null;
    const avg = (arr) => (arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0);
    const lib = avg(selected.map((p) => getScores(p).lib));
    const auth = avg(selected.map((p) => getScores(p).auth));
    const cen = avg(selected.map((p) => getScores(p).cen));
    return { lib, auth, cen };
  }, [selected]);

  /** Similarity Matrix (cosine on [lib, auth, cen]) */
  const SimilarityMatrix = ({ items }) => {
    const n = items.length;
    if (n < 2) return null;

    const sims = [];
    for (let i = 0; i < n; i++) {
      sims[i] = [];
      for (let j = 0; j < n; j++) {
        sims[i][j] = cosineSim(items[i].vec, items[j].vec);
      }
    }

    const cellClass = (v) => {
      if (v >= 0.95) return "bg-emerald-100 border-emerald-200 text-emerald-900";
      if (v >= 0.9) return "bg-emerald-50 border-emerald-200 text-emerald-900";
      if (v >= 0.8) return "bg-indigo-50 border-indigo-200 text-indigo-900";
      if (v >= 0.7) return "bg-amber-50 border-amber-200 text-amber-900";
      return "bg-rose-50 border-rose-200 text-rose-900";
    };

    return (
      <div className="bg-white rounded-2xl border border-gray-200 p-6">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h3 className="font-semibold text-gray-900 flex items-center gap-2">
              <GitCompare className="w-5 h-5 text-indigo-600" />
              Rhetorical Similarity Matrix
            </h3>
            <p className="text-sm text-gray-500 mt-1">Cosine similarity of ideological vectors</p>
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full border-collapse">
            <thead>
              <tr>
                <th className="text-sm text-gray-500 font-semibold text-left p-3 bg-gray-50 rounded-l-xl">Speech</th>
                {items.map((it, idx) => (
                  <th key={String(it.id) || idx} className="text-sm text-gray-500 font-semibold text-center p-3 bg-gray-50">
                    #{idx + 1}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {items.map((row, i) => (
                <tr key={String(row.id) || i}>
                  <td className="p-3 bg-gray-50 rounded-l-xl border-t border-gray-100">
                    <div className="text-sm font-medium text-gray-900 truncate max-w-[150px]">{row.short}</div>
                    <div className="text-xs text-gray-500">{row.speaker}</div>
                  </td>
                  {items.map((col, j) => {
                    const value = sims[i][j];
                    const isDiagonal = i === j;
                    return (
                      <td key={`${String(row.id)}-${String(col.id)}`} className="border-t border-gray-100">
                        <div
                          className={`
                          flex items-center justify-center p-3 
                          ${isDiagonal ? "rounded-full mx-auto w-10 h-10" : "rounded-lg"}
                          ${cellClass(value)}
                          transition-all hover:scale-105
                        `}
                          title={`${row.title} vs ${col.title}\nSimilarity: ${value.toFixed(3)}`}
                        >
                          <span className={`text-sm font-bold ${isDiagonal ? "text-gray-400" : ""}`}>
                            {isDiagonal ? "—" : value.toFixed(2)}
                          </span>
                        </div>
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="mt-6 p-4 bg-gray-50 rounded-xl">
          <div className="flex flex-wrap items-center gap-4 text-sm">
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded bg-emerald-100 border border-emerald-200" />
              <span className="text-gray-700">≥ 0.90 (Very High)</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded bg-indigo-50 border border-indigo-200" />
              <span className="text-gray-700">0.80–0.90 (High)</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded bg-amber-50 border border-amber-200" />
              <span className="text-gray-700">0.70–0.80 (Moderate)</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded bg-rose-50 border border-rose-200" />
              <span className="text-gray-700">≤ 0.70 (Low)</span>
            </div>
          </div>
        </div>
      </div>
    );
  };

  const evidenceWarning = useMemo(() => {
    if (selected.length < 2) return "Select at least 2 speeches for a meaningful comparison.";
    // Optional: evidence volume and mean confidence hints can be reintroduced if needed.
    return null;
  }, [selected]);

  return (
    <div className={`${expandedView ? "fixed inset-0 z-50 bg-white p-6 overflow-auto" : "space-y-6"}`}>
      {/* Header */}
      <div className={`flex items-center justify-between ${expandedView ? "mb-8" : ""}`}>
        <div>
          <div className="flex items-center gap-3 mb-2">
            <div className="p-2 rounded-xl bg-indigo-100">
              <BarChart3 className="w-6 h-6 text-indigo-600" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-gray-900">Advanced Ideology Comparison</h1>
              <p className="text-sm text-gray-600 mt-1">
                Compare ideological balance across speeches and track speaker evolution over time
              </p>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-3">
          {expandedView ? (
            <button
              onClick={() => setExpandedView(false)}
              className="inline-flex items-center px-4 py-2 rounded-lg border border-gray-300 text-sm text-gray-700 hover:bg-gray-50"
              type="button"
            >
              <X className="w-4 h-4 mr-2" />
              Exit Fullscreen
            </button>
          ) : (
            <button
              onClick={() => setExpandedView(true)}
              className="inline-flex items-center px-4 py-2 rounded-lg border border-gray-300 text-sm text-gray-700 hover:bg-gray-50"
              type="button"
            >
              <Maximize2 className="w-4 h-4 mr-2" />
              Fullscreen
            </button>
          )}
          <button
            onClick={refreshProjects}
            className="inline-flex items-center px-4 py-2 rounded-lg bg-indigo-600 text-white font-medium hover:bg-indigo-700"
            type="button"
          >
            {loading ? <Loader2 className="w-4 h-4 mr-2 animate-spin" /> : <RefreshCw className="w-4 h-4 mr-2" />}
            Refresh
          </button>
        </div>
      </div>

      {/* Control Panel */}
      <div className="bg-white rounded-2xl border border-gray-200 p-5">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
          <div className="flex flex-wrap items-center gap-3">
            <div className="inline-flex items-center gap-2 text-sm font-semibold text-gray-700">
              <Filter className="w-4 h-4" />
              Analysis Mode
            </div>

            <div className="inline-flex rounded-xl border border-gray-300 overflow-hidden bg-gray-100 p-1">
              <button
                onClick={() => setMode("cross")}
                className={`px-4 py-2 text-sm font-semibold rounded-lg transition-all ${
                  mode === "cross"
                    ? "bg-gradient-to-r from-indigo-500 to-indigo-600 text-white shadow-md"
                    : "text-gray-700 hover:bg-white"
                }`}
                type="button"
              >
                Across Speeches
              </button>
              <button
                onClick={() => setMode("trend")}
                className={`px-4 py-2 text-sm font-semibold rounded-lg transition-all ${
                  mode === "trend"
                    ? "bg-gradient-to-r from-indigo-500 to-indigo-600 text-white shadow-md"
                    : "text-gray-700 hover:bg-white"
                }`}
                type="button"
              >
                Speaker Evolution
              </button>
            </div>

            <div className="relative">
              <Search className="w-4 h-4 text-gray-400 absolute left-3 top-1/2 -translate-y-1/2" />
              <input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Search speeches or speakers…"
                className="pl-10 pr-4 py-2.5 text-sm border border-gray-300 rounded-xl bg-white focus:outline-none focus:ring-2 focus:ring-indigo-500"
              />
            </div>

            <select
              value={speakerFilter}
              onChange={(e) => setSpeakerFilter(e.target.value)}
              className="px-4 py-2.5 text-sm border border-gray-300 rounded-xl bg-white focus:outline-none focus:ring-2 focus:ring-indigo-500"
            >
              {speakers.map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
          </div>

          <div className="flex items-center gap-4">
            <div className="text-sm text-gray-600">
              <span className="font-semibold">{selectedIds.length}</span> selected •
              <span className="font-semibold"> {filteredList.length}</span> available
            </div>
            <button
              onClick={() => setSelectedIds([])}
              className="text-sm px-3 py-1.5 rounded-lg border border-gray-300 hover:bg-gray-50"
              type="button"
            >
              Clear All
            </button>
          </div>
        </div>

        {selected.length < 2 && (
          <div className="mt-4 text-sm text-amber-800 bg-amber-50 border border-amber-200 rounded-xl p-3 inline-flex items-center gap-2">
            <AlertTriangle className="w-4 h-4" />
            Select at least 2 speeches for a meaningful comparison.
          </div>
        )}
      </div>

      {/* Stats Overview */}
      {simpleStats && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <EnhancedStatCard
            title="Average Libertarian"
            value={`${simpleStats.lib.toFixed(1)}%`}
            subtitle="Individual freedom emphasis"
            icon={Shield}
            color="emerald"
          />
          <EnhancedStatCard
            title="Average Authoritarian"
            value={`${simpleStats.auth.toFixed(1)}%`}
            subtitle="Central authority emphasis"
            icon={Lock}
            color="rose"
          />
          <EnhancedStatCard
            title="Average Centrist"
            value={`${simpleStats.cen.toFixed(1)}%`}
            subtitle="Balanced/mixed content"
            icon={Scale}
            color="blue"
          />
          <EnhancedStatCard
            title="Selected Speeches"
            value={selected.length}
            subtitle={`of ${MAX_SELECTED} maximum`}
            icon={Database}
            color="indigo"
          />
        </div>
      )}

      {/* Main Content */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Panel - Speech Selection */}
        <div className="lg:col-span-1">
          <div className="bg-white rounded-2xl border border-gray-200 p-5 sticky top-6">
            <div className="flex items-center justify-between mb-6">
              <h2 className="font-semibold text-gray-900 flex items-center gap-2">
                <Users className="w-5 h-5 text-blue-500" />
                Available Speeches
              </h2>
              <div className="text-sm text-gray-500">{selectedIds.length}/{MAX_SELECTED}</div>
            </div>

            {loading ? (
              <div className="text-center py-12">
                <Loader2 className="w-8 h-8 animate-spin text-gray-400 mx-auto" />
                <p className="text-sm text-gray-500 mt-3">Loading speeches...</p>
              </div>
            ) : filteredList.length === 0 ? (
              <div className="text-center py-12 bg-gray-50 rounded-xl">
                <div className="w-12 h-12 mx-auto mb-4 rounded-xl bg-gray-200 flex items-center justify-center">
                  <Search className="w-6 h-6 text-gray-500" />
                </div>
                <h4 className="font-medium text-gray-900">No speeches found</h4>
                <p className="text-sm text-gray-500 mt-1">Try adjusting your search filters</p>
              </div>
            ) : (
              <div className="space-y-3 max-h-[600px] overflow-y-auto pr-2">
                {filteredList.map((speech, index) => (
                  <EnhancedSpeechCard
                    key={`${String(speech.id)}-${index}`}
                    speech={speech}
                    isSelected={selectedIds.includes(speech.id)}
                    onClick={() => toggleSelection(speech.id)}
                    index={index}
                  />
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Right Panel - Visualizations */}
        <div className="lg:col-span-2 space-y-6">
          {selected.length === 0 ? (
            <div className="bg-white rounded-2xl border border-gray-200 p-12 text-center">
              <div className="w-16 h-16 mx-auto mb-6 rounded-2xl bg-gradient-to-br from-indigo-100 to-indigo-200 flex items-center justify-center">
                <Eye className="w-8 h-8 text-indigo-600" />
              </div>
              <h3 className="text-lg font-semibold text-gray-900 mb-2">No Speeches Selected</h3>
              <p className="text-gray-600 max-w-md mx-auto mb-6">
                Select speeches from the left panel to begin analysis.
                Choose 2–8 speeches for cross-comparison or multiple speeches from one speaker for trend analysis.
              </p>
              <button
                onClick={() => {
                  if (filteredList.length > 0) {
                    setSelectedIds(filteredList.slice(0, 4).map((p) => p.id));
                  }
                }}
                className="inline-flex items-center px-5 py-2.5 rounded-xl bg-gradient-to-r from-indigo-500 to-indigo-600 text-white font-medium hover:shadow-lg"
                type="button"
              >
                <Sparkles className="w-4 h-4 mr-2" />
                Auto-select 4 Speeches
              </button>
            </div>
          ) : mode === "trend" ? (
            trendItems.length < 2 ? (
              <div className="bg-white rounded-2xl border border-gray-200 p-8 text-center">
                <div className="w-16 h-16 mx-auto mb-6 rounded-2xl bg-gradient-to-br from-amber-100 to-amber-200 flex items-center justify-center">
                  <Clock className="w-8 h-8 text-amber-600" />
                </div>
                <h3 className="text-lg font-semibold text-gray-900 mb-2">Need More Data for Trend Analysis</h3>
                <p className="text-gray-600 max-w-md mx-auto">
                  Trend analysis requires at least 2 speeches from the same speaker with dates.
                  Select multiple speeches by one speaker or use the speaker filter above.
                </p>
              </div>
            ) : (
              <EnhancedTrendChart
                series={trendItems.map((x) => ({
                  date: x.date,
                  lib: x.lib,
                  auth: x.auth,
                  cen: x.cen,
                }))}
                speaker={trendItems[0]?.speaker}
              />
            )
          ) : (
            <>
              <EnhancedScatterPlot
                items={vizItems.map((x) => ({
                  title: x.title,
                  speaker: x.speaker,
                  x: x.x,
                  y: x.y,
                  conf: x.conf,
                  evidence: x.evidence,
                }))}
              />
              {vizItems.length >= 2 && <SimilarityMatrix items={vizItems} />}
            </>
          )}

          {/* Analysis Summary */}
          {selected.length > 0 && (
            <div className="bg-gradient-to-br from-indigo-50/50 to-indigo-100/30 rounded-2xl border border-indigo-200 p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="font-semibold text-indigo-900 flex items-center gap-2">
                  <Cpu className="w-5 h-5" />
                  Analysis Insights
                </h3>
                <div className="text-sm text-indigo-700">
                  {mode === "cross" ? "Cross-speech Analysis" : "Temporal Analysis"}
                </div>
              </div>

              <div className="space-y-3">
                {selected.length >= 2 && (
                  <div className="text-sm text-indigo-900">
                    <span className="font-semibold">Selected {selected.length} speeches</span>
                    <span>
                      {" "}
                      • Averages: L {pct(simpleStats.lib)}, A {pct(simpleStats.auth)}, C {pct(simpleStats.cen)}
                    </span>
                  </div>
                )}

                {mode === "cross" && selected.length >= 2 && (
                  <div className="text-sm text-indigo-900">
                    <span className="font-semibold">Spread analysis:</span>
                    <span>
                      {" "}
                      Libertarian range: {Math.min(...vizItems.map((i) => i.lib)).toFixed(1)}% to{" "}
                      {Math.max(...vizItems.map((i) => i.lib)).toFixed(1)}%
                    </span>
                    <span>
                      {" "}
                      • Authoritarian range: {Math.min(...vizItems.map((i) => i.auth)).toFixed(1)}% to{" "}
                      {Math.max(...vizItems.map((i) => i.auth)).toFixed(1)}%
                    </span>
                  </div>
                )}

                {mode === "trend" && trendItems.length >= 2 && (
                  <div className="text-sm text-indigo-900">
                    <span className="font-semibold">Temporal shift detected:</span>
                    <span>
                      {" "}
                      From {trendItems[0].date?.toLocaleDateString()} to{" "}
                      {trendItems[trendItems.length - 1].date?.toLocaleDateString()}
                    </span>
                    <span>
                      {" "}
                      • Net change: L{" "}
                      {(trendItems[trendItems.length - 1].lib - trendItems[0].lib).toFixed(1)}%, A{" "}
                      {(trendItems[trendItems.length - 1].auth - trendItems[0].auth).toFixed(1)}%
                    </span>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>

      {!expandedView && (
        <div className="text-center text-sm text-gray-500 pt-6 border-t border-gray-200">
          <p>Advanced Ideology Comparison Dashboard • Centrist-only policy</p>
        </div>
      )}
    </div>
  );
};

export default ComparisonPage;
