import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Activity,
  AlertTriangle,
  BarChart3,
  Calendar,
  CheckCircle,
  Filter,
  GitCompare,
  Loader2,
  Lock,
  Maximize2,
  RefreshCw,
  Scale,
  Search,
  Shield,
  TrendingUp,
  Users,
  X,
} from "lucide-react";
import * as api from "../services/api";

const LIB = "Libertarian";
const AUTH = "Authoritarian";
const ECON_LEFT = "Economic-Left";
const ECON_RIGHT = "Economic-Right";

const clamp = (n, min, max) => Math.max(min, Math.min(max, n));
const toInt = (v, d = 0) => {
  const n = Number(v);
  return Number.isFinite(n) ? Math.max(0, Math.floor(n)) : d;
};
const toFloat = (v, d = 0) => {
  const n = Number(v);
  return Number.isFinite(n) ? n : d;
};

const familyConfig = {
  [LIB]: { icon: Shield, label: "Libertarian", pill: "bg-emerald-100 text-emerald-800", bar: "from-emerald-500 to-emerald-400" },
  [AUTH]: { icon: Lock, label: "Authoritarian", pill: "bg-rose-100 text-rose-800", bar: "from-rose-500 to-rose-400" },
  [ECON_LEFT]: { icon: Scale, label: "Economic-Left", pill: "bg-blue-100 text-blue-800", bar: "from-blue-500 to-blue-400" },
  [ECON_RIGHT]: { icon: TrendingUp, label: "Economic-Right", pill: "bg-amber-100 text-amber-800", bar: "from-amber-500 to-amber-400" },
};

const normalizeFamily = (fam) => {
  const raw = String(fam ?? "").trim();
  if (!raw) return null;
  if (raw === LIB || raw === AUTH || raw === ECON_LEFT || raw === ECON_RIGHT) return raw;

  const s = raw
    .toLowerCase()
    .replace(/[_\s]+/g, "-")
    .replace(/[^\w-]+/g, "")
    .trim();

  const map = {
    libertarian: LIB,
    // authoritar ian: AUTH,
    authoritarian: AUTH,
    "economic-left": ECON_LEFT,
    "economic-right": ECON_RIGHT,
    "econ-left": ECON_LEFT,
    "econ-right": ECON_RIGHT,
    "economicleft": ECON_LEFT,
    "economicright": ECON_RIGHT,
    "economic-leftist": ECON_LEFT,
    "economic-rightist": ECON_RIGHT,
    "lib": LIB,
    "auth": AUTH,
  };

  return map[s] || null;
};

const getIdKey = (p) => String(p?.id ?? p?.speech_id ?? p?.speechId ?? p?._id ?? "").trim();
const getSpeaker = (p) => String(p?.speaker_name ?? p?.speaker ?? p?.speakerName ?? "Unknown speaker").trim();
const getTitle = (p) => String(p?.title ?? p?.name ?? "Untitled").trim();

const getDateValue = (p) => {
  const d = p?.date || p?.analyzed_at || p?.created_at || p?.updated_at || null;
  if (!d) return null;
  const dt = new Date(d);
  return Number.isNaN(dt.getTime()) ? null : dt;
};

const getAnalysisRoot = (obj) => {
  if (!obj || typeof obj !== "object") return obj;
  return obj.analysis || obj.result || obj.payload || obj.data || obj;
};

const toArrayCandidate = (x) => {
  if (!x) return [];
  if (Array.isArray(x)) return x;
  if (typeof x === "object") {
    const cands = [x.items, x.units, x.segments, x.statements, x.results, x.data];
    for (const c of cands) if (Array.isArray(c)) return c;
  }
  return [];
};

const extractEvidenceItems = (analysisIn) => {
  const analysis = getAnalysisRoot(analysisIn);
  if (!analysis || typeof analysis !== "object") return [];

  const out = [];

  const push = (x) => {
    const arr = toArrayCandidate(x);
    if (arr.length) out.push(...arr);
  };

  push(analysis.segments);
  push(analysis.sections);
  push(analysis.evidence_segments);
  push(analysis.statement_list);
  push(analysis.key_statements);
  push(analysis.key_segments);
  push(analysis.highlights);
  push(analysis.key_evidence);
  push(analysis.statements);
  push(analysis.sentence_segments);

  push(analysis.evidence_items);
  push(analysis.evidence_units);
  push(analysis.evidenceUnits);
  push(analysis.evidence);

  if (analysis.evidence && typeof analysis.evidence === "object") {
    push(analysis.evidence.items);
    push(analysis.evidence.units);
    push(analysis.evidence.segments);
    push(analysis.evidence.statements);
  }

  if (Array.isArray(analysis.sections)) {
    for (const sec of analysis.sections) {
      if (!sec || typeof sec !== "object") continue;
      push(sec.evidence_items);
      push(sec.evidence_units);
      push(sec.items);
      push(sec.units);
      push(sec.segments);
      push(sec.statements);
      push(sec.key_statements);
    }
  }

  return out.filter((x) => x && typeof x === "object");
};

const countsFromAggregates = (analysis) => {
  const a = analysis && typeof analysis === "object" ? analysis : null;
  if (!a) return null;

  const dicts = [
    a.evidence_family_counts,
    a.family_counts,
    a.counts,
    a.ideology_family_counts,
    a.ideology_counts,
    a.summary?.family_counts,
    a.summary?.evidence_family_counts,
    a.speech_level?.family_counts,
    a.speech_level?.evidence_family_counts,
  ];

  const pick = (o, keys) => {
    if (!o || typeof o !== "object") return null;
    for (const k of keys) {
      if (o[k] != null) return toInt(o[k], 0);
    }
    return null;
  };

  for (const d of dicts) {
    if (!d || typeof d !== "object") continue;

    const lib = pick(d, ["Libertarian", "libertarian", "lib"]);
    const auth = pick(d, ["Authoritarian", "authoritarian", "auth"]);
    const eLeft = pick(d, ["Economic-Left", "economic-left", "economic_left", "econ-left"]);
    const eRight = pick(d, ["Economic-Right", "economic-right", "economic_right", "econ-right"]);

    if ([lib, auth, eLeft, eRight].some((x) => x != null)) {
      const L = lib ?? 0;
      const A = auth ?? 0;
      const EL = eLeft ?? 0;
      const ER = eRight ?? 0;
      const total = L + A + EL + ER;
      return { lib: L, auth: A, eLeft: EL, eRight: ER, total };
    }
  }

  return null;
};

const countsFromAnalysis = (analysisIn) => {
  const analysis = getAnalysisRoot(analysisIn) || {};
  const items = extractEvidenceItems(analysis);

  let lib = 0;
  let auth = 0;
  let eLeft = 0;
  let eRight = 0;

  let confSum = 0;
  let confN = 0;

  if (items.length === 0) {
    const agg = countsFromAggregates(analysis);
    if (agg) {
      lib = agg.lib;
      auth = agg.auth;
      eLeft = agg.eLeft;
      eRight = agg.eRight;
    }
  } else {
    for (const it of items) {
      const famRaw =
        it.ideology_family ??
        it.family ??
        it.dominant_family ??
        it.label_family ??
        it.family_name ??
        it.political_family ??
        it.ideology?.family ??
        it.label?.family ??
        it.classification?.family;

      const fam = normalizeFamily(famRaw);
      if (fam === LIB) lib += 1;
      else if (fam === AUTH) auth += 1;
      else if (fam === ECON_LEFT) eLeft += 1;
      else if (fam === ECON_RIGHT) eRight += 1;

      const cRaw =
        it.confidence_score ??
        it.confidence ??
        it.confidence01 ??
        it.confidence_pct ??
        it.score ??
        it.probability ??
        it.label_confidence;

      const cNum = toFloat(cRaw, 0);
      const c01 = cNum > 1.01 ? clamp(cNum / 100, 0, 1) : clamp(cNum, 0, 1);
      if (Number.isFinite(c01) && c01 > 0) {
        confSum += c01;
        confN += 1;
      }
    }
  }

  const total = lib + auth + eLeft + eRight;

  let avgConfidence = 0;
  if (confN > 0) avgConfidence = confSum / confN;

  const block = analysis?.ideology_2d || analysis?.speech_level?.ideology_2d || analysis?.metadata?.ideology_2d || null;

  let coords = null;
  if (block && typeof block === "object" && block.coordinates && typeof block.coordinates === "object") {
    const x = clamp(toFloat(block.coordinates.economic ?? block.coordinates.x ?? 0, 0), -1, 1);
    const y = clamp(toFloat(block.coordinates.social ?? block.coordinates.y ?? 0, 0), -1, 1);
    coords = { x, y, source: "backend" };
  } else {
    const socialTot = lib + auth;
    const econTot = eLeft + eRight;
    const y = socialTot > 0 ? (lib - auth) / socialTot : 0;
    const x = econTot > 0 ? (eRight - eLeft) / econTot : 0;
    coords = { x: clamp(x, -1, 1), y: clamp(y, -1, 1), source: "derived" };
  }

  return { lib, auth, eLeft, eRight, total, avgConfidence, coords };
};

const normalizeVec = (v) => {
  const arr = Array.isArray(v) ? v.map((x) => Number(x) || 0) : [0, 0, 0, 0];
  const norm = Math.sqrt(arr.reduce((s, x) => s + x * x, 0)) || 1;
  return arr.map((x) => x / norm);
};

const cosineSim = (a, b) => {
  const A = normalizeVec(a);
  const B = normalizeVec(b);
  const dot = A.reduce((s, x, i) => s + x * (B[i] ?? 0), 0);
  return clamp(dot, -1, 1);
};

const StatCard = ({ title, value, subtitle, icon: Icon, tone = "indigo" }) => {
  const toneMap = {
    emerald: { text: "text-emerald-700", bg: "bg-emerald-50", border: "border-emerald-100", iconBg: "bg-emerald-100" },
    rose: { text: "text-rose-700", bg: "bg-rose-50", border: "border-rose-100", iconBg: "bg-rose-100" },
    blue: { text: "text-blue-700", bg: "bg-blue-50", border: "border-blue-100", iconBg: "bg-blue-100" },
    amber: { text: "text-amber-700", bg: "bg-amber-50", border: "border-amber-100", iconBg: "bg-amber-100" },
    indigo: { text: "text-indigo-700", bg: "bg-indigo-50", border: "border-indigo-100", iconBg: "bg-indigo-100" },
  };
  const t = toneMap[tone] || toneMap.indigo;

  return (
    <div className={`rounded-2xl border ${t.border} ${t.bg} p-5 transition-all hover:shadow-md`}>
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <div className="flex items-center gap-3 mb-3">
            <div className={`p-2 rounded-xl ${t.iconBg} ${t.text}`}>{Icon ? <Icon className="w-5 h-5" /> : null}</div>
            <div className="text-sm font-semibold text-gray-700">{title}</div>
          </div>
          <div className={`text-3xl font-bold ${t.text}`}>{value}</div>
          <div className="text-sm text-gray-500 mt-2">{subtitle}</div>
        </div>
      </div>
    </div>
  );
};

const SpeechCard = ({ speech, derived, isSelected, onClick, index, isLoadingAnalysis, analysisError }) => {
  const conf = clamp(toFloat(derived?.avgConfidence ?? 0, 0), 0, 1);
  const total = toInt(derived?.total ?? 0, 0);

  const dt = getDateValue(speech);
  const speaker = getSpeaker(speech);
  const title = getTitle(speech);

  const speechId = speech?.id ?? speech?.speech_id ?? speech?.speechId ?? speech?._id;
  const displayId = typeof speechId === "string" ? speechId.substring(0, 6) : String(speechId || "").substring(0, 6);

  const shares = (() => {
    if (!derived || derived.total <= 0) return { lib: 0, auth: 0, eLeft: 0, eRight: 0 };
    const t = derived.total || 1;
    return { lib: (derived.lib / t) * 100, auth: (derived.auth / t) * 100, eLeft: (derived.eLeft / t) * 100, eRight: (derived.eRight / t) * 100 };
  })();

  return (
    <div
      onClick={onClick}
      className={`relative rounded-xl border p-4 transition-all cursor-pointer ${
        isSelected
          ? "border-indigo-500 bg-gradient-to-r from-indigo-50/50 to-white ring-2 ring-indigo-500/20 ring-offset-1"
          : "border-gray-200 bg-white hover:border-gray-300 hover:shadow-sm"
      }`}
    >
      {isSelected && (
        <div className="absolute -top-2 -right-2 w-6 h-6 bg-indigo-500 rounded-full flex items-center justify-center">
          <CheckCircle className="w-4 h-4 text-white" />
        </div>
      )}

      <div className="flex items-start justify-between mb-3 gap-3">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1 min-w-0">
            <div className={`w-2 h-2 rounded-full ${isSelected ? "bg-indigo-500" : "bg-gray-300"}`} />
            <h4 className="font-semibold text-sm text-gray-900 truncate">{title}</h4>
          </div>
          <p className="text-xs text-gray-500 truncate">{speaker}</p>

          {analysisError ? (
            <div className="mt-2 text-xs text-rose-700 bg-rose-50 border border-rose-200 rounded-lg px-2 py-1 inline-flex items-center gap-2">
              <AlertTriangle className="w-3.5 h-3.5" />
              {String(analysisError).slice(0, 80)}
            </div>
          ) : null}
        </div>

        <div className="flex flex-col items-end gap-1 flex-shrink-0">
          <div
            className={`text-xs px-2 py-1 rounded-full ${
              isLoadingAnalysis
                ? "bg-gray-100 text-gray-700"
                : conf >= 0.8
                ? "bg-emerald-100 text-emerald-800"
                : conf >= 0.6
                ? "bg-amber-100 text-amber-800"
                : "bg-rose-100 text-rose-800"
            }`}
            title={isLoadingAnalysis ? "Loading analysis…" : "Avg confidence (from evidence items)"}
          >
            {isLoadingAnalysis ? (
              <span className="inline-flex items-center gap-2">
                <Loader2 className="w-3.5 h-3.5 animate-spin" />
                Loading
              </span>
            ) : (
              `${Math.round(conf * 100)}% conf`
            )}
          </div>
          {dt && (
            <div className="text-xs text-gray-500 flex items-center gap-1">
              <Calendar className="w-3 h-3" />
              {dt.toLocaleDateString()}
            </div>
          )}
        </div>
      </div>

      <div className="space-y-2 mb-4">
        {[{ k: LIB, v: shares.lib }, { k: AUTH, v: shares.auth }, { k: ECON_LEFT, v: shares.eLeft }, { k: ECON_RIGHT, v: shares.eRight }].map(({ k, v }) => {
          const cfg = familyConfig[k];
          const Icon = cfg.icon;
          return (
            <div key={k}>
              <div className="flex justify-between text-xs text-gray-600 mb-1">
                <span className="inline-flex items-center gap-1">
                  <Icon className="w-3.5 h-3.5" />
                  {cfg.label}
                </span>
                <span className="font-semibold">{v.toFixed(1)}%</span>
              </div>
              <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
                <div className={`h-full bg-gradient-to-r ${cfg.bar} rounded-full transition-all duration-500`} style={{ width: `${v}%` }} />
              </div>
            </div>
          );
        })}
      </div>

      <div className="flex items-center justify-between text-xs">
        <div className="flex items-center gap-2">
          <span className="text-gray-500">{isLoadingAnalysis ? "Loading evidence…" : total > 0 ? `${total} evidence units` : "No evidence found"}</span>
          <span className="text-gray-300">•</span>
          <span className="text-gray-500">ID: {displayId || "N/A"}</span>
        </div>
        <div className="text-gray-400">#{index + 1}</div>
      </div>
    </div>
  );
};

const IdeologyScatter2D = ({ items }) => {
  const W = 640;
  const H = 360;
  const pad = 50;

  const xMin = -1;
  const xMax = 1;
  const yMin = -1;
  const yMax = 1;

  const xTo = (v) => pad + ((v - xMin) / (xMax - xMin)) * (W - pad * 2);
  const yTo = (v) => pad + ((yMax - v) / (yMax - yMin)) * (H - pad * 2);

  const labelFor = (x, y) => {
    const econ = Math.abs(x) >= 0.25 ? (x > 0 ? "Right" : "Left") : "";
    const soc = Math.abs(y) >= 0.25 ? (y > 0 ? "Libertarian" : "Authoritarian") : "";
    if (econ && soc) return `${soc}-${econ}`;
    return soc || econ || "—";
  };

  return (
    <div className="bg-white rounded-2xl border border-gray-200 p-6">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h3 className="font-semibold text-gray-900 flex items-center gap-2">
            <BarChart3 className="w-5 h-5 text-indigo-600" />
            2D Evidence Map
          </h3>
          <p className="text-sm text-gray-500 mt-1">X = Economic (Left ↔ Right), Y = Social (Authoritarian ↔ Libertarian)</p>
        </div>
      </div>

      <svg width={W} height={H} className="rounded-lg border border-gray-100">
        <defs>
          <pattern id="grid2d" width="40" height="40" patternUnits="userSpaceOnUse">
            <path d="M 40 0 L 0 0 0 40" fill="none" stroke="#f1f5f9" strokeWidth="1" />
          </pattern>
        </defs>

        <rect width={W} height={H} fill="url(#grid2d)" />

        <line x1={pad} y1={H - pad} x2={W - pad} y2={H - pad} stroke="#cbd5e1" strokeWidth="1.5" />
        <line x1={pad} y1={pad} x2={pad} y2={H - pad} stroke="#cbd5e1" strokeWidth="1.5" />

        <line x1={xTo(0)} y1={pad} x2={xTo(0)} y2={H - pad} stroke="#94a3b8" strokeWidth="2" strokeDasharray="4 4" />
        <line x1={pad} y1={yTo(0)} x2={W - pad} y2={yTo(0)} stroke="#94a3b8" strokeWidth="2" strokeDasharray="4 4" />

        <text x={W / 2} y={H - 12} fontSize="11" fill="#6b7280" textAnchor="middle">
          Economic (Left ← 0 → Right)
        </text>
        <text x={18} y={H / 2} fontSize="11" fill="#6b7280" textAnchor="middle" transform={`rotate(-90, 18, ${H / 2})`}>
          Social (Authoritarian ↓ 0 ↑ Libertarian)
        </text>

        <text x={pad + 6} y={pad + 14} fontSize="10" fill="#6b7280">
          Lib-Left
        </text>
        <text x={W - pad - 6} y={pad + 14} fontSize="10" fill="#6b7280" textAnchor="end">
          Lib-Right
        </text>
        <text x={pad + 6} y={H - pad - 8} fontSize="10" fill="#6b7280">
          Auth-Left
        </text>
        <text x={W - pad - 6} y={H - pad - 8} fontSize="10" fill="#6b7280" textAnchor="end">
          Auth-Right
        </text>

        {items.map((it, idx) => {
          const r = clamp(5 + (it.evidence ? Math.sqrt(it.evidence) / 4 : 0), 5, 14);
          const opacity = clamp(0.55 + it.conf * 0.45, 0.55, 1);

          const fill = it.y > 0 ? (it.x > 0 ? "#10b981" : "#3b82f6") : it.x > 0 ? "#f59e0b" : "#f43f5e";

          return (
            <g key={idx}>
              <circle cx={xTo(it.x)} cy={yTo(it.y)} r={r} fill={fill} opacity={opacity} stroke="white" strokeWidth="2" className="transition-all duration-200 hover:opacity-100 cursor-pointer" />
              <title>
                {it.title}
                {"\n"}Speaker: {it.speaker}
                {"\n"}Quadrant: {labelFor(it.x, it.y)}
                {"\n"}X (econ): {it.x.toFixed(2)}
                {"\n"}Y (social): {it.y.toFixed(2)}
                {"\n"}Avg Confidence: {(it.conf * 100).toFixed(0)}%
                {"\n"}Evidence Units: {it.evidence || "N/A"}
              </title>
            </g>
          );
        })}
      </svg>

      <div className="mt-5 p-4 bg-gray-50 rounded-xl">
        <div className="flex flex-wrap items-center gap-4 text-sm">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full" style={{ background: "#10b981" }} />
            <span className="text-gray-700">Lib-Right</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full" style={{ background: "#3b82f6" }} />
            <span className="text-gray-700">Lib-Left</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full" style={{ background: "#f59e0b" }} />
            <span className="text-gray-700">Auth-Right</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full" style={{ background: "#f43f5e" }} />
            <span className="text-gray-700">Auth-Left</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-gray-700">Point size ≈ evidence units</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-gray-700">Opacity ≈ avg confidence</span>
          </div>
        </div>
      </div>
    </div>
  );
};

const SimilarityMatrix = ({ items }) => {
  const n = items.length;
  if (n < 2) return null;

  const sims = [];
  for (let i = 0; i < n; i++) {
    sims[i] = [];
    for (let j = 0; j < n; j++) sims[i][j] = cosineSim(items[i].vec, items[j].vec);
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
            Evidence Similarity Matrix
          </h3>
          <p className="text-sm text-gray-500 mt-1">Cosine similarity over 4D evidence-share vectors</p>
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
                  <div className="text-sm font-medium text-gray-900 truncate max-w-[170px]">{row.short}</div>
                  <div className="text-xs text-gray-500">{row.speaker}</div>
                </td>
                {items.map((col, j) => {
                  const value = sims[i][j];
                  const isDiagonal = i === j;
                  return (
                    <td key={`${String(row.id)}-${String(col.id)}`} className="border-t border-gray-100">
                      <div
                        className={`flex items-center justify-center p-3 ${isDiagonal ? "rounded-full mx-auto w-10 h-10" : "rounded-lg"} ${cellClass(value)} transition-all hover:scale-105`}
                        title={`${row.title} vs ${col.title}\nSimilarity: ${value.toFixed(3)}`}
                      >
                        <span className={`text-sm font-bold ${isDiagonal ? "text-gray-400" : ""}`}>{isDiagonal ? "—" : value.toFixed(2)}</span>
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

const TrendChart = ({ series, speaker }) => {
  const W = 640;
  const H = 300;
  const pad = 45;

  const maxX = Math.max(series.length - 1, 1);
  const yMin = 0;
  const yMax = 100;

  const xTo = (i) => pad + (i / maxX) * (W - pad * 2);
  const yTo = (v) => pad + ((yMax - v) / (yMax - yMin)) * (H - pad * 2);

  const pathFor = (key) => series.map((s, i) => `${i === 0 ? "M" : "L"} ${xTo(i)} ${yTo(s[key])}`).join(" ");

  const firstDate = series[0]?.date?.toLocaleDateString() || "";
  const lastDate = series[series.length - 1]?.date?.toLocaleDateString() || "";

  return (
    <div className="bg-white rounded-2xl border border-gray-200 p-6">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h3 className="font-semibold text-gray-900 flex items-center gap-2">
            <Activity className="w-5 h-5 text-indigo-600" />
            Evidence Distribution Over Time {speaker && <span className="font-normal text-gray-600">• {speaker}</span>}
          </h3>
          <p className="text-sm text-gray-500 mt-1">Trends reflect % share of labeled evidence units (4-family system)</p>
        </div>
      </div>

      <svg width={W} height={H} className="rounded-lg border border-gray-100">
        {[0, 25, 50, 75, 100].map((v) => (
          <g key={v}>
            <line x1={pad} y1={yTo(v)} x2={W - pad} y2={yTo(v)} stroke="#f1f5f9" strokeWidth="1" />
            <text x={pad - 8} y={yTo(v) + 4} fontSize="10" fill="#94a3b8" textAnchor="end">
              {v}%
            </text>
          </g>
        ))}

        <path d={pathFor("lib")} fill="none" stroke="#10b981" strokeWidth="3" strokeLinecap="round" />
        <path d={pathFor("auth")} fill="none" stroke="#f43f5e" strokeWidth="3" strokeLinecap="round" />
        <path d={pathFor("eLeft")} fill="none" stroke="#3b82f6" strokeWidth="3" strokeLinecap="round" />
        <path d={pathFor("eRight")} fill="none" stroke="#f59e0b" strokeWidth="3" strokeLinecap="round" />

        {series.map((s, i) => (
          <g key={i}>
            <circle cx={xTo(i)} cy={yTo(s.lib)} r="4" fill="#10b981" stroke="white" strokeWidth="2" />
            <circle cx={xTo(i)} cy={yTo(s.auth)} r="4" fill="#f43f5e" stroke="white" strokeWidth="2" />
            <circle cx={xTo(i)} cy={yTo(s.eLeft)} r="4" fill="#3b82f6" stroke="white" strokeWidth="2" />
            <circle cx={xTo(i)} cy={yTo(s.eRight)} r="4" fill="#f59e0b" stroke="white" strokeWidth="2" />

            {i === 0 && (
              <text x={xTo(i)} y={H - 14} fontSize="10" fill="#6b7280" textAnchor="start">
                {firstDate}
              </text>
            )}
            {i === series.length - 1 && (
              <text x={xTo(i)} y={H - 14} fontSize="10" fill="#6b7280" textAnchor="end">
                {lastDate}
              </text>
            )}
          </g>
        ))}
      </svg>

      <div className="flex flex-wrap items-center gap-4 mt-4 text-sm">
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-emerald-500" />
          <span className="text-gray-700">Libertarian</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-rose-500" />
          <span className="text-gray-700">Authoritarian</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-blue-500" />
          <span className="text-gray-700">Economic-Left</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-amber-500" />
          <span className="text-gray-700">Economic-Right</span>
        </div>
      </div>
    </div>
  );
};

const ComparisonPage = ({ projects, refreshProjects, loading: loadingProp }) => {
  const externalProvided = projects !== undefined;

  const [remoteSpeeches, setRemoteSpeeches] = useState([]);
  const [remoteLoading, setRemoteLoading] = useState(false);
  const [remoteError, setRemoteError] = useState("");

  const [countsById, setCountsById] = useState({});
  const [analysisLoadingById, setAnalysisLoadingById] = useState({});
  const [analysisErrorById, setAnalysisErrorById] = useState({});

  const mountedRef = useRef(true);
  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  const reqSeqRef = useRef(0);
  const reqTokenByIdRef = useRef({});

  const fetchSpeeches = useCallback(async () => {
    setRemoteError("");
    setRemoteLoading(true);
    try {
      const speeches = await api.listSpeechesAll?.({ pageSize: 100 });
      const list = Array.isArray(speeches) ? speeches : await api.listSpeeches();
      setRemoteSpeeches(Array.isArray(list) ? list : []);
    } catch (e) {
      setRemoteError(String(e?.message || "Failed to load speeches"));
      setRemoteSpeeches([]);
    } finally {
      setRemoteLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchSpeeches();
  }, [fetchSpeeches]);

  const mergedSpeeches = useMemo(() => {
    const ext = Array.isArray(projects) ? projects : [];
    const rem = Array.isArray(remoteSpeeches) ? remoteSpeeches : [];
    if (!externalProvided) return rem;
    if (rem.length === 0) return ext;

    const map = new Map();
    for (const r of rem) map.set(getIdKey(r), r);
    for (const e of ext) {
      const k = getIdKey(e);
      if (!map.has(k)) map.set(k, e);
    }
    return Array.from(map.values());
  }, [externalProvided, projects, remoteSpeeches]);

  const loading = Boolean(loadingProp) || remoteLoading;

  const doRefresh = useCallback(() => {
    setCountsById({});
    setAnalysisLoadingById({});
    setAnalysisErrorById({});
    reqTokenByIdRef.current = {};
    if (externalProvided && typeof refreshProjects === "function") refreshProjects();
    fetchSpeeches();
  }, [externalProvided, refreshProjects, fetchSpeeches]);

  const availableSpeeches = useMemo(() => {
    const arr = Array.isArray(mergedSpeeches) ? mergedSpeeches : [];
    return arr.filter((p) => getIdKey(p));
  }, [mergedSpeeches]);

  const MAX_SELECTED = 8;
  const [selectedIds, setSelectedIds] = useState([]);
  const [mode, setMode] = useState("cross");
  const [query, setQuery] = useState("");
  const [speakerFilter, setSpeakerFilter] = useState("All speakers");
  const [expandedView, setExpandedView] = useState(false);

  useEffect(() => {
    if (selectedIds.length !== 0 || availableSpeeches.length === 0) return;

    const sorted = [...availableSpeeches].sort((a, b) => {
      const da = getDateValue(a);
      const db = getDateValue(b);
      if (!da && !db) return 0;
      if (!da) return 1;
      if (!db) return -1;
      return db - da;
    });

    setSelectedIds(sorted.slice(0, 4).map((p) => getIdKey(p)));
  }, [availableSpeeches, selectedIds.length]);

  const speakers = useMemo(() => {
    const set = new Set(availableSpeeches.map(getSpeaker));
    return ["All speakers", ...Array.from(set).sort((a, b) => a.localeCompare(b))];
  }, [availableSpeeches]);

  const filteredList = useMemo(() => {
    const q = (query || "").trim().toLowerCase();
    return availableSpeeches.filter((p) => {
      const sp = getSpeaker(p);
      if (speakerFilter !== "All speakers" && sp !== speakerFilter) return false;
      if (!q) return true;
      const title = getTitle(p).toLowerCase();
      const speaker = sp.toLowerCase();
      return title.includes(q) || speaker.includes(q);
    });
  }, [availableSpeeches, query, speakerFilter]);

  const toggleSelection = (idKey) => {
    setSelectedIds((prev) => {
      if (prev.includes(idKey)) return prev.filter((x) => x !== idKey);
      if (prev.length >= MAX_SELECTED) return prev;
      return [...prev, idKey];
    });
  };

  const selectedRaw = useMemo(() => {
    const sel = new Set(selectedIds);
    return availableSpeeches.filter((p) => sel.has(getIdKey(p)));
  }, [availableSpeeches, selectedIds]);

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

  useEffect(() => {
    const idsToFetch = selected
      .map((p) => getIdKey(p))
      .filter(Boolean)
      .filter((k) => !countsById[k] && !analysisLoadingById[k]);

    if (idsToFetch.length === 0) return;

    const token = ++reqSeqRef.current;

    setAnalysisLoadingById((prev) => {
      const next = { ...prev };
      for (const k of idsToFetch) next[k] = true;
      return next;
    });

    setAnalysisErrorById((prev) => {
      const next = { ...prev };
      for (const k of idsToFetch) next[k] = "";
      return next;
    });

    idsToFetch.forEach((k) => {
      reqTokenByIdRef.current[k] = token;
    });

    const fetchAnalysisForId = async (idKey) => {
      try {
        return await api.getAnalysis(idKey);
      } catch (e) {
        if (typeof api.getAnalysisLegacy === "function") return await api.getAnalysisLegacy(idKey);
        throw e;
      }
    };

    Promise.allSettled(
      idsToFetch.map(async (k) => {
        try {
          const analysis = await fetchAnalysisForId(k);
          const counts = countsFromAnalysis(analysis);
          if (!mountedRef.current) return;
          if (reqTokenByIdRef.current[k] !== token) return;

          setCountsById((prev) => ({ ...prev, [k]: counts }));
          setAnalysisErrorById((prev) => ({ ...prev, [k]: "" }));
        } catch (e) {
          const msg = e?.message || String(e || "Failed to load analysis");
          if (!mountedRef.current) return;
          if (reqTokenByIdRef.current[k] !== token) return;

          setAnalysisErrorById((prev) => ({ ...prev, [k]: msg }));
        } finally {
          if (!mountedRef.current) return;
          if (reqTokenByIdRef.current[k] !== token) return;

          setAnalysisLoadingById((prev) => ({ ...prev, [k]: false }));
        }
      })
    );
  }, [selected, countsById, analysisLoadingById]);

  const derivedBySpeechId = useMemo(() => {
    const out = {};
    for (const p of availableSpeeches) {
      const k = getIdKey(p);
      out[k] =
        countsById[k] || {
          lib: 0,
          auth: 0,
          eLeft: 0,
          eRight: 0,
          total: 0,
          avgConfidence: 0,
          coords: { x: 0, y: 0, source: "none" },
        };
    }
    return out;
  }, [availableSpeeches, countsById]);

  const vizItems = useMemo(() => {
    return selected.map((p) => {
      const k = getIdKey(p);
      const d = derivedBySpeechId[k] || {};
      const dt = getDateValue(p);
      const speaker = getSpeaker(p);
      const title = getTitle(p);

      const total = toInt(d.total ?? 0, 0) || 0;
      const t = total > 0 ? total : 1;

      const libPct = (toInt(d.lib ?? 0, 0) / t) * 100;
      const authPct = (toInt(d.auth ?? 0, 0) / t) * 100;
      const eLeftPct = (toInt(d.eLeft ?? 0, 0) / t) * 100;
      const eRightPct = (toInt(d.eRight ?? 0, 0) / t) * 100;

      const coords = d.coords || { x: 0, y: 0, source: "none" };
      const conf = clamp(toFloat(d.avgConfidence ?? 0, 0), 0, 1);

      return {
        id: k,
        title,
        short: title.length > 16 ? title.slice(0, 16) + "…" : title,
        speaker,
        date: dt,
        conf,
        evidence: total,
        lib: libPct,
        auth: authPct,
        eLeft: eLeftPct,
        eRight: eRightPct,
        x: clamp(toFloat(coords.x ?? 0, 0), -1, 1),
        y: clamp(toFloat(coords.y ?? 0, 0), -1, 1),
        vec: [libPct, authPct, eLeftPct, eRightPct],
      };
    });
  }, [selected, derivedBySpeechId]);

  const trendItems = useMemo(() => {
    if (mode !== "trend") return [];
    if (!vizItems.length) return [];

    if (speakerFilter !== "All speakers") {
      return vizItems.filter((x) => x.speaker === speakerFilter && x.date);
    }

    const counts = new Map();
    for (const it of vizItems) {
      if (!it.date) continue;
      counts.set(it.speaker, (counts.get(it.speaker) || 0) + 1);
    }

    let bestSpeaker = "";
    let bestCount = -1;
    for (const [sp, c] of counts.entries()) {
      if (c > bestCount) {
        bestCount = c;
        bestSpeaker = sp;
      }
    }

    return bestSpeaker ? vizItems.filter((x) => x.speaker === bestSpeaker && x.date) : [];
  }, [mode, vizItems, speakerFilter]);

  const simpleStats = useMemo(() => {
    if (vizItems.length === 0) return null;
    const avg = (arr) => (arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0);
    return { lib: avg(vizItems.map((x) => x.lib)), auth: avg(vizItems.map((x) => x.auth)), eLeft: avg(vizItems.map((x) => x.eLeft)), eRight: avg(vizItems.map((x) => x.eRight)) };
  }, [vizItems]);

  return (
    <div className={`${expandedView ? "fixed inset-0 z-50 bg-white p-6 overflow-auto" : "space-y-6"}`}>
      <div className={`flex items-center justify-between ${expandedView ? "mb-8" : ""}`}>
        <div>
          <div className="flex items-center gap-3 mb-2">
            <div className="p-2 rounded-xl bg-indigo-100">
              <BarChart3 className="w-6 h-6 text-indigo-600" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-gray-900">Evidence Comparison</h1>
              <p className="text-sm text-gray-600 mt-1">Select speeches; this page loads analysis on-demand for accurate counts</p>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-3">
          {expandedView ? (
            <button onClick={() => setExpandedView(false)} className="inline-flex items-center px-4 py-2 rounded-lg border border-gray-300 text-sm text-gray-700 hover:bg-gray-50" type="button">
              <X className="w-4 h-4 mr-2" />
              Exit Fullscreen
            </button>
          ) : (
            <button onClick={() => setExpandedView(true)} className="inline-flex items-center px-4 py-2 rounded-lg border border-gray-300 text-sm text-gray-700 hover:bg-gray-50" type="button">
              <Maximize2 className="w-4 h-4 mr-2" />
              Fullscreen
            </button>
          )}

          <button onClick={doRefresh} className="inline-flex items-center px-4 py-2 rounded-lg bg-indigo-600 text-white font-medium hover:bg-indigo-700" type="button">
            {loading ? <Loader2 className="w-4 h-4 mr-2 animate-spin" /> : <RefreshCw className="w-4 h-4 mr-2" />}
            Refresh
          </button>
        </div>
      </div>

      {remoteError ? (
        <div className="bg-rose-50 border border-rose-200 text-rose-900 rounded-2xl p-4 flex items-start gap-3">
          <AlertTriangle className="w-5 h-5 mt-0.5" />
          <div>
            <div className="font-semibold">Failed to load speeches</div>
            <div className="text-sm mt-1">{remoteError}</div>
          </div>
        </div>
      ) : null}

      <div className="bg-white rounded-2xl border border-gray-200 p-5">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
          <div className="flex flex-wrap items-center gap-3">
            <div className="inline-flex items-center gap-2 text-sm font-semibold text-gray-700">
              <Filter className="w-4 h-4" />
              Mode
            </div>

            <div className="inline-flex rounded-xl border border-gray-300 overflow-hidden bg-gray-100 p-1">
              <button
                onClick={() => setMode("cross")}
                className={`px-4 py-2 text-sm font-semibold rounded-lg transition-all ${mode === "cross" ? "bg-gradient-to-r from-indigo-500 to-indigo-600 text-white shadow-md" : "text-gray-700 hover:bg-white"}`}
                type="button"
              >
                Across Speeches
              </button>
              <button
                onClick={() => setMode("trend")}
                className={`px-4 py-2 text-sm font-semibold rounded-lg transition-all ${mode === "trend" ? "bg-gradient-to-r from-indigo-500 to-indigo-600 text-white shadow-md" : "text-gray-700 hover:bg-white"}`}
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
              <span className="font-semibold">{selectedIds.length}</span> selected • <span className="font-semibold">{filteredList.length}</span> available
            </div>
            <button onClick={() => setSelectedIds([])} className="text-sm px-3 py-1.5 rounded-lg border border-gray-300 hover:bg-gray-50" type="button">
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

      {simpleStats && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
          <StatCard title="Avg Libertarian share" value={`${simpleStats.lib.toFixed(1)}%`} subtitle="Evidence share labeled Libertarian" icon={Shield} tone="emerald" />
          <StatCard title="Avg Authoritarian share" value={`${simpleStats.auth.toFixed(1)}%`} subtitle="Evidence share labeled Authoritarian" icon={Lock} tone="rose" />
          <StatCard title="Avg Economic-Left share" value={`${simpleStats.eLeft.toFixed(1)}%`} subtitle="Evidence share labeled Economic-Left" icon={Scale} tone="blue" />
          <StatCard title="Avg Economic-Right share" value={`${simpleStats.eRight.toFixed(1)}%`} subtitle="Evidence share labeled Economic-Right" icon={TrendingUp} tone="amber" />
          <StatCard title="Selected Speeches" value={selected.length} subtitle={`of ${MAX_SELECTED} maximum`} icon={Users} tone="indigo" />
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-1">
          <div className="bg-white rounded-2xl border border-gray-200 p-5 sticky top-6">
            <div className="flex items-center justify-between mb-6">
              <h2 className="font-semibold text-gray-900 flex items-center gap-2">
                <Users className="w-5 h-5 text-blue-500" />
                Available Speeches
              </h2>
              <div className="text-sm text-gray-500">
                {selectedIds.length}/{MAX_SELECTED}
              </div>
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
              <div className="space-y-3 max-h-[640px] overflow-y-auto pr-2">
                {filteredList.map((speech, index) => {
                  const k = getIdKey(speech);
                  const derived = derivedBySpeechId[k];
                  return (
                    <SpeechCard
                      key={`${k}-${index}`}
                      speech={speech}
                      derived={derived}
                      isSelected={selectedIds.includes(k)}
                      onClick={() => toggleSelection(k)}
                      index={index}
                      isLoadingAnalysis={Boolean(analysisLoadingById[k])}
                      analysisError={analysisErrorById[k]}
                    />
                  );
                })}
              </div>
            )}
          </div>
        </div>

        <div className="lg:col-span-2 space-y-6">
          {selected.length === 0 ? (
            <div className="bg-white rounded-2xl border border-gray-200 p-12 text-center">
              <div className="w-16 h-16 mx-auto mb-6 rounded-2xl bg-gradient-to-br from-indigo-100 to-indigo-200 flex items-center justify-center">
                <GitCompare className="w-8 h-8 text-indigo-600" />
              </div>
              <h3 className="text-lg font-semibold text-gray-900 mb-2">No Speeches Selected</h3>
              <p className="text-gray-600 max-w-md mx-auto mb-6">Select speeches from the left panel.</p>
              <button
                onClick={() => {
                  if (filteredList.length > 0) setSelectedIds(filteredList.slice(0, 4).map((p) => getIdKey(p)));
                }}
                className="inline-flex items-center px-5 py-2.5 rounded-xl bg-gradient-to-r from-indigo-500 to-indigo-600 text-white font-medium hover:shadow-lg"
                type="button"
              >
                Auto-select 4 Speeches
              </button>
            </div>
          ) : mode === "trend" ? (
            trendItems.length < 2 ? (
              <div className="bg-white rounded-2xl border border-gray-200 p-8 text-center">
                <div className="w-16 h-16 mx-auto mb-6 rounded-2xl bg-gradient-to-br from-amber-100 to-amber-200 flex items-center justify-center">
                  <Activity className="w-8 h-8 text-amber-600" />
                </div>
                <h3 className="text-lg font-semibold text-gray-900 mb-2">Need More Data for Trend Analysis</h3>
                <p className="text-gray-600 max-w-md mx-auto">Select at least 2 dated speeches by one speaker.</p>
              </div>
            ) : (
              <TrendChart series={trendItems.map((x) => ({ date: x.date, lib: x.lib, auth: x.auth, eLeft: x.eLeft, eRight: x.eRight }))} speaker={trendItems[0]?.speaker} />
            )
          ) : (
            <>
              <IdeologyScatter2D items={vizItems.map((x) => ({ title: x.title, speaker: x.speaker, x: x.x, y: x.y, conf: x.conf, evidence: x.evidence }))} />
              {vizItems.length >= 2 && <SimilarityMatrix items={vizItems} />}
            </>
          )}
        </div>
      </div>

      {!expandedView && (
        <div className="text-center text-sm text-gray-500 pt-6 border-t border-gray-200">
          <p>Comparison Dashboard • Evidence-only • 4-family 2D system</p>
        </div>
      )}
    </div>
  );
};

export default ComparisonPage;