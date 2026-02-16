import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import {
  AlertCircle,
  ArrowLeft,
  Award,
  Check,
  ChevronDown,
  ChevronUp,
  Code,
  Compass,
  Copy,
  Cpu,
  Database,
  DollarSign,
  Download,
  FileText,
  Filter,
  GitCompare,
  Headphones,
  Info,
  Layers,
  Lock,
  MessageSquare,
  Pause,
  PieChart,
  Play,
  PlayCircle,
  RefreshCw,
  Search,
  Settings,
  Shield,
  SkipBack,
  SkipForward,
  Sparkles,
  Target,
  TrendingUp,
  User,
  Volume2,
  VolumeX,
  X,
  Zap,
} from "lucide-react";

import apiClient, { API_BASE_URL, getAnalysis, getAnalysisLegacy } from "../services/api";

/* ----------------------------- Constants ----------------------------- */

const LIB = "Libertarian";
const AUTH = "Authoritarian";
const ECON_LEFT = "Economic-Left";
const ECON_RIGHT = "Economic-Right";

const AXIS_CENTER_BAND = 0.25;

const COLORS = {
  primary: {
    light: "#00D4AA",
    main: "#0096FF",
    dark: "#0052CC",
    gradient: "linear-gradient(135deg, #00D4AA 0%, #0096FF 50%, #0052CC 100%)",
  },
  success: { light: "#00E5FF", main: "#00B8D9", dark: "#008DA6" },
  warning: { light: "#FFAB00", main: "#FF6B00", dark: "#E65100" },
  error: { light: "#FF5252", main: "#FF1744", dark: "#D50000" },
  neutral: { 500: "#9E9E9E", 700: "#616161" },
  data: {
    libertarian: "#00D4AA",
    authoritarian: "#FF5252",
    left: "#0096FF",
    right: "#FFAB00",
    neutral: "#9E9E9E",
  },
  glass: { border: "rgba(255,255,255,0.10)" },
};

const clamp = (n, min, max) => Math.max(min, Math.min(max, n));
const safeNum = (x, def = 0) => {
  const n = Number(x);
  return Number.isFinite(n) ? n : def;
};

/* ------------------------------ Helpers ------------------------------ */

function unwrapResponse(input) {
  // Handles backend pattern: { success, data, error, message, ... }
  if (input && typeof input === "object" && "success" in input) {
    if (input.success === false) throw new Error(input.error || input.message || "Request failed");
    return "data" in input ? input.data : input;
  }
  return input;
}

function extractBackendMessage(err) {
  const d = err?.response?.data ?? err;
  if (!d) return null;
  if (typeof d === "string") return d.trim() || null;
  if (typeof d === "object") {
    const s =
      (typeof d.detail === "string" && d.detail.trim()) ||
      (typeof d.message === "string" && d.message.trim()) ||
      (typeof d.error === "string" && d.error.trim()) ||
      (typeof d?.error?.message === "string" && d.error.message.trim());
    if (s) return s;
    try {
      return JSON.stringify(d);
    } catch {
      return null;
    }
  }
  return null;
}

function joinBaseUrl(base, pathLike) {
  const baseClean = String(base || "").replace(/\/+$/, "");
  const s = String(pathLike || "").trim();
  if (!s) return null;
  if (/^https?:\/\//i.test(s)) return s;
  const p = s.startsWith("/") ? s : `/${s}`;
  return `${baseClean}${p}`;
}

const toAbsoluteUrl = (urlLike) => joinBaseUrl(API_BASE_URL, urlLike);

const formatClock = (seconds) => {
  if (seconds === null || seconds === undefined || Number.isNaN(seconds)) return "0:00";
  const s = Math.max(0, Number(seconds));
  return `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
};

const formatTimeFull = (seconds) => {
  if (seconds === null || seconds === undefined || Number.isNaN(seconds)) return "0:00";
  const s = Math.max(0, Number(seconds));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = Math.floor(s % 60);
  if (h > 0) return `${h}:${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}`;
  return `${m}:${String(sec).padStart(2, "0")}`;
};

const confidenceTier = (s) => {
  const v = clamp(safeNum(s, 0), 0, 1);
  if (v > 0.8) return "high";
  if (v > 0.6) return "medium";
  if (v > 0.4) return "low";
  return "very-low";
};

const confidenceTierText = (s) =>
  confidenceTier(s)
    .split("-")
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(" ");

const getConfidenceColor = (s) => {
  const t = confidenceTier(s);
  if (t === "high") return COLORS.success.main;
  if (t === "medium") return COLORS.warning.main;
  if (t === "low") return COLORS.error.light;
  return COLORS.error.dark;
};

const axisSplit = (a, b) => {
  const A = Math.max(0, safeNum(a, 0));
  const B = Math.max(0, safeNum(b, 0));
  const total = A + B;
  if (total <= 0) return { total: 0, a: A, b: B, aPct: 0, bPct: 0 };
  return { total, a: A, b: B, aPct: (A / total) * 100, bPct: (B / total) * 100 };
};

const downloadTextFile = (filename, text) => {
  const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
};

const downloadJsonFile = (filename, obj) => {
  const blob = new Blob([JSON.stringify(obj || {}, null, 2)], { type: "application/json;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
};

const getAnalysisRoot = (obj) => {
  if (!obj || typeof obj !== "object") return obj;
  return obj.analysis || obj.result || obj.payload || obj.data || obj;
};

const snippetAroundSpan = (fullText, startChar, endChar, windowChars = 240) => {
  const t = String(fullText || "");
  const sc = Number.isFinite(Number(startChar)) ? Number(startChar) : null;
  const ec = Number.isFinite(Number(endChar)) ? Number(endChar) : null;
  if (!t || sc === null || ec === null || sc < 0 || ec < 0 || ec < sc) return { before: "", after: "" };
  const beforeStart = Math.max(0, sc - windowChars);
  const afterEnd = Math.min(t.length, ec + windowChars);
  return { before: t.slice(beforeStart, sc).trim(), after: t.slice(ec, afterEnd).trim() };
};

const getContextForItem = (item, transcriptText) => {
  const cb = String(item?.context_before || "").trim();
  const ca = String(item?.context_after || "").trim();
  if (cb || ca) return { before: cb, after: ca };
  return snippetAroundSpan(transcriptText, item?.start_char, item?.end_char, 240);
};

const stableItemKey = (it) => {
  const sc = Number.isFinite(Number(it?.start_char)) ? Number(it.start_char) : "";
  const ec = Number.isFinite(Number(it?.end_char)) ? Number(it.end_char) : "";
  const tx = String(it?.text || it?.full_text || "").slice(0, 120);
  return `${sc}::${ec}::${tx}`;
};

const normalizeFamily = (fam) => {
  const raw = String(fam || "").trim();
  if (!raw) return null;

  const s = raw.toLowerCase().replace(/[\s_]+/g, "-");


  if (s.includes("libertarian") || s === "lib") return LIB;
  if (s.includes("authoritarian") || s === "auth") return AUTH;

  if (s.includes("left") && s.includes("econom")) return ECON_LEFT;
  if (s.includes("right") && s.includes("econom")) return ECON_RIGHT;
  if (s === "left") return ECON_LEFT;  
  if (s === "right") return ECON_RIGHT;  

  if (s === "centrist" || s === "neutral" || s.includes("no-signal")) return null;

  return null;
};

const normalizeEvidenceItem = (item) => {
  if (!item || typeof item !== "object") return null;
  const famRaw = item.ideology_family || item.family || item.dominant_family || item.label_family;
  const subRaw = item.ideology_subtype || item.subtype || item.dominant_subtype || item.label_subtype;
  const fam = normalizeFamily(famRaw);
  const signal = clamp(safeNum(item.signal_strength || item.signal, 0), 0, 100);
  const tb = item.time_begin ?? item.start_time ?? item.startTime ?? null;
  const te = item.time_end ?? item.end_time ?? item.endTime ?? null;

  return {
    ...item,
    ideology_family: fam,
    ideology_subtype: fam ? (String(subRaw || "").trim() || null) : null,
    text: String(item.text || ""),
    full_text: String(item.full_text || item.text || ""),
    confidence_score: clamp(safeNum(item.confidence_score || item.confidence, 0), 0, 1),
    signal_strength: signal,
    time_begin: Number.isFinite(Number(tb)) ? Number(tb) : null,
    time_end: Number.isFinite(Number(te)) ? Number(te) : null,
    start_char: Number.isFinite(Number(item.start_char)) ? Number(item.start_char) : null,
    end_char: Number.isFinite(Number(item.end_char)) ? Number(item.end_char) : null,
    evidence_count: safeNum(item.evidence_count, 0),
    is_key_statement: Boolean(item.is_key_statement || item.is_key),
    marpor_codes: Array.isArray(item.marpor_codes) ? item.marpor_codes : [],
    ideology_2d: item.ideology_2d && typeof item.ideology_2d === "object" ? item.ideology_2d : null,
    context_before: item.context_before,
    context_after: item.context_after,
  };
};

const extractKeyStatements = (data) => {
  const root = getAnalysisRoot(data);
  const arr = root?.key_statements || root?.key_segments || root?.highlights || root?.key_evidence;
  return Array.isArray(arr) ? arr.map(normalizeEvidenceItem).filter(Boolean) : [];
};

const extractSegments = (data) => {
  const root = getAnalysisRoot(data);
  for (const src of [
    root?.statements,
    root?.sections,
    root?.segments,
    root?.statement_list,
    root?.evidence_segments,
    root?.sentence_segments,
  ]) {
    if (Array.isArray(src) && src.length) return src.map(normalizeEvidenceItem).filter(Boolean);
  }
  return [];
};

const extractArgumentUnits = (data) => {
  const root = getAnalysisRoot(data);
  const argUnits = root?.argument_units;
  if (!Array.isArray(argUnits)) return [];
  return argUnits
    .filter((u) => u && typeof u === "object")
    .map((u) => ({
      ...u,
      argument_unit_index: safeNum(u.argument_unit_index, 0),
      sentence_range: Array.isArray(u.sentence_range) ? u.sentence_range : [0, 0],
      start_char: Number.isFinite(Number(u.start_char)) ? Number(u.start_char) : null,
      end_char: Number.isFinite(Number(u.end_char)) ? Number(u.end_char) : null,
      text: String(u.text || ""),
      pivot_detected: Boolean(u.pivot_detected),
      pivot_axes: Array.isArray(u.pivot_axes) ? u.pivot_axes : [],
      time_begin: Number.isFinite(Number(u.time_begin)) ? Number(u.time_begin) : null,
      time_end: Number.isFinite(Number(u.time_end)) ? Number(u.time_end) : null,
      spans: Array.isArray(u.spans)
        ? u.spans
            .filter((sp) => sp && typeof sp === "object")
            .map((sp) => ({
              ...sp,
              role: String(sp.role || "transition"),
              text: String(sp.text || ""),
              time_begin: Number.isFinite(Number(sp.time_begin)) ? Number(sp.time_begin) : null,
              time_end: Number.isFinite(Number(sp.time_end)) ? Number(sp.time_end) : null,
            }))
        : [],
      ideology_2d: u.ideology_2d && typeof u.ideology_2d === "object" ? u.ideology_2d : null,
    }));
};

const dedupeEvidence = (arr = []) => {
  const seen = new Set();
  const out = [];
  for (const it of Array.isArray(arr) ? arr : []) {
    if (!it || typeof it !== "object") continue;
    const k = stableItemKey(it);
    if (seen.has(k)) continue;
    seen.add(k);
    out.push(it);
  }
  return out;
};

const computeFamilyCounts = (items = []) => {
  const counts = { [LIB]: 0, [AUTH]: 0, [ECON_LEFT]: 0, [ECON_RIGHT]: 0 };
  for (const it of Array.isArray(items) ? items : []) {
    const fam = normalizeFamily(it?.ideology_family);
    if (fam) counts[fam] += 1;
  }
  return counts;
};

const coordFromMasses = (pos, neg) => {
  const p = safeNum(pos);
  const n = safeNum(neg);
  const t = p + n;
  return t <= 0 ? 0 : (p - n) / t;
};

const buildQuadrantFromCoords = (social, econ) => {
  const socialDir = Math.abs(social) >= AXIS_CENTER_BAND ? (social > 0 ? LIB : AUTH) : "";
  const econDir = Math.abs(econ) >= AXIS_CENTER_BAND ? (econ > 0 ? ECON_RIGHT : ECON_LEFT) : "";
  let name = "";
  if (socialDir && econDir) name = `${socialDir}-${econDir}`;
  else if (socialDir) name = socialDir;
  else if (econDir) name = econDir;

  const mag = Math.sqrt(social ** 2 + econ ** 2);
  let intensity = "Minimal";
  if (mag >= 0.7) intensity = "Strong";
  else if (mag >= 0.5) intensity = "Moderate";
  else if (mag >= 0.3) intensity = "Weak";

  return { name, intensity, magnitude: mag, socialDirection: socialDir, economicDirection: econDir };
};

const normalizeIdeology2DFromBackend = (block) => {
  // Backend canonical: ideology_2d.axis_strengths, coordinates, confidence_2d, quadrant_2d
  if (!block || typeof block !== "object") return null;

  const axis = block.axis_strengths || {};
  const soc = axis.social || {};
  const eco = axis.economic || {};

  const sLib = Math.max(0, safeNum(soc.libertarian, 0));
  const sAuth = Math.max(0, safeNum(soc.authoritarian, 0));
  const eLeft = Math.max(0, safeNum(eco.left, 0));
  const eRight = Math.max(0, safeNum(eco.right, 0));

  const socialTotal = safeNum(soc.total, sLib + sAuth);
  const econTotal = safeNum(eco.total, eLeft + eRight);

  const coords = block.coordinates || {};
  const coordsXY = block.coordinates_xy || {};
  let socialCoord = clamp(safeNum(coords.social ?? coordsXY.y, NaN), -1, 1);
  let econCoord = clamp(safeNum(coords.economic ?? coordsXY.x, NaN), -1, 1);

  if (!Number.isFinite(socialCoord) && socialTotal > 0) socialCoord = coordFromMasses(sLib, sAuth);
  if (!Number.isFinite(econCoord) && econTotal > 0) econCoord = coordFromMasses(eRight, eLeft);
  if (!Number.isFinite(socialCoord)) socialCoord = 0;
  if (!Number.isFinite(econCoord)) econCoord = 0;

  const conf2d = block.confidence_2d || block.confidence || {};
  const socialConf = clamp(safeNum(conf2d.social, 0), 0, 1);
  const econConf = clamp(safeNum(conf2d.economic, 0), 0, 1);
  const overallConf = clamp(safeNum(conf2d.overall, (socialConf + econConf) / 2), 0, 1);

  // "No signal" if everything is zero-ish
  if (socialTotal <= 0 && econTotal <= 0 && socialCoord === 0 && econCoord === 0) return null;

  const quadrantBackend = block.quadrant_2d || block.quadrant || null;
  const quadrant =
    quadrantBackend && typeof quadrantBackend === "object"
      ? {
          name:
            String(quadrantBackend.name || quadrantBackend?.axis_directions ? "" : quadrantBackend.name || "").trim() ||
            "",
          intensity: quadrantBackend.intensity,
          magnitude: safeNum(quadrantBackend.magnitude, Math.sqrt(socialCoord ** 2 + econCoord ** 2)),
        }
      : null;

  const computedQuadrant = buildQuadrantFromCoords(socialCoord, econCoord);

  return {
    axis_strengths: {
      social: { libertarian: sLib, authoritarian: sAuth, total: socialTotal },
      economic: { left: eLeft, right: eRight, total: econTotal },
    },
    coordinates: { social: socialCoord, economic: econCoord },
    confidence_2d: { social: socialConf, economic: econConf, overall: overallConf },
    quadrant: {
      ...computedQuadrant,
      ...(quadrant ? { magnitude: safeNum(quadrant.magnitude, computedQuadrant.magnitude) } : {}),
    },
  };
};

const buildIdeology2D = (segmentCounts, backend2d = null) => {
  const normalizedBackend = normalizeIdeology2DFromBackend(backend2d);
  if (normalizedBackend) return normalizedBackend;

  const lib = safeNum(segmentCounts?.[LIB], 0);
  const auth = safeNum(segmentCounts?.[AUTH], 0);
  const left = safeNum(segmentCounts?.[ECON_LEFT], 0);
  const right = safeNum(segmentCounts?.[ECON_RIGHT], 0);

  const socialTotal = lib + auth;
  const econTotal = left + right;
  if (socialTotal <= 0 && econTotal <= 0) return null;

  const socialCoord = socialTotal > 0 ? coordFromMasses(lib, auth) : 0;
  const econCoord = econTotal > 0 ? coordFromMasses(right, left) : 0;

  const socialConf = socialTotal > 0 ? Math.abs(socialCoord) : 0;
  const econConf = econTotal > 0 ? Math.abs(econCoord) : 0;
  const overall = clamp((socialConf + econConf) / 2, 0, 1);

  return {
    axis_strengths: {
      social: { libertarian: lib, authoritarian: auth, total: socialTotal },
      economic: { left, right, total: econTotal },
    },
    coordinates: { social: socialCoord, economic: econCoord },
    confidence_2d: { social: socialConf, economic: econConf, overall },
    quadrant: buildQuadrantFromCoords(socialCoord, econCoord),
  };
};

// FIXED: This function now properly trusts backend counts when available
const normalizeAnalysisSummary = (summaryObj, { segments = [], keyStatements = [], argumentUnits = [] } = {}) => {
  const s = (summaryObj && typeof summaryObj === "object" ? summaryObj : {}) || {};
  const countsFromSegments = computeFamilyCounts(segments);

  const countsRaw = s.evidence_counts || s.statement_counts_by_family_ideology || {};
  const countsFromBackend = {
    [LIB]: safeNum(countsRaw?.[LIB], 0),
    [AUTH]: safeNum(countsRaw?.[AUTH], 0),
    [ECON_LEFT]: safeNum(countsRaw?.[ECON_LEFT], 0),
    [ECON_RIGHT]: safeNum(countsRaw?.[ECON_RIGHT], 0),
  };

  // ✅ Trust backend if it has any counts, otherwise compute from segments
  const backendTotal = Object.values(countsFromBackend).reduce((a, b) => a + safeNum(b, 0), 0);
  const evidence_counts = backendTotal > 0 ? countsFromBackend : countsFromSegments;

  const total_evidence = Object.values(evidence_counts).reduce((a, b) => a + safeNum(b, 0), 0);

  return {
    ...s,
    evidence_counts,
    total_evidence: Math.max(0, safeNum(s.total_evidence, total_evidence)),
    avg_confidence_score: clamp(safeNum(s.avg_confidence_score, 0), 0, 1),
    avg_signal_strength: clamp(safeNum(s.avg_signal_strength, 0), 0, 100),
    key_statement_count: safeNum(s.key_statement_count, Array.isArray(keyStatements) ? keyStatements.length : 0),
    argument_unit_count: safeNum(s.argument_unit_count, Array.isArray(argumentUnits) ? argumentUnits.length : 0),
    pivot_unit_count: safeNum(
      s.pivot_unit_count,
      Array.isArray(argumentUnits) ? argumentUnits.filter((u) => Boolean(u?.pivot_detected)).length : 0
    ),
    marpor_codes: Array.isArray(s.marpor_codes) ? s.marpor_codes : [],
  };
};

const extractAnalysisSummary = (analysisRoot, ctx = {}) => {
  const root = getAnalysisRoot(analysisRoot);
  if (!root) return null;
  if (root.analysis_summary && typeof root.analysis_summary === "object") {
    return normalizeAnalysisSummary(root.analysis_summary, ctx);
  }

  // fallback: compute from evidence items
  const segments = Array.isArray(ctx.segments) ? ctx.segments : [];
  const keyStatements = Array.isArray(ctx.keyStatements) ? ctx.keyStatements : [];
  const argumentUnits = Array.isArray(ctx.argumentUnits) ? ctx.argumentUnits : [];

  const counts = computeFamilyCounts(segments);
  const total_evidence = Object.values(counts).reduce((a, b) => a + safeNum(b, 0), 0);

  const all = dedupeEvidence(segments);
  let wConfNum = 0;
  let wSigNum = 0;
  let wDen = 0;
  const marpor = new Set();

  for (const it of all) {
    const w = 1;
    wConfNum += clamp(safeNum(it.confidence_score, 0), 0, 1) * w;
    wSigNum += clamp(safeNum(it.signal_strength, 0), 0, 100) * w;
    wDen += w;

    for (const c of Array.isArray(it.marpor_codes) ? it.marpor_codes : []) {
      const cs = String(c || "").trim();
      if (cs) marpor.add(cs);
    }
  }

  return {
    evidence_counts: counts,
    total_evidence,
    avg_confidence_score: wDen > 0 ? wConfNum / wDen : 0,
    avg_signal_strength: wDen > 0 ? wSigNum / wDen : 0,
    key_statement_count: keyStatements.length,
    argument_unit_count: argumentUnits.length,
    pivot_unit_count: argumentUnits.filter((u) => Boolean(u?.pivot_detected)).length,
    marpor_codes: Array.from(marpor),
  };
};

const formatQuadrantLabel = (name) => String(name || "").replaceAll("-", " ").trim();

const getFamilyConfig = (family) => {
  const fam = normalizeFamily(family);
  const configs = {
    [LIB]: {
      color: COLORS.data.libertarian,
      lightColor: "rgba(0,212,170,0.10)",
      gradient: "linear-gradient(135deg,#00D4AA,#00B8A9)",
      icon: Shield,
      name: "Libertarian",
      short: "LIB",
      description: "Individual freedom",
    },
    [AUTH]: {
      color: COLORS.data.authoritarian,
      lightColor: "rgba(255,82,82,0.10)",
      gradient: "linear-gradient(135deg,#FF5252,#FF1744)",
      icon: Lock,
      name: "Authoritarian",
      short: "AUTH",
      description: "Central authority",
    },
    [ECON_LEFT]: {
      color: COLORS.data.left,
      lightColor: "rgba(0,150,255,0.10)",
      gradient: "linear-gradient(135deg,#0096FF,#0052CC)",
      icon: TrendingUp,
      name: "Economic Left",
      short: "LEFT",
      description: "Equality, welfare",
    },
    [ECON_RIGHT]: {
      color: COLORS.data.right,
      lightColor: "rgba(255,171,0,0.10)",
      gradient: "linear-gradient(135deg,#FFAB00,#FF6B00)",
      icon: DollarSign,
      name: "Economic Right",
      short: "RIGHT",
      description: "Markets, enterprise",
    },
  };
  return (
    configs[fam] || {
      color: COLORS.data.neutral,
      lightColor: "rgba(158,158,158,0.10)",
      gradient: "linear-gradient(135deg,#9E9E9E,#757575)",
      icon: Compass,
      name: "No signal",
      short: "—",
      description: "No ideology label",
    }
  );
};

const getQuadrantConfig = (name) => {
  const configs = {
    "Libertarian-Economic-Right": {
      gradient: "linear-gradient(135deg,#00D4AA,#FFAB00)",
      bg: "linear-gradient(135deg,rgba(0,212,170,0.15),rgba(255,171,0,0.15))",
      borderColor: "rgba(0,212,170,0.35)",
      icon: GitCompare,
    },
    "Libertarian-Economic-Left": {
      gradient: "linear-gradient(135deg,#00D4AA,#0096FF)",
      bg: "linear-gradient(135deg,rgba(0,212,170,0.15),rgba(0,150,255,0.15))",
      borderColor: "rgba(0,212,170,0.35)",
      icon: GitCompare,
    },
    "Authoritarian-Economic-Right": {
      gradient: "linear-gradient(135deg,#FF5252,#FFAB00)",
      bg: "linear-gradient(135deg,rgba(255,82,82,0.15),rgba(255,171,0,0.15))",
      borderColor: "rgba(255,82,82,0.35)",
      icon: GitCompare,
    },
    "Authoritarian-Economic-Left": {
      gradient: "linear-gradient(135deg,rgba(255,82,82,1),rgba(0,150,255,1))",
      bg: "linear-gradient(135deg,rgba(255,82,82,0.15),rgba(0,150,255,0.15))",
      borderColor: "rgba(255,82,82,0.35)",
      icon: GitCompare,
    },
    Libertarian: {
      gradient: "linear-gradient(135deg,#00D4AA,#00B8A9)",
      bg: "linear-gradient(135deg,rgba(0,212,170,0.15),rgba(0,184,169,0.15))",
      borderColor: "rgba(0,212,170,0.35)",
      icon: Shield,
    },
    Authoritarian: {
      gradient: "linear-gradient(135deg,#FF5252,#FF1744)",
      bg: "linear-gradient(135deg,rgba(255,82,82,0.15),rgba(255,23,68,0.15))",
      borderColor: "rgba(255,82,82,0.35)",
      icon: Lock,
    },
    "Economic-Right": {
      gradient: "linear-gradient(135deg,#FFAB00,#FF6B00)",
      bg: "linear-gradient(135deg,rgba(255,171,0,0.15),rgba(255,107,0,0.15))",
      borderColor: "rgba(255,171,0,0.35)",
      icon: DollarSign,
    },
    "Economic-Left": {
      gradient: "linear-gradient(135deg,#0096FF,#0052CC)",
      bg: "linear-gradient(135deg,rgba(0,150,255,0.15),rgba(0,82,204,0.15))",
      borderColor: "rgba(0,150,255,0.35)",
      icon: TrendingUp,
    },
    "": {
      gradient: "linear-gradient(135deg,#9E9E9E,#616161)",
      bg: "linear-gradient(135deg,rgba(158,158,158,0.15),rgba(97,97,97,0.15))",
      borderColor: "rgba(158,158,158,0.35)",
      icon: Compass,
    },
  };
  return configs[name] || configs[""];
};

/* --------------------------- API “safe” calls --------------------------- */

async function getSpeechSafe(speechId) {
  const id = encodeURIComponent(String(speechId));
  const candidates = [`/api/speeches/${id}`, `/api/speeches/${id}/detail`, `/api/speech/${id}`, `/api/speeches?id=${id}`];
  let lastErr = null;

  for (const url of candidates) {
    try {
      const res = await apiClient.get(url);
      return unwrapResponse(res.data);
    } catch (e) {
      lastErr = e;
      const st = e?.response?.status;
      if (st && st !== 404 && st !== 405) break;
    }
  }

  throw new Error(extractBackendMessage(lastErr) || lastErr?.message || "Failed to load speech.");
}

async function getSpeechFullSafe(speechId) {
  const id = encodeURIComponent(String(speechId));
  const candidates = [`/api/speeches/${id}/full`, `/api/speeches/${id}/full_text`, `/api/speeches/${id}/transcript`, `/api/speeches/${id}?full=1`];

  let lastErr = null;
  for (const url of candidates) {
    try {
      const res = await apiClient.get(url);
      return unwrapResponse(res.data);
    } catch (e) {
      lastErr = e;
      const st = e?.response?.status;
      if (st && st !== 404 && st !== 405) break;
    }
  }

  // fallback
  return getSpeechSafe(speechId);
}

async function generateQuestionsSafe(speechId, { question_type = "journalistic", max_questions = 5 } = {}) {
  const id = encodeURIComponent(String(speechId));
  const sid = Number(speechId);
  const payload = { ...(Number.isFinite(sid) ? { speech_id: sid } : {}), question_type, max_questions };

  const candidates = [
    { method: "post", url: `/api/speeches/${id}/questions/generate` },
    { method: "post", url: `/api/speeches/${id}/questions` },
    { method: "post", url: `/api/questions/generate` },
    { method: "post", url: `/api/generate_questions` },
  ];

  let lastErr = null;
  for (const c of candidates) {
    try {
      const res = await apiClient[c.method](c.url, payload);
      return unwrapResponse(res.data);
    } catch (e) {
      lastErr = e;
      const st = e?.response?.status;
      if (st && st !== 404 && st !== 405) break;
    }
  }

  throw new Error(extractBackendMessage(lastErr) || lastErr?.message || "Failed to generate questions.");
}

async function reanalyzeSpeechSafe(speechId, { force = true } = {}) {
  const id = encodeURIComponent(String(speechId));
  const url = force ? `/api/speeches/${id}/analyze?force=true` : `/api/speeches/${id}/analyze`;

  try {
    const res = await apiClient.post(url); // FastAPI endpoint has no body
    return unwrapResponse(res.data);
  } catch (e) {
    const st = e?.response?.status;
    if (st === 404 || st === 405) return null;
    throw new Error(extractBackendMessage(e) || e?.message || "Failed to re-analyze.");
  }
}

/* ----------------------------- UI Primitives ----------------------------- */

const GlassCard = ({ children, className = "", hover = false, onClick }) => {
  const [isHovered, setIsHovered] = useState(false);
  return (
    <div
      className={`relative overflow-hidden rounded-2xl transition-all duration-500 ${className} ${hover && isHovered ? "scale-[1.01]" : ""}`}
      style={{
        background: "linear-gradient(135deg,rgba(255,255,255,0.05) 0%,rgba(255,255,255,0.02) 50%,rgba(0,0,0,0.12) 100%)",
        backdropFilter: "blur(18px)",
        border: `1px solid ${COLORS.glass.border}`,
        boxShadow: "0 10px 35px rgba(0,0,0,0.25)",
      }}
      onMouseEnter={() => hover && setIsHovered(true)}
      onMouseLeave={() => hover && setIsHovered(false)}
      onClick={onClick}
      role={onClick ? "button" : "region"}
      tabIndex={onClick ? 0 : undefined}
      onKeyDown={(e) => {
        if (!onClick) return;
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          onClick();
        }
      }}
    >
      {children}
    </div>
  );
};

const GradientBadge = ({ children, color = COLORS.primary.main }) => (
  <div
    className="px-3 py-1.5 rounded-full text-xs font-extrabold uppercase tracking-wider"
    style={{
      background: `linear-gradient(135deg,${color}40,${color}18)`,
      border: `1px solid ${color}45`,
      color,
    }}
  >
    {children}
  </div>
);

const ProgressBar = ({ value, max = 100, color = COLORS.primary.main, showLabel = true, height = "h-2" }) => {
  const pct = max > 0 ? clamp((safeNum(value, 0) / max) * 100, 0, 100) : 0;
  return (
    <div className="space-y-2">
      <div className={`relative ${height} rounded-full overflow-hidden bg-gradient-to-r from-gray-900/50 to-gray-800/40`}>
        <div
          className="absolute top-0 left-0 h-full rounded-full transition-all duration-700"
          style={{
            width: `${pct}%`,
            background: `linear-gradient(90deg,${color}75,${color})`,
            boxShadow: `0 0 18px ${color}40`,
          }}
        />
      </div>
      {showLabel ? (
        <div className="flex justify-between text-[11px] text-gray-400">
          <span>0%</span>
          <span className="font-bold" style={{ color }}>
            {Math.round(pct)}%
          </span>
          <span>100%</span>
        </div>
      ) : null}
    </div>
  );
};

const StatCard = ({ title, value, subtitle, icon: Icon, color = COLORS.primary.main }) => (
  <GlassCard hover className="p-4">
    <div className="flex items-start justify-between">
      <div className="flex-1">
        <p className="text-xs text-gray-400 mb-1">{title}</p>
        <p className="text-2xl font-extrabold text-white">{value}</p>
        {subtitle ? <p className="text-xs text-gray-500 mt-1">{subtitle}</p> : null}
      </div>
      <div className="p-2.5 rounded-xl" style={{ background: `linear-gradient(135deg,${color}20,${color}10)`, border: `1px solid ${color}35` }}>
        <Icon className="w-5 h-5" style={{ color }} />
      </div>
    </div>
  </GlassCard>
);

const ModalShell = ({ open, title, icon: Icon, onClose, children }) => {
  if (!open) return null;
  return (
    <div className="fixed inset-0 z-[80]">
      <div className="absolute inset-0 bg-black/70 backdrop-blur-sm" onClick={onClose} />
      <div className="absolute inset-0 flex items-center justify-center p-4 overflow-y-auto">
        <GlassCard className="w-full max-w-2xl my-8">
          <div className="p-6 border-b border-gray-700/50 flex items-center justify-between sticky top-0 bg-gray-900/95 backdrop-blur-xl rounded-t-2xl z-10">
            <div className="flex items-center gap-3">
              <div className="p-2.5 rounded-xl border border-cyan-500/25 bg-gradient-to-br from-cyan-500/10 to-blue-500/10">
                {Icon ? <Icon className="w-5 h-5 text-cyan-300" /> : null}
              </div>
              <p className="text-white font-extrabold">{title}</p>
            </div>
            <button onClick={onClose} className="p-2 rounded-xl hover:bg-white/5 text-gray-300 hover:text-white transition" type="button">
              <X className="w-5 h-5" />
            </button>
          </div>
          <div className="p-6 max-h-[70vh] overflow-y-auto">{children}</div>
        </GlassCard>
      </div>
    </div>
  );
};

const LoadingScreen = () => (
  <div className="min-h-screen bg-gradient-to-br from-gray-950 via-gray-900 to-gray-950 flex items-center justify-center p-4">
    <div className="max-w-md w-full text-center">
      <div className="relative w-32 h-32 mx-auto mb-8">
        <div className="absolute inset-0 bg-gradient-to-r from-cyan-500 to-blue-500 rounded-full blur-2xl opacity-20 animate-pulse" />
        <div className="absolute inset-6 border-4 border-transparent rounded-full border-t-cyan-400 border-r-blue-400 animate-spin" />
        <div className="absolute inset-10 bg-gradient-to-br from-cyan-400 to-blue-400 rounded-full flex items-center justify-center">
          <Cpu className="w-12 h-12 text-white animate-pulse" />
        </div>
      </div>
      <h2 className="text-3xl font-extrabold bg-gradient-to-r from-cyan-300 via-blue-300 to-cyan-300 bg-clip-text text-transparent">
        Loading Analysis
      </h2>
      <p className="text-gray-400 mt-2 text-sm">Fetching speech analysis data…</p>
    </div>
  </div>
);

const ErrorScreen = ({ error, onBack }) => (
  <div className="min-h-screen bg-gradient-to-br from-gray-950 via-gray-900 to-gray-950 flex items-center justify-center p-4">
    <GlassCard className="max-w-lg w-full">
      <div className="p-10 text-center">
        <div className="relative w-24 h-24 mx-auto mb-8">
          <div className="absolute inset-0 bg-gradient-to-br from-red-500 to-rose-500 rounded-full blur-xl opacity-20" />
          <div className="relative bg-gradient-to-br from-red-600 to-rose-600 rounded-full w-24 h-24 flex items-center justify-center">
            <AlertCircle className="w-12 h-12 text-white" />
          </div>
        </div>
        <h2 className="text-2xl font-extrabold bg-gradient-to-r from-red-300 to-rose-300 bg-clip-text text-transparent mb-4">
          Analysis Failed
        </h2>
        <p className="text-gray-300 mb-6">We couldn't load analysis for this speech.</p>
        <div className="bg-gray-900/50 rounded-xl p-4 mb-8 max-h-32 overflow-y-auto">
          <p className="text-gray-400 text-sm font-mono break-all whitespace-pre-wrap">{String(error)}</p>
        </div>
        <div className="flex flex-col sm:flex-row gap-4">
          <button
            onClick={onBack}
            className="flex-1 px-6 py-3 bg-gradient-to-r from-gray-800 to-gray-900 hover:from-gray-700 hover:to-gray-800 text-white rounded-xl font-semibold border border-gray-700"
            type="button"
          >
            Return to Dashboard
          </button>
          <button
            onClick={() => window.location.reload()}
            className="flex-1 px-6 py-3 bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white rounded-xl font-semibold"
            type="button"
          >
            Try Again
          </button>
        </div>
      </div>
    </GlassCard>
  </div>
);

/* ----------------------------- UI Pieces ----------------------------- */

const AnalysisHeader = ({ speech, onBack, onExport, onSettings, onReanalyze, isReanalyzing }) => {
  const [isScrolled, setIsScrolled] = useState(false);
  useEffect(() => {
    const h = () => setIsScrolled(window.scrollY > 20);
    window.addEventListener("scroll", h);
    return () => window.removeEventListener("scroll", h);
  }, []);

  return (
    <div
      className={`sticky top-0 z-50 transition-all duration-500 ${
        isScrolled ? "bg-gray-900/95 backdrop-blur-2xl border-b border-gray-800/50 shadow-2xl" : "bg-gradient-to-b from-gray-950 to-gray-950/95"
      }`}
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
        <div className="flex items-center justify-between gap-3">
          <div className="flex items-center gap-4 min-w-0">
            <button
              onClick={onBack}
              className="group flex items-center gap-2 text-gray-300 hover:text-white transition-all px-3 py-2 rounded-xl hover:bg-gray-800/50"
              type="button"
            >
              <ArrowLeft className="w-5 h-5 group-hover:-translate-x-1 transition-transform" />
              <span className="font-semibold text-sm hidden sm:inline">Back</span>
            </button>

            <div className="h-8 w-px bg-gradient-to-b from-gray-700/50 to-transparent hidden sm:block" />

            <div className="flex items-center gap-3 min-w-0">
              <div className="p-2.5 bg-gradient-to-br from-cyan-500/10 to-blue-500/10 rounded-lg border border-cyan-500/20">
                <FileText className="w-5 h-5 text-cyan-400" />
              </div>
              <div className="min-w-0">
                <p className="text-sm text-gray-400">Analysis</p>
                <p className="text-sm font-semibold text-white truncate max-w-[180px] sm:max-w-[280px]">
                  {speech?.title || "Untitled"}
                </p>
              </div>
            </div>

            {speech?.speaker ? (
              <>
                <div className="h-8 w-px bg-gradient-to-b from-gray-700/50 to-transparent hidden md:block" />
                <div className="hidden md:flex items-center gap-2 px-3 py-1.5 bg-gray-800/30 rounded-lg border border-gray-700/50">
                  <User className="w-4 h-4 text-gray-400" />
                  <span className="text-sm text-gray-300">{speech.speaker}</span>
                </div>
              </>
            ) : null}
          </div>

          <div className="flex items-center gap-2">
            {onReanalyze ? (
              <button
                onClick={onReanalyze}
                disabled={isReanalyzing}
                className={`p-2.5 rounded-xl transition-all ${
                  isReanalyzing ? "bg-cyan-600/20 text-cyan-300" : "text-gray-400 hover:text-white hover:bg-gray-800/50"
                }`}
                type="button"
                title="Re-analyze"
              >
                <RefreshCw className={`w-5 h-5 ${isReanalyzing ? "animate-spin" : ""}`} />
              </button>
            ) : null}

            <button
              className="p-2.5 text-gray-400 hover:text-white hover:bg-gray-800/50 rounded-xl transition-all"
              type="button"
              title="Export"
              onClick={onExport}
            >
              <Download className="w-5 h-5" />
            </button>
            <button
              className="p-2.5 text-gray-400 hover:text-white hover:bg-gray-800/50 rounded-xl transition-all"
              type="button"
              title="Settings"
              onClick={onSettings}
            >
              <Settings className="w-5 h-5" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

const AnalysisSummaryStats = ({ summary, keyStatements }) => {
  if (!summary) return null;
  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-4">
      <StatCard title="Total Evidence" value={summary.total_evidence ?? 0} subtitle="Evidence units" icon={Database} color={COLORS.primary.main} />
      <StatCard title="Key Statements" value={summary.key_statement_count ?? (keyStatements?.length || 0)} subtitle="High signal" icon={Sparkles} color={COLORS.warning.main} />
      <StatCard
        title="Avg Confidence"
        value={`${Math.round((summary.avg_confidence_score ?? 0) * 100)}%`}
        subtitle={confidenceTierText(summary.avg_confidence_score)}
        icon={Target}
        color={getConfidenceColor(summary.avg_confidence_score)}
      />
      <StatCard title="Avg Signal" value={`${Math.round(summary.avg_signal_strength ?? 0)}%`} subtitle="Signal strength" icon={Zap} color={COLORS.success.main} />
      <StatCard title="MARPOR Codes" value={summary.marpor_codes?.length ?? 0} subtitle="Unique codes" icon={Code} color={COLORS.warning.light} />
    </div>
  );
};

const PoliticalMap = ({ ideology2d, animationsEnabled = true }) => {
  const coords = ideology2d?.coordinates || { social: 0, economic: 0 };
  const x = clamp(safeNum(coords.economic, 0), -1, 1);
  const y = clamp(safeNum(coords.social, 0), -1, 1);
  const confidence = clamp(safeNum(ideology2d?.confidence_2d?.overall, 0), 0, 1);
  const quadrant = ideology2d?.quadrant || {};

  const leftPct = ((x + 1) / 2) * 100;
  const topPct = ((1 - y) / 2) * 100;

  const intensity = Math.sqrt(x * x + y * y);
  const radius = 16 + intensity * 24;

  const qColor =
    x > 0 && y > 0
      ? COLORS.data.libertarian
      : x < 0 && y > 0
      ? COLORS.data.left
      : x > 0 && y < 0
      ? COLORS.data.right
      : x < 0 && y < 0
      ? COLORS.data.authoritarian
      : COLORS.neutral[500];

  return (
    <GlassCard className="p-6">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h3 className="text-lg font-extrabold text-white flex items-center gap-2">
            <Compass className="w-5 h-5 text-cyan-400" />
            Political Map
          </h3>
          <p className="text-sm text-gray-400 mt-1">2D axis visualization</p>
        </div>
        <GradientBadge color={qColor}>{formatQuadrantLabel(quadrant.name) || "No signal"}</GradientBadge>
      </div>

      <div className="relative aspect-square mb-6">
        <div className="absolute inset-0 rounded-xl bg-gradient-to-br from-gray-900/40 to-gray-800/30 border border-gray-700/30 overflow-hidden">
          <div className="absolute left-1/2 top-0 bottom-0 w-px bg-gradient-to-b from-gray-600/50 via-gray-500/30 to-transparent" />
          <div className="absolute top-1/2 left-0 right-0 h-px bg-gradient-to-r from-gray-600/50 via-gray-500/30 to-transparent" />

          <div className="absolute left-1/2 top-2 -translate-x-1/2">
            <div className="flex items-center gap-1 px-2 py-1 rounded-lg" style={{ background: "rgba(0,212,170,0.10)", border: "1px solid rgba(0,212,170,0.22)" }}>
              <Shield className="w-3 h-3 text-teal-300" />
              <span className="text-[10px] font-extrabold text-teal-200">LIB</span>
            </div>
          </div>

          <div className="absolute left-1/2 bottom-2 -translate-x-1/2">
            <div className="flex items-center gap-1 px-2 py-1 rounded-lg" style={{ background: "rgba(255,82,82,0.10)", border: "1px solid rgba(255,82,82,0.22)" }}>
              <Lock className="w-3 h-3 text-red-300" />
              <span className="text-[10px] font-extrabold text-red-200">AUTH</span>
            </div>
          </div>

          <div className="absolute left-2 top-1/2 -translate-y-1/2">
            <div className="flex items-center gap-1 px-2 py-1 rounded-lg" style={{ background: "rgba(0,150,255,0.10)", border: "1px solid rgba(0,150,255,0.22)" }}>
              <TrendingUp className="w-3 h-3 text-blue-300" />
              <span className="text-[10px] font-extrabold text-blue-200">LEFT</span>
            </div>
          </div>

          <div className="absolute right-2 top-1/2 -translate-y-1/2">
            <div className="flex items-center gap-1 px-2 py-1 rounded-lg" style={{ background: "rgba(255,171,0,0.10)", border: "1px solid rgba(255,171,0,0.22)" }}>
              <DollarSign className="w-3 h-3 text-amber-300" />
              <span className="text-[10px] font-extrabold text-amber-200">RIGHT</span>
            </div>
          </div>

          <div className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2">
            <div className="w-4 h-4 rounded-full border-2 border-gray-500/30 bg-gray-900/50 flex items-center justify-center">
              <div className="w-1 h-1 rounded-full bg-gray-400" />
            </div>
          </div>

          <div className="absolute z-10 transition-all duration-1000 ease-out" style={{ left: `${leftPct}%`, top: `${topPct}%`, transform: "translate(-50%, -50%)" }}>
            {animationsEnabled ? (
              <div
                className="absolute inset-0 rounded-full animate-ping"
                style={{
                  background: `radial-gradient(circle, ${qColor}40, transparent 70%)`,
                  animationDuration: "3s",
                  width: `${radius * 2.5}px`,
                  height: `${radius * 2.5}px`,
                  marginLeft: `-${radius * 1.25}px`,
                  marginTop: `-${radius * 1.25}px`,
                }}
              />
            ) : null}

            <div className="relative">
              <div
                className="rounded-full border-2 shadow-2xl flex items-center justify-center"
                style={{
                  width: `${radius}px`,
                  height: `${radius}px`,
                  background: `radial-gradient(circle at 30% 30%, ${qColor}, ${qColor}90)`,
                  borderColor: `rgba(255,255,255, ${0.4 + confidence * 0.4})`,
                  boxShadow: `0 0 ${20 + confidence * 40}px ${qColor}50`,
                }}
              >
                <Target className="w-3 h-3 text-white/90" />
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-3">
        <div className="bg-gray-900/30 rounded-xl p-3 border border-gray-700/30">
          <p className="text-[10px] text-gray-500">Economic (X)</p>
          <p className="text-lg font-extrabold text-white">{x.toFixed(2)}</p>
        </div>
        <div className="bg-gray-900/30 rounded-xl p-3 border border-gray-700/30">
          <p className="text-[10px] text-gray-500">Social (Y)</p>
          <p className="text-lg font-extrabold text-white">{y.toFixed(2)}</p>
        </div>
        <div className="bg-gray-900/30 rounded-xl p-3 border border-gray-700/30">
          <p className="text-[10px] text-gray-500">Intensity</p>
          <p className="text-lg font-extrabold" style={{ color: qColor }}>
            {(intensity * 100).toFixed(0)}%
          </p>
        </div>
      </div>
    </GlassCard>
  );
};

const DominantClassificationCard = ({ ideology2d, showMap = true, animationsEnabled = true }) => {
  const quadrant = ideology2d?.quadrant || {};
  const axisStrengths = ideology2d?.axis_strengths || {};
  const confidence = ideology2d?.confidence_2d || {};

  const qConfig = getQuadrantConfig(quadrant.name || "");
  const QIcon = qConfig.icon;

  const social = axisSplit(axisStrengths.social?.authoritarian, axisStrengths.social?.libertarian);
  const economic = axisSplit(axisStrengths.economic?.left, axisStrengths.economic?.right);

  const totalEvidence = social.total + economic.total;
  const libPctOfTotal = totalEvidence > 0 ? (social.b / totalEvidence) * 100 : 0;
  const authPctOfTotal = totalEvidence > 0 ? (social.a / totalEvidence) * 100 : 0;
  const leftPctOfTotal = totalEvidence > 0 ? (economic.a / totalEvidence) * 100 : 0;
  const rightPctOfTotal = totalEvidence > 0 ? (economic.b / totalEvidence) * 100 : 0;

  const socialConf = clamp(safeNum(confidence.social, 0), 0, 1);
  const econConf = clamp(safeNum(confidence.economic, 0), 0, 1);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between flex-wrap gap-4">
        <div className="flex items-center gap-3">
          <div className="relative">
            <div className="p-3 bg-gradient-to-br from-cyan-500/20 to-blue-500/20 rounded-xl border border-cyan-500/30">
              <Award className="w-6 h-6 text-cyan-300" />
            </div>
            <div className="absolute -top-1 -right-1 w-3 h-3 bg-gradient-to-br from-cyan-400 to-blue-400 rounded-full border-2 border-gray-950" />
          </div>
          <div>
            <h1 className="text-2xl font-extrabold text-white">Ideology Classification</h1>
            <p className="text-gray-400">2D positioning & evidence breakdown</p>
          </div>
        </div>
      </div>

      <div className={`grid grid-cols-1 ${showMap ? "lg:grid-cols-3" : "lg:grid-cols-1"} gap-6`}>
        <div className={`${showMap ? "lg:col-span-2 space-y-6" : "space-y-6"}`}>
          <GlassCard className="p-6">
            <div className="flex items-start justify-between mb-6">
              <div className="flex items-center gap-4">
                <div className="p-4 rounded-xl border" style={{ background: qConfig.bg, borderColor: qConfig.borderColor }}>
                  <QIcon className="w-6 h-6 text-white" />
                </div>
                <div>
                  <p className="text-sm text-gray-400">Dominant Position</p>
                  <h2 className="text-2xl font-extrabold text-white mt-1">{formatQuadrantLabel(quadrant.name) || "No signal"}</h2>
                  <div className="flex items-center gap-2 mt-2 flex-wrap">
                    <GradientBadge color={COLORS.primary.main}>{quadrant.intensity || "Minimal"}</GradientBadge>
                    <span className="text-xs text-gray-400">Magnitude: {safeNum(quadrant.magnitude, 0).toFixed(2)}</span>
                    {economic.total <= 0 ? (
                      <span className="text-xs text-amber-300 inline-flex items-center gap-1">
                        <Info className="w-3 h-3" />
                        No economic evidence
                      </span>
                    ) : null}
                    {social.total <= 0 ? (
                      <span className="text-xs text-amber-300 inline-flex items-center gap-1">
                        <Info className="w-3 h-3" />
                        No social evidence
                      </span>
                    ) : null}
                  </div>
                </div>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Shield className="w-4 h-4" style={{ color: COLORS.data.libertarian }} />
                    <span className="text-sm font-semibold text-gray-200">Social Axis</span>
                  </div>
                  <span className="text-xs text-gray-400">{social.total.toFixed(2)} units</span>
                </div>
                <div className="space-y-3">
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-400">Authoritarian</span>
                    <span className="font-extrabold text-white">{authPctOfTotal.toFixed(1)}%</span>
                  </div>
                  <ProgressBar value={authPctOfTotal} color={COLORS.data.authoritarian} showLabel={false} />
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-400">Libertarian</span>
                    <span className="font-extrabold text-white">{libPctOfTotal.toFixed(1)}%</span>
                  </div>
                  <ProgressBar value={libPctOfTotal} color={COLORS.data.libertarian} showLabel={false} />
                </div>
              </div>

              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <DollarSign className="w-4 h-4" style={{ color: COLORS.data.right }} />
                    <span className="text-sm font-semibold text-gray-200">Economic Axis</span>
                  </div>
                  <span className="text-xs text-gray-400">{economic.total.toFixed(2)} units</span>
                </div>
                <div className="space-y-3">
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-400">Left</span>
                    <span className="font-extrabold text-white">{leftPctOfTotal.toFixed(1)}%</span>
                  </div>
                  <ProgressBar value={leftPctOfTotal} color={COLORS.data.left} showLabel={false} />
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-400">Right</span>
                    <span className="font-extrabold text-white">{rightPctOfTotal.toFixed(1)}%</span>
                  </div>
                  <ProgressBar value={rightPctOfTotal} color={COLORS.data.right} showLabel={false} />
                </div>
              </div>
            </div>
          </GlassCard>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <StatCard title="Social Confidence" value={`${Math.round(socialConf * 100)}%`} subtitle={confidenceTierText(socialConf)} icon={Shield} color={getConfidenceColor(socialConf)} />
            <StatCard title="Economic Confidence" value={`${Math.round(econConf * 100)}%`} subtitle={confidenceTierText(econConf)} icon={DollarSign} color={getConfidenceColor(econConf)} />
          </div>
        </div>

        {showMap ? <PoliticalMap ideology2d={ideology2d} animationsEnabled={animationsEnabled} /> : null}
      </div>
    </div>
  );
};

const MarporChips = ({ codes }) => {
  const arr = Array.isArray(codes) ? codes.map((c) => String(c || "").trim()).filter(Boolean) : [];
  if (!arr.length) return null;
  return (
    <div className="flex flex-wrap gap-2 mt-2">
      {arr.slice(0, 10).map((c) => (
        <span
          key={c}
          className="inline-flex items-center gap-1 px-2 py-1 rounded-lg text-[10px] font-extrabold tracking-wide uppercase"
          style={{
            background: "rgba(255,255,255,0.06)",
            border: "1px solid rgba(255,255,255,0.10)",
            color: "rgba(255,255,255,0.80)",
          }}
          title="MARPOR code"
        >
          <Code className="w-3 h-3 text-gray-300" />
          {c}
        </span>
      ))}
      {arr.length > 10 ? <span className="text-[10px] text-gray-500 ml-1">+{arr.length - 10}</span> : null}
    </div>
  );
};

const ContextToggle = ({ open, onToggle }) => (
  <button
    onClick={onToggle}
    className="px-2 py-1 rounded-lg bg-gray-800/50 hover:bg-gray-700/50 border border-gray-700/50 text-xs font-extrabold text-gray-200 inline-flex items-center gap-2 transition"
    type="button"
    title="Show context before/after"
  >
    <Info className="w-4 h-4 text-gray-300" />
    Context
    {open ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
  </button>
);

const EvidenceList = ({
  title,
  icon: TitleIcon,
  items,
  activeFilter,
  onClearFilter,
  onJumpToTime,
  transcriptText,
  searchable = false,
  prefix = "Item",
}) => {
  const [searchQ, setSearchQ] = useState("");
  const [openCtx, setOpenCtx] = useState(() => new Set());

  const filtered = useMemo(() => {
    let arr = Array.isArray(items) ? items : [];
    if (activeFilter) arr = arr.filter((s) => normalizeFamily(s.ideology_family) === activeFilter);
    if (searchable && searchQ.trim()) {
      const q = searchQ.toLowerCase();
      arr = arr.filter((s) => String(s.text || s.full_text || "").toLowerCase().includes(q));
    }
    return arr;
  }, [items, activeFilter, searchable, searchQ]);

  const toggleCtx = (key) => {
    setOpenCtx((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  };

  return (
    <GlassCard className="h-full flex flex-col">
      <div className="p-6 border-b border-gray-700/50">
        <div className="flex items-center justify-between gap-4 flex-wrap">
          <div className="flex items-center gap-3">
            <div className="p-3 bg-gradient-to-br from-cyan-500/10 to-blue-500/10 rounded-xl border border-cyan-500/20">
              <TitleIcon className="w-5 h-5 text-cyan-400" />
            </div>
            <div>
              <h3 className="text-lg font-extrabold text-white">{title}</h3>
              <p className="text-sm text-gray-400 mt-1">{filtered.length} items</p>
            </div>
          </div>

          {activeFilter ? (
            <button
              onClick={onClearFilter}
              className="px-4 py-2 bg-gray-800/50 hover:bg-gray-700/50 text-gray-300 hover:text-white rounded-xl border border-gray-700/50 transition-all flex items-center gap-2"
              type="button"
            >
              <X className="w-4 h-4" />
              Clear Filter
            </button>
          ) : null}
        </div>

        {searchable ? (
          <div className="mt-4 relative">
            <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-500" />
            <input
              type="text"
              placeholder="Search…"
              value={searchQ}
              onChange={(e) => setSearchQ(e.target.value)}
              className="w-full pl-12 pr-10 py-3 bg-gray-900/50 border border-gray-700/50 rounded-xl text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-cyan-500/40 transition-all"
            />
            {searchQ ? (
              <button
                onClick={() => setSearchQ("")}
                className="absolute right-3 top-1/2 -translate-y-1/2 p-2 text-gray-500 hover:text-gray-300"
                type="button"
              >
                <X className="w-5 h-5" />
              </button>
            ) : null}
          </div>
        ) : null}
      </div>

      <div className="p-6 space-y-3 overflow-y-auto max-h-[700px]">
        {filtered.length ? (
          filtered.map((it, idx) => {
            const cfg = getFamilyConfig(it.ideology_family);
            const Icon = cfg.icon;
            const key = stableItemKey(it);
            const ctxOpen = openCtx.has(key);
            const ctx = getContextForItem(it, transcriptText);

            return (
              <div key={key || idx} className="p-4 rounded-xl border border-gray-700/30 bg-gray-900/20 hover:border-gray-600/50 transition">
                <div className="min-w-0">
                  <div className="flex items-center gap-2 flex-wrap mb-2">
                    <div className="px-2 py-1 rounded bg-gray-800/60 border border-gray-700/50 text-xs font-extrabold text-white">
                      {prefix} {idx + 1}
                    </div>

                    <div className="px-2 py-1 rounded flex items-center gap-2" style={{ background: cfg.lightColor, border: `1px solid ${cfg.color}30` }}>
                      <Icon className="w-3 h-3" style={{ color: cfg.color }} />
                      <span className="text-xs font-extrabold" style={{ color: cfg.color }}>
                        {cfg.name}
                      </span>
                    </div>

                    {it.confidence_score > 0 ? (
                      <div className="px-2 py-1 rounded bg-gray-800/50 border border-gray-700/50 text-xs font-extrabold text-gray-200">
                        {Math.round(it.confidence_score * 100)}% conf
                      </div>
                    ) : null}

                    {it.time_begin !== null ? (
                      <button
                        className="px-2 py-1 rounded bg-gray-800/50 border border-gray-700/50 text-xs font-semibold text-gray-200 hover:bg-gray-700/50 transition inline-flex items-center gap-1"
                        type="button"
                        onClick={() => onJumpToTime && onJumpToTime(it.time_begin)}
                      >
                        <PlayCircle className="w-3 h-3" />
                        {formatClock(it.time_begin)}
                      </button>
                    ) : null}
                  </div>

                  <p className="text-sm text-gray-200 leading-relaxed">{it.text || it.full_text}</p>

                  <MarporChips codes={it.marpor_codes} />

                  {ctx.before || ctx.after ? (
                    <div className="mt-3">
                      <ContextToggle open={ctxOpen} onToggle={() => toggleCtx(key)} />
                      {ctxOpen ? (
                        <div className="mt-3 space-y-2">
                          {ctx.before ? (
                            <div className="p-3 rounded-xl bg-gray-900/35 border border-gray-700/30">
                              <p className="text-[11px] font-extrabold text-gray-400 mb-1">Before</p>
                              <p className="text-sm text-gray-300 whitespace-pre-wrap">{ctx.before}</p>
                            </div>
                          ) : null}
                          {ctx.after ? (
                            <div className="p-3 rounded-xl bg-gray-900/35 border border-gray-700/30">
                              <p className="text-[11px] font-extrabold text-gray-400 mb-1">After</p>
                              <p className="text-sm text-gray-300 whitespace-pre-wrap">{ctx.after}</p>
                            </div>
                          ) : null}
                        </div>
                      ) : null}
                    </div>
                  ) : null}
                </div>
              </div>
            );
          })
        ) : (
          <div className="text-center py-10">
            <MessageSquare className="w-12 h-12 text-gray-700 mx-auto mb-3" />
            <p className="text-gray-500">No items.</p>
          </div>
        )}
      </div>
    </GlassCard>
  );
};

const EvidenceDistribution = ({ summary, segments, activeFilter, setActiveFilter, ideology2d }) => {
  const [expandedFamily, setExpandedFamily] = useState(null);
  
  const useAxisStrengths = ideology2d?.axis_strengths && typeof ideology2d.axis_strengths === 'object';
  
  const evidenceCounts = useAxisStrengths 
    ? {
        [LIB]: safeNum(ideology2d.axis_strengths.social?.libertarian, 0),
        [AUTH]: safeNum(ideology2d.axis_strengths.social?.authoritarian, 0),
        [ECON_LEFT]: safeNum(ideology2d.axis_strengths.economic?.left, 0),
        [ECON_RIGHT]: safeNum(ideology2d.axis_strengths.economic?.right, 0),
      }
    : summary?.evidence_counts || {};
    
  const totalEvidence = Object.values(evidenceCounts).reduce((a, b) => a + safeNum(b, 0), 0);
  const families = [LIB, AUTH, ECON_LEFT, ECON_RIGHT];

  const familyData = useMemo(() => {
    return families
      .map((family) => {
        const config = getFamilyConfig(family);
        const count = safeNum(evidenceCounts[family], 0);
        const pct = totalEvidence > 0 ? (count / totalEvidence) * 100 : 0;
        
        // ✅ FIX: Filter samples by axis contribution when using axis_strengths
        let samples = [];
        if (useAxisStrengths) {
          samples = (Array.isArray(segments) ? segments : [])
            .filter((s) => {
              if (!s?.ideology_2d?.axis_strengths) return false;
              
              const axis = s.ideology_2d.axis_strengths;
              switch(family) {
                case LIB:
                  return safeNum(axis.social?.libertarian, 0) > 0;
                case AUTH:
                  return safeNum(axis.social?.authoritarian, 0) > 0;
                case ECON_LEFT:
                  return safeNum(axis.economic?.left, 0) > 0;
                case ECON_RIGHT:
                  return safeNum(axis.economic?.right, 0) > 0;
                default:
                  return false;
              }
            })
            .slice(0, 3);
        } else {
          samples = (Array.isArray(segments) ? segments : [])
            .filter((s) => s?.ideology_family === family)
            .slice(0, 3);
        }
        
        return { family, config, count, pct, samples, isExpanded: expandedFamily === family };
      })
      .sort((a, b) => b.count - a.count);
  }, [segments, evidenceCounts, totalEvidence, expandedFamily, useAxisStrengths]);

  return (
    <GlassCard className="p-6">
      <div className="flex items-center justify-between mb-6 gap-3 flex-wrap">
        <div className="flex items-center gap-3">
          <div className="p-3 bg-gradient-to-br from-cyan-500/10 to-blue-500/10 rounded-xl border border-cyan-500/20">
            <PieChart className="w-5 h-5 text-cyan-400" />
          </div>
          <div>
            <h3 className="text-lg font-extrabold text-white">Evidence Distribution</h3>
            <p className="text-sm text-gray-400 mt-1">
              {useAxisStrengths ? 'Weighted axis contributions' : 'By ideology category'}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <div className="text-right">
            <p className="text-xs text-gray-500">Total</p>
            <p className="text-lg font-extrabold text-white">
              {useAxisStrengths ? totalEvidence.toFixed(2) : Math.round(totalEvidence)}
            </p>
          </div>
          {activeFilter ? (
            <button
              onClick={() => setActiveFilter(null)}
              className="px-4 py-2 bg-gray-800/50 hover:bg-gray-700/50 text-gray-300 hover:text-white rounded-xl border border-gray-700/50 transition-all flex items-center gap-2"
              type="button"
            >
              <X className="w-4 h-4" />
              Clear
            </button>
          ) : null}
        </div>
      </div>

      <div className="space-y-3">
        {familyData.map(({ family, config, count, pct, samples, isExpanded }) => {
          const Icon = config.icon;
          const isActive = activeFilter === family;

          return (
            <div
              key={family}
              className={`rounded-xl border transition-all overflow-hidden ${
                isActive ? "ring-2 ring-cyan-500/30 border-cyan-500/30" : "border-gray-700/30 hover:border-gray-600/50"
              }`}
              style={{ background: "linear-gradient(135deg, rgba(255,255,255,0.02), rgba(0,0,0,0.05))" }}
            >
              <div
                className="p-4 cursor-pointer hover:bg-white/5 transition-colors"
                onClick={() => setExpandedFamily(isExpanded ? null : family)}
                role="button"
                tabIndex={0}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    setExpandedFamily(isExpanded ? null : family);
                  }
                }}
              >
                <div className="flex items-center justify-between gap-3">
                  <div className="flex items-center gap-3 min-w-0">
                    <div className="p-2 rounded-lg border flex-shrink-0" style={{ background: config.lightColor, borderColor: `${config.color}30` }}>
                      <Icon className="w-4 h-4" style={{ color: config.color }} />
                    </div>
                    <div className="min-w-0">
                      <div className="flex items-center gap-2 flex-wrap">
                        <h4 className="font-extrabold" style={{ color: config.color }}>
                          {config.name}
                        </h4>
                        <span className="px-2 py-0.5 rounded text-[10px] font-extrabold" style={{ background: `${config.color}20`, color: config.color }}>
                          {config.short}
                        </span>
                      </div>
                      <p className="text-xs text-gray-400 mt-0.5">{config.description}</p>
                    </div>
                  </div>

                  <div className="flex items-center gap-4 flex-shrink-0">
                    <div className="text-right">
                      <p className="text-xl font-extrabold text-white">
                        {useAxisStrengths ? count.toFixed(2) : Math.round(count)}
                      </p>
                      <p className="text-xs text-gray-400">{pct.toFixed(1)}%</p>
                    </div>
                    <ChevronDown className={`w-4 h-4 text-gray-400 transition-transform ${isExpanded ? "rotate-180" : ""}`} />
                  </div>
                </div>

                <div className="mt-3">
                  <div className="h-2 bg-gray-800/50 rounded-full overflow-hidden">
                    <div className="h-full rounded-full transition-all duration-1000" style={{ width: `${pct}%`, background: config.gradient }} />
                  </div>
                </div>
              </div>

              {isExpanded ? (
                <div className="px-4 pb-4 border-t border-gray-700/30 pt-4">
                  {count > 0 ? (
                    <div className="space-y-3">
                      <p className="text-sm font-semibold text-gray-300">Sample Evidence</p>
                      {samples.length > 0 ? (
                        <div className="space-y-2">
                          {samples.map((item, i) => (
                            <div key={i} className="p-3 bg-gray-800/20 rounded-lg border border-gray-700/20">
                              <p className="text-sm text-gray-300 line-clamp-2 italic">"{item.text || item.full_text}"</p>
                              {item.confidence_score > 0 ? <div className="mt-2 text-xs text-gray-500">{Math.round(item.confidence_score * 100)}% conf</div> : null}
                            </div>
                          ))}
                        </div>
                      ) : (
                        <div className="p-3 bg-amber-900/10 rounded-lg border border-amber-700/30">
                          <p className="text-xs text-amber-200">
                            No segments with {config.name} as dominant label. Score comes from secondary axis contributions in mixed segments.
                          </p>
                        </div>
                      )}

                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          setActiveFilter(isActive ? null : family);
                        }}
                        className="w-full py-2.5 rounded-lg font-semibold text-sm"
                        style={{
                          background: isActive ? `linear-gradient(135deg, ${config.color}20, ${config.color}10)` : "rgba(255,255,255,0.05)",
                          border: `1px solid ${isActive ? `${config.color}40` : "rgba(255,255,255,0.10)"}`,
                          color: isActive ? config.color : "rgba(255,255,255,0.70)",
                        }}
                        type="button"
                      >
                        <div className="flex items-center justify-center gap-2">
                          <Filter className="w-4 h-4" />
                          {isActive ? "Remove filter" : `Filter by ${config.name}`}
                        </div>
                      </button>
                    </div>
                  ) : (
                    <div className="text-center py-4">
                      <p className="text-gray-500">No evidence found</p>
                    </div>
                  )}
                </div>
              ) : null}
            </div>
          );
        })}
      </div>
    </GlassCard>
  );
};



const QuestionGenerator = ({ speechId }) => {
  const [qType, setQType] = useState("journalistic");
  const [numQ, setNumQ] = useState(5);
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState("");
  const [questions, setQuestions] = useState([]);
  const [copiedAll, setCopiedAll] = useState(false);
  const [copiedIdx, setCopiedIdx] = useState(null);

  const generate = async () => {
    setError("");
    setGenerating(true);
    try {
      const payload = await generateQuestionsSafe(speechId, { question_type: qType, max_questions: clamp(Number(numQ) || 5, 1, 8) });
      const root = getAnalysisRoot(payload) || payload || {};
      const qs = root.questions || root.data?.questions || [];
      setQuestions(Array.isArray(qs) ? qs.map((x) => String(x || "").trim()).filter(Boolean) : []);
    } catch (e) {
      setError(String(e?.message || "Failed"));
    } finally {
      setGenerating(false);
    }
  };

  const copyAll = async () => {
    try {
      await navigator.clipboard.writeText(questions.join("\n\n"));
      setCopiedAll(true);
      setTimeout(() => setCopiedAll(false), 1500);
    } catch {}
  };

  const copyOne = async (q, idx) => {
    try {
      await navigator.clipboard.writeText(q);
      setCopiedIdx(idx);
      setTimeout(() => setCopiedIdx(null), 900);
    } catch {}
  };

  return (
    <GlassCard className="h-full flex flex-col">
      <div className="p-6 border-b border-gray-700/50">
        <div className="flex items-center justify-between gap-4 flex-wrap">
          <div className="flex items-center gap-3">
            <div className="relative">
              <div className="p-3 bg-gradient-to-br from-amber-500/10 to-orange-500/10 rounded-xl border border-amber-500/20">
                <Zap className="w-5 h-5 text-amber-300" />
              </div>
              <div className="absolute -top-1 -right-1 w-3 h-3 bg-gradient-to-br from-amber-300 to-orange-300 rounded-full border-2 border-gray-950" />
            </div>
            <div>
              <h3 className="text-lg font-extrabold text-white">Question Generator</h3>
              <p className="text-sm text-gray-400 mt-1">AI-generated interview questions</p>
            </div>
          </div>

          <button
            onClick={generate}
            disabled={generating}
            className={`px-5 py-2.5 rounded-xl font-extrabold transition-all ${
              generating
                ? "opacity-60 cursor-not-allowed bg-gray-800/60 text-gray-200 border border-gray-700/50"
                : "bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white shadow-lg hover:shadow-xl"
            }`}
            type="button"
          >
            {generating ? (
              <span className="inline-flex items-center gap-2">
                <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                Generating
              </span>
            ) : (
              "Generate"
            )}
          </button>
        </div>
      </div>

      <div className="p-6 flex-1 flex flex-col gap-6">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="bg-gray-900/30 rounded-xl p-4 border border-gray-700/30">
            <p className="text-xs font-extrabold text-gray-400 mb-3">Type</p>
            <div className="grid grid-cols-2 gap-2">
              <button
                onClick={() => setQType("journalistic")}
                className={`px-3 py-2.5 rounded-lg text-sm font-extrabold transition-all ${
                  qType === "journalistic"
                    ? "bg-gradient-to-r from-cyan-600/90 to-blue-600/90 text-white shadow"
                    : "bg-gray-800/50 text-gray-400 hover:text-white hover:bg-gray-700/50 border border-gray-700/40"
                }`}
                type="button"
              >
                Journalistic
              </button>
              <button
                onClick={() => setQType("technical")}
                className={`px-3 py-2.5 rounded-lg text-sm font-extrabold transition-all ${
                  qType === "technical"
                    ? "bg-gradient-to-r from-cyan-600/90 to-blue-600/90 text-white shadow"
                    : "bg-gray-800/50 text-gray-400 hover:text-white hover:bg-gray-700/50 border border-gray-700/40"
                }`}
                type="button"
              >
                Technical
              </button>
            </div>
          </div>

          <div className="bg-gray-900/30 rounded-xl p-4 border border-gray-700/30">
            <p className="text-xs font-extrabold text-gray-400 mb-3">Count</p>
            <div className="grid grid-cols-3 gap-2">
              {[3, 5, 8].map((n) => (
                <button
                  key={n}
                  onClick={() => setNumQ(n)}
                  className={`px-3 py-2.5 rounded-lg text-sm font-extrabold transition-all ${
                    numQ === n
                      ? "bg-gradient-to-r from-emerald-600/90 to-teal-600/90 text-white shadow"
                      : "bg-gray-800/50 text-gray-400 hover:text-white hover:bg-gray-700/50 border border-gray-700/40"
                  }`}
                  type="button"
                >
                  {n}
                </button>
              ))}
            </div>
          </div>
        </div>

        {error ? (
          <div className="p-4 bg-gradient-to-r from-red-900/20 to-rose-900/20 rounded-xl border border-red-700/30">
            <div className="flex items-center gap-3">
              <X className="w-5 h-5 text-red-300 flex-shrink-0" />
              <span className="text-sm text-red-200">{error}</span>
            </div>
          </div>
        ) : null}

        <div className="flex-1 flex flex-col">
          <div className="flex items-center justify-between mb-4 gap-3 flex-wrap">
            <h4 className="text-sm font-extrabold text-gray-300">Generated Questions{questions.length ? ` (${questions.length})` : ""}</h4>

            {questions.length ? (
              <button
                onClick={copyAll}
                className="px-3 py-1.5 bg-gray-800/50 hover:bg-gray-700/50 text-gray-300 hover:text-white rounded-lg border border-gray-700/50 text-xs font-extrabold flex items-center gap-2 transition-all"
                type="button"
              >
                {copiedAll ? (
                  <>
                    <Check className="w-4 h-4 text-emerald-300" />
                    <span className="text-emerald-300">Copied!</span>
                  </>
                ) : (
                  <>
                    <Copy className="w-4 h-4" />
                    Copy All
                  </>
                )}
              </button>
            ) : null}
          </div>

          <div className="space-y-3 flex-1 overflow-y-auto pr-2">
            {questions.length ? (
              questions.map((q, i) => (
                <GlassCard key={i} hover className="p-4">
                  <div className="flex items-start gap-3">
                    <span className="px-2.5 py-1.5 bg-gradient-to-br from-cyan-600 to-blue-600 rounded-lg text-xs font-extrabold text-white flex-shrink-0">
                      Q{i + 1}
                    </span>
                    <p className="text-sm text-gray-200 leading-relaxed flex-1">{q}</p>
                    <button onClick={() => copyOne(q, i)} className="p-2 text-gray-400 hover:text-cyan-300 transition-all" type="button">
                      {copiedIdx === i ? <Check className="w-4 h-4 text-emerald-300" /> : <Copy className="w-4 h-4" />}
                    </button>
                  </div>
                </GlassCard>
              ))
            ) : (
              <div className="text-center py-10">
                <div className="w-16 h-16 bg-gradient-to-br from-gray-900/50 to-gray-800/50 rounded-full flex items-center justify-center mx-auto mb-4 border border-gray-700/30">
                  <Zap className="w-8 h-8 text-gray-600" />
                </div>
                <p className="text-gray-500 font-semibold mb-1">No questions yet</p>
                <p className="text-xs text-gray-600">Press "Generate" to create</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </GlassCard>
  );
};

const MediaPlayer = ({
  mediaUrl,
  isVideo,
  keyStatements,
  currentTime,
  duration,
  isPlaying,
  onSetIsPlaying,
  mediaRef,
  onSetDuration,
  onSetCurrentTime,
  onJumpToTime,
  autoplayOnJump,
  animationsEnabled,
}) => {
  const [volume, setVolume] = useState(0.8);
  const [isMuted, setIsMuted] = useState(false);
  const [playbackRate, setPlaybackRate] = useState(1);
  const progressRef = useRef(null);
  const draggingRef = useRef(false);

  const safePlay = useCallback(async () => {
    const el = mediaRef.current;
    if (!el) return;
    try {
      await el.play();
      onSetIsPlaying(true);
    } catch {
      onSetIsPlaying(false);
    }
  }, [mediaRef, onSetIsPlaying]);

  const togglePlay = useCallback(async () => {
    const el = mediaRef.current;
    if (!el) return;
    if (isPlaying) {
      el.pause();
      onSetIsPlaying(false);
    } else {
      await safePlay();
    }
  }, [isPlaying, mediaRef, onSetIsPlaying, safePlay]);

  const handleTimeUpdate = () => {
    if (mediaRef.current) onSetCurrentTime(mediaRef.current.currentTime || 0);
  };

  const handleLoadedMetadata = () => {
    if (mediaRef.current) {
      onSetDuration(Number(mediaRef.current.duration || 0));
      mediaRef.current.playbackRate = playbackRate;
      mediaRef.current.volume = clamp(volume, 0, 1);
      mediaRef.current.muted = isMuted;
    }
  };

  const setTimeByClientX = (clientX) => {
    if (!progressRef.current || !mediaRef.current || !duration) return;
    const rect = progressRef.current.getBoundingClientRect();
    const pos = clamp((clientX - rect.left) / rect.width, 0, 1);
    const t = pos * duration;
    mediaRef.current.currentTime = t;
    onSetCurrentTime(t);
  };

  const onPointerDown = (e) => {
    draggingRef.current = true;
    setTimeByClientX(e.clientX);

    const onMove = (ev) => draggingRef.current && setTimeByClientX(ev.clientX);
    const onUp = () => {
      draggingRef.current = false;
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
    };

    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
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
    const v = clamp(Number(e.target.value), 0, 1);
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
    const r = clamp(Number(rate), 0.25, 3);
    setPlaybackRate(r);
    if (mediaRef.current) mediaRef.current.playbackRate = r;
  };

  const progressPct = duration > 0 ? clamp((currentTime / duration) * 100, 0, 100) : 0;

  const markerFracs = useMemo(() => {
    const ks = Array.isArray(keyStatements) ? keyStatements : [];
    if (!ks.length || !duration) return [];
    const hasTime = ks.some((k) => Number.isFinite(k?.time_begin));
    const raw = hasTime
      ? ks.map((k, i) => ({
          idx: i,
          frac: Number.isFinite(k?.time_begin) && k.time_begin >= 0 ? clamp(k.time_begin / duration, 0, 1) : (i + 1) / (ks.length + 1),
        }))
      : ks.map((_, i) => ({ idx: i, frac: (i + 1) / (ks.length + 1) }));

    return raw
      .sort((a, b) => a.frac - b.frac)
      .filter((m, i, arr) => !i || Math.abs(arr[i - 1].frac - m.frac) > 0.01);
  }, [keyStatements, duration]);

  const jumpToFrac = async (frac) => {
    if (!duration || !onJumpToTime) return;
    await onJumpToTime(clamp(frac * duration, 0, duration), { autoplay: autoplayOnJump });
  };

  return (
    <GlassCard>
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
            <div className="absolute inset-0 flex items-center justify-center bg-gradient-to-br from-cyan-900/10 to-blue-900/10">
              <div className="text-center p-8">
                <div className="w-32 h-32 bg-gradient-to-br from-cyan-500/10 to-blue-500/10 rounded-full flex items-center justify-center mx-auto mb-6 backdrop-blur-xl border border-cyan-500/20">
                  <Headphones className="w-16 h-16 text-cyan-400/50" />
                </div>
                <p className="text-xl font-extrabold text-white mb-2">Audio Track</p>
                <p className="text-gray-400">Time-synced playback</p>
              </div>
            </div>
          </>
        )}

        {!isPlaying ? (
          <button onClick={togglePlay} className="absolute inset-0 flex items-center justify-center bg-black/55 backdrop-blur-sm transition-all group" type="button">
            <div className="p-6 bg-gradient-to-br from-cyan-500/90 to-blue-500/90 rounded-full group-hover:scale-110 transition-transform duration-500 shadow-2xl">
              <Play className="w-12 h-12 text-white ml-1" />
            </div>
          </button>
        ) : null}

        <div className="absolute top-4 right-4">
          <div className="px-4 py-2 bg-black/60 backdrop-blur-xl rounded-lg border border-gray-700/50">
            <span className="text-white font-mono text-sm">
              {formatClock(currentTime)} / {formatClock(duration)}
            </span>
          </div>
        </div>
      </div>

      <div className="p-6 bg-gradient-to-b from-gray-900/80 to-gray-900/95">
        <div className="mb-6">
          <div ref={progressRef} onPointerDown={onPointerDown} className="relative h-2 bg-gray-800/80 rounded-full cursor-pointer hover:h-3 transition-all group">
            <div className="absolute h-full bg-gradient-to-r from-cyan-500 via-blue-500 to-purple-500 rounded-full transition-all" style={{ width: `${progressPct}%` }}>
              <div className="absolute right-0 top-1/2 -translate-y-1/2 w-4 h-4 bg-white rounded-full shadow-lg ring-2 ring-cyan-500/50" />
              {animationsEnabled ? <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/15 to-transparent animate-shimmer" /> : null}
            </div>

            {duration > 0
              ? markerFracs.map((m) => (
                  <button
                    key={`${m.idx}-${m.frac}`}
                    onClick={(e) => {
                      e.stopPropagation();
                      jumpToFrac(m.frac);
                    }}
                    className="absolute top-1/2 -translate-y-1/2 w-3 h-3 rounded-full bg-gradient-to-br from-amber-500 to-amber-600 shadow-lg hover:scale-150 transition-transform hover:ring-2 hover:ring-white/50 cursor-pointer z-10"
                    style={{ left: `calc(${m.frac * 100}% - 6px)` }}
                    title={`Key statement ${m.idx + 1}`}
                    type="button"
                  />
                ))
              : null}
          </div>

          <div className="flex justify-between mt-3 text-sm text-gray-400">
            <span>{formatTimeFull(currentTime)}</span>
            <span>{formatTimeFull(duration)}</span>
          </div>
        </div>

        <div className="flex items-center justify-between flex-col sm:flex-row gap-4">
          <div className="flex items-center gap-3">
            <button onClick={() => skip(-10)} className="p-3 bg-gray-800/50 hover:bg-gray-700/50 rounded-xl border border-gray-700/50 transition-all group" type="button" title="Back 10s">
              <SkipBack className="w-5 h-5 text-gray-300 group-hover:text-white" />
            </button>

            <button onClick={togglePlay} className="p-4 bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 rounded-xl shadow-xl hover:shadow-2xl transition-all transform hover:scale-105" type="button">
              {isPlaying ? <Pause className="w-6 h-6 text-white" /> : <Play className="w-6 h-6 text-white ml-0.5" />}
            </button>

            <button onClick={() => skip(10)} className="p-3 bg-gray-800/50 hover:bg-gray-700/50 rounded-xl border border-gray-700/50 transition-all group" type="button" title="Forward 10s">
              <SkipForward className="w-5 h-5 text-gray-300 group-hover:text-white" />
            </button>

            <div className="flex items-center gap-2 ml-2">
              <button onClick={toggleMute} className="p-2 text-gray-300 hover:text-white transition-colors" type="button">
                {isMuted || volume === 0 ? <VolumeX className="w-5 h-5" /> : <Volume2 className="w-5 h-5" />}
              </button>
              <div className="w-20">
                <input type="range" min="0" max="1" step="0.01" value={volume} onChange={handleVolumeChange} className="w-full accent-cyan-500" />
              </div>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <select value={playbackRate} onChange={(e) => changePlaybackRate(Number(e.target.value))} className="px-3 py-2 bg-gray-800/50 border border-gray-700/50 rounded-lg text-sm text-white hover:border-gray-600/50 transition-colors">
              {[0.5, 0.75, 1, 1.25, 1.5, 2].map((r) => (
                <option key={r} value={r}>
                  {r}x
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>
    </GlassCard>
  );
};

const Transcript = ({ speech, transcriptText }) => {
  const [expanded, setExpanded] = useState(false);
  const [copied, setCopied] = useState(false);

  const text = (transcriptText || "").trim();
  const wordCount = useMemo(() => (text ? text.split(/\s+/).filter(Boolean).length : 0), [text]);

  const copyTranscript = async () => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {}
  };

  const downloadTranscriptFile = () => {
    if (!text) return;
    downloadTextFile(`${(speech?.title || "transcript").replace(/[^\w\-]+/g, "_")}.txt`, text);
  };

  return (
    <GlassCard className="h-full flex flex-col">
      <div className="p-6 border-b border-gray-700/50">
        <div className="flex items-center justify-between gap-4 flex-wrap">
          <div className="flex items-center gap-3">
            <div className="p-3 bg-gradient-to-br from-cyan-500/10 to-blue-500/10 rounded-xl border border-cyan-500/20">
              <FileText className="w-5 h-5 text-cyan-400" />
            </div>
            <div>
              <h3 className="text-lg font-extrabold text-white">Transcript</h3>
              <p className="text-sm text-gray-400 mt-1">{text ? `${wordCount.toLocaleString()} words` : "No transcript"}</p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <button onClick={copyTranscript} disabled={!text} className="px-4 py-2.5 bg-gray-800/50 hover:bg-gray-700/50 disabled:opacity-50 text-gray-300 hover:text-white rounded-xl border border-gray-700/50 transition-all flex items-center gap-2" type="button">
              {copied ? (
                <>
                  <Check className="w-4 h-4 text-emerald-300" />
                  <span className="text-emerald-300">Copied</span>
                </>
              ) : (
                <>
                  <Copy className="w-4 h-4" />
                  Copy
                </>
              )}
            </button>

            <button onClick={downloadTranscriptFile} disabled={!text} className="px-4 py-2.5 bg-gradient-to-r from-cyan-600/80 to-blue-600/80 hover:from-cyan-500 hover:to-blue-500 text-white rounded-xl transition-all flex items-center gap-2" type="button">
              <Download className="w-4 h-4" />
              Export
            </button>
          </div>
        </div>
      </div>

      <div className="flex-1 overflow-hidden">
        {!text ? (
          <div className="flex items-center justify-center h-full p-8">
            <div className="text-center">
              <FileText className="w-16 h-16 text-gray-700 mx-auto mb-4" />
              <p className="text-gray-500 font-semibold">No transcript available</p>
            </div>
          </div>
        ) : (
          <div className={`h-full overflow-y-auto p-6 ${expanded ? "" : "max-h-96"}`}>
            <p className="text-gray-300 text-sm leading-relaxed whitespace-pre-wrap">{text}</p>
          </div>
        )}
      </div>

      {text && text.length > 1500 ? (
        <div className="p-4 border-t border-gray-700/50">
          <button onClick={() => setExpanded(!expanded)} className="w-full py-3 bg-gradient-to-r from-gray-800/50 to-gray-900/50 hover:from-gray-700/50 hover:to-gray-800/50 border border-gray-700/50 rounded-xl text-gray-300 hover:text-white transition-all flex items-center justify-center gap-3" type="button">
            {expanded ? (
              <>
                <ChevronUp className="w-5 h-5" />
                Collapse Transcript
              </>
            ) : (
              <>
                <ChevronDown className="w-5 h-5" />
                Show Full Transcript
              </>
            )}
          </button>
        </div>
      ) : null}
    </GlassCard>
  );
};

/* ----------------------------- Page Logic ----------------------------- */

const AnalysisPage = () => {
  const { id } = useParams();
  const navigate = useNavigate();

  const [speech, setSpeech] = useState(null);
  const [segments, setSegments] = useState([]);
  const [keyStatements, setKeyStatements] = useState([]);
  const [argumentUnits, setArgumentUnits] = useState([]);
  const [analysisSummary, setAnalysisSummary] = useState(null);
  const [ideology2d, setIdeology2d] = useState(null);
  const [rawAnalysis, setRawAnalysis] = useState(null);

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const [activeFilter, setActiveFilter] = useState(null);
  const [isReanalyzing, setIsReanalyzing] = useState(false);

  const mediaRef = useRef(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const durationEnrichedRef = useRef(false);

  const [settingsOpen, setSettingsOpen] = useState(false);
  const [exportOpen, setExportOpen] = useState(false);

  const [showMap, setShowMap] = useState(true);
  const [animationsEnabled, setAnimationsEnabled] = useState(true);
  const [autoplayOnJump, setAutoplayOnJump] = useState(true);

  const transcriptText = speech?.text ?? speech?.transcript_text ?? speech?.transcript ?? speech?.full_text ?? "";

  const hasMedia = Boolean(speech?.media_url);
  const mediaUrl = useMemo(() => (hasMedia ? toAbsoluteUrl(speech?.media_url) : null), [hasMedia, speech?.media_url]);

  const isVideo = useMemo(() => {
    const mediaLower = String(speech?.media_url || "").toLowerCase();
    return Boolean(mediaLower.includes(".mp4") || mediaLower.includes(".webm") || mediaLower.includes(".mov") || mediaLower.includes(".mkv"));
  }, [speech?.media_url]);

  const fetchAnalysisWithFallback = useCallback(
    async ({ mediaDurationSeconds } = {}) => {
      const opts =
        Number.isFinite(mediaDurationSeconds) && Number(mediaDurationSeconds) > 0
          ? { media_duration_seconds: Number(mediaDurationSeconds) }
          : {};
      try {
        return await getAnalysis(id, opts);
      } catch {}
      try {
        return await getAnalysisLegacy(id, opts);
      } catch {}
      return null;
    },
    [id]
  );

  const hydrateFromAnalysis = useCallback(
    (analysisDataRaw, speechDataForContext) => {
      const root = getAnalysisRoot(analysisDataRaw);
      const analysisData = getAnalysisRoot(root);

      setRawAnalysis(analysisData);

      const segs = extractSegments(analysisData);
      const ks = extractKeyStatements(analysisData);
      const aus = extractArgumentUnits(analysisData);

      setSegments(segs);
      setKeyStatements(ks);
      setArgumentUnits(aus);

      const summary = extractAnalysisSummary(analysisData, { segments: segs, keyStatements: ks, argumentUnits: aus });
      setAnalysisSummary(summary);

      const segmentCounts = computeFamilyCounts(segs);
      const backend2d = analysisData?.ideology_2d || analysisData?.speech_level?.ideology_2d || analysisData?.metadata?.ideology_2d || null;
      setIdeology2d(buildIdeology2D(segmentCounts, backend2d));

      if (speechDataForContext) setSpeech(speechDataForContext);
    },
    []
  );

  useEffect(() => {
    durationEnrichedRef.current = false;
    setDuration(0);
    setCurrentTime(0);
    setIsPlaying(false);
    setActiveFilter(null);
    setError("");
    if (mediaRef.current) {
      try {
        mediaRef.current.pause();
        mediaRef.current.currentTime = 0;
      } catch {}
    }
  }, [id]);

  useEffect(() => {
    let alive = true;

    const load = async () => {
      setLoading(true);
      setError("");

      try {
        let speechData = null;
        try {
          speechData = await getSpeechFullSafe(id);
        } catch {
          speechData = await getSpeechSafe(id);
        }
        if (!alive) return;
        setSpeech(speechData || null);

        const analysisDataRaw = await fetchAnalysisWithFallback();
        if (!alive) return;
        if (!analysisDataRaw) throw new Error("No analysis returned");

        hydrateFromAnalysis(analysisDataRaw, speechData || null);
      } catch (e) {
        if (!alive) return;
        setError(String(e?.message || e || "Unknown error"));
      } finally {
        if (!alive) return;
        setLoading(false);
      }
    };

    load();
    return () => {
      alive = false;
    };
  }, [id, fetchAnalysisWithFallback, hydrateFromAnalysis]);

  useEffect(() => {
    let alive = true;

    const maybeRefetch = async () => {
      if (!speech || !duration || duration <= 0 || durationEnrichedRef.current) return;

      try {
        const analysisDataRaw = await fetchAnalysisWithFallback({ mediaDurationSeconds: duration });
        if (!alive || !analysisDataRaw) return;
        durationEnrichedRef.current = true;
        hydrateFromAnalysis(analysisDataRaw, speech);
      } catch {}
    };

    maybeRefetch();
    return () => {
      alive = false;
    };
  }, [duration, speech, fetchAnalysisWithFallback, hydrateFromAnalysis]);

  const jumpToTime = useCallback(
    async (t, opts = {}) => {
      const el = mediaRef.current;
      if (!el) return;

      const target = clamp(Number(t) || 0, 0, Math.max(0, duration || 0));
      try {
        el.currentTime = target;
      } catch {}
      setCurrentTime(target);

      const shouldAutoplay = typeof opts.autoplay === "boolean" ? opts.autoplay : autoplayOnJump;
      if (shouldAutoplay) {
        try {
          await el.play();
          setIsPlaying(true);
        } catch {
          setIsPlaying(false);
        }
      }
    },
    [duration, autoplayOnJump]
  );

  const handleReanalyze = useCallback(async () => {
    setIsReanalyzing(true);
    try {
      await reanalyzeSpeechSafe(id, { force: true });
      const analysisDataRaw = await fetchAnalysisWithFallback({ mediaDurationSeconds: duration });
      if (!analysisDataRaw) throw new Error("No analysis returned after re-analyze.");
      hydrateFromAnalysis(analysisDataRaw, speech);
    } catch (e) {
      console.error("Re-analyze failed:", e);
      setError(String(e?.message || e || "Re-analyze failed"));
    } finally {
      setIsReanalyzing(false);
    }
  }, [id, duration, speech, fetchAnalysisWithFallback, hydrateFromAnalysis]);

  const exportTranscript = () => {
    const text = String(transcriptText || "").trim();
    if (!text) return;
    downloadTextFile(`${(speech?.title || "transcript").replace(/[^\w\-]+/g, "_")}.txt`, text);
  };

  const exportAnalysisJson = () => {
    downloadJsonFile(`${(speech?.title || "analysis").replace(/[^\w\-]+/g, "_")}.analysis.json`, {
      speech,
      ideology_2d: ideology2d,
      segments,
      key_statements: keyStatements,
      argument_units: argumentUnits,
      analysis_summary: analysisSummary,
      raw: rawAnalysis,
    });
  };

  const exportEvidenceJson = () => {
    downloadJsonFile(`${(speech?.title || "evidence").replace(/[^\w\-]+/g, "_")}.evidence.json`, {
      segments,
      key_statements: keyStatements,
      argument_units: argumentUnits,
    });
  };

  const onBack = () => navigate("/dashboard");

  const distributionEvidence = useMemo(() => dedupeEvidence(segments || []), [segments]);

  if (loading) return <LoadingScreen />;
  if (error) return <ErrorScreen error={error} onBack={onBack} />;

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-950 via-gray-900 to-gray-950">
      <AnalysisHeader
        speech={speech}
        onBack={onBack}
        onExport={() => setExportOpen(true)}
        onSettings={() => setSettingsOpen(true)}
        onReanalyze={handleReanalyze}
        isReanalyzing={isReanalyzing}
      />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
        <AnalysisSummaryStats summary={analysisSummary} keyStatements={keyStatements} />

        <DominantClassificationCard ideology2d={ideology2d} showMap={showMap} animationsEnabled={animationsEnabled} />

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="space-y-6 h-full">
            {hasMedia && mediaUrl ? (
              <MediaPlayer
                mediaUrl={mediaUrl}
                isVideo={isVideo}
                keyStatements={keyStatements}
                currentTime={currentTime}
                duration={duration}
                isPlaying={isPlaying}
                onSetIsPlaying={setIsPlaying}
                mediaRef={mediaRef}
                onSetDuration={setDuration}
                onSetCurrentTime={setCurrentTime}
                onJumpToTime={jumpToTime}
                autoplayOnJump={autoplayOnJump}
                animationsEnabled={animationsEnabled}
              />
            ) : (
              <GlassCard className="p-8 text-center h-full flex flex-col items-center justify-center">
                <Headphones className="w-14 h-14 text-gray-700 mx-auto mb-4" />
                <h3 className="text-xl font-extrabold text-white mb-2">No Media</h3>
                <p className="text-gray-400">Audio/video not attached</p>
              </GlassCard>
            )}
          </div>

          <Transcript speech={speech} transcriptText={transcriptText} />
        </div>

        {/* ✅ FIX: pass ideology2d so EvidenceDistribution can use axis_strengths */}
        <EvidenceDistribution
          summary={analysisSummary}
          segments={distributionEvidence}
          activeFilter={activeFilter}
          setActiveFilter={setActiveFilter}
          ideology2d={ideology2d}
        />

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <EvidenceList
            title="Key Statements"
            icon={Sparkles}
            items={keyStatements}
            activeFilter={activeFilter}
            onClearFilter={() => setActiveFilter(null)}
            onJumpToTime={jumpToTime}
            transcriptText={transcriptText}
            searchable={false}
            prefix="K"
          />
          <QuestionGenerator speechId={id} />
        </div>

        <div className="col-span-full">
          <EvidenceList
            title="Evidence Segments"
            icon={Layers}
            items={segments}
            activeFilter={activeFilter}
            onClearFilter={() => setActiveFilter(null)}
            onJumpToTime={jumpToTime}
            transcriptText={transcriptText}
            searchable
            prefix="#"
          />
        </div>
      </div>

      <ModalShell open={settingsOpen} title="View Settings" icon={Settings} onClose={() => setSettingsOpen(false)}>
        <div className="space-y-5">
          <div className="bg-gray-900/40 border border-gray-700/40 rounded-xl p-4">
            <p className="text-sm font-extrabold text-white mb-1">Visualization</p>
            <p className="text-xs text-gray-400 mb-4">Toggle UI modules</p>
            <div className="space-y-3">
              <label className="flex items-center justify-between gap-4 p-3 rounded-xl bg-gray-900/30 border border-gray-700/30">
                <div>
                  <p className="text-sm font-semibold text-gray-200">Show 2D Map</p>
                  <p className="text-xs text-gray-500">Political positioning map</p>
                </div>
                <input type="checkbox" checked={showMap} onChange={(e) => setShowMap(e.target.checked)} className="w-5 h-5 accent-cyan-500" />
              </label>

              <label className="flex items-center justify-between gap-4 p-3 rounded-xl bg-gray-900/30 border border-gray-700/30">
                <div>
                  <p className="text-sm font-semibold text-gray-200">Animations</p>
                  <p className="text-xs text-gray-500">Ping, shimmer effects</p>
                </div>
                <input type="checkbox" checked={animationsEnabled} onChange={(e) => setAnimationsEnabled(e.target.checked)} className="w-5 h-5 accent-cyan-500" />
              </label>

              <label className="flex items-center justify-between gap-4 p-3 rounded-xl bg-gray-900/30 border border-gray-700/30">
                <div>
                  <p className="text-sm font-semibold text-gray-200">Autoplay on Jump</p>
                  <p className="text-xs text-gray-500">Auto-play when clicking timestamps</p>
                </div>
                <input type="checkbox" checked={autoplayOnJump} onChange={(e) => setAutoplayOnJump(e.target.checked)} className="w-5 h-5 accent-cyan-500" />
              </label>
            </div>
          </div>

          <div className="flex justify-end">
            <button onClick={() => setSettingsOpen(false)} className="px-5 py-2.5 bg-gray-800/50 hover:bg-gray-700/50 text-gray-200 rounded-xl border border-gray-700/50 transition" type="button">
              Close
            </button>
          </div>
        </div>
      </ModalShell>

      <ModalShell open={exportOpen} title="Export" icon={Download} onClose={() => setExportOpen(false)}>
        <div className="space-y-5">
          <div className="bg-gray-900/40 border border-gray-700/40 rounded-xl p-4">
            <p className="text-sm font-extrabold text-white mb-1">Download Assets</p>
            <p className="text-xs text-gray-400 mb-4">Export transcript + analysis</p>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
              <button
                onClick={exportTranscript}
                disabled={!String(transcriptText || "").trim()}
                className="px-4 py-3 rounded-xl bg-gradient-to-r from-cyan-600/90 to-blue-600/90 hover:from-cyan-500 hover:to-blue-500 disabled:opacity-50 text-white font-extrabold transition flex items-center justify-center gap-2"
                type="button"
              >
                <FileText className="w-4 h-4" />
                Transcript
              </button>

              <button
                onClick={exportAnalysisJson}
                className="px-4 py-3 rounded-xl bg-gray-800/50 hover:bg-gray-700/50 text-gray-200 font-extrabold border border-gray-700/50 transition flex items-center justify-center gap-2"
                type="button"
              >
                <Download className="w-4 h-4" />
                Full Analysis
              </button>

              <button
                onClick={exportEvidenceJson}
                className="px-4 py-3 rounded-xl bg-gray-800/50 hover:bg-gray-700/50 text-gray-200 font-extrabold border border-gray-700/50 transition flex items-center justify-center gap-2"
                type="button"
              >
                <Layers className="w-4 h-4" />
                Evidence Only
              </button>
            </div>
          </div>

          <div className="flex justify-end">
            <button onClick={() => setExportOpen(false)} className="px-5 py-2.5 bg-gray-800/50 hover:bg-gray-700/50 text-gray-200 rounded-xl border border-gray-700/50 transition" type="button">
              Close
            </button>
          </div>
        </div>
      </ModalShell>

      <style>{`
        @keyframes shimmer { 0% { transform: translateX(-100%);} 100% { transform: translateX(200%);} }
        .animate-shimmer { animation: shimmer 2s infinite; }
      `}</style>
    </div>
  );
};

export default AnalysisPage;