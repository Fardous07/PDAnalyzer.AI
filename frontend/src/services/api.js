// frontend/src/services/api.js
import axios from "axios";

export const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || "http://localhost:8000").replace(/\/+$/, "");

export function formatTime(seconds) {
  if (seconds === null || seconds === undefined || Number.isNaN(seconds)) return "00:00";
  const s = Math.max(0, Number(seconds));
  const mins = Math.floor(s / 60);
  const secs = Math.floor(s % 60);
  return `${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;
}

export function formatTimeWithHours(seconds) {
  if (seconds === null || seconds === undefined || Number.isNaN(seconds)) return "00:00";
  const s = Math.max(0, Number(seconds));
  const hours = Math.floor(s / 3600);
  const mins = Math.floor((s % 3600) / 60);
  const secs = Math.floor(s % 60);
  if (hours > 0) return `${hours}:${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;
  return `${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;
}

export function formatTimeRange(startTime, endTime) {
  return `${formatTime(startTime)} - ${formatTime(endTime)}`;
}

export function jumpToTime(seconds) {
  const videoPlayer = document.getElementById("video-player");
  if (videoPlayer) {
    videoPlayer.currentTime = Number(seconds) || 0;
    videoPlayer.play?.();
  }
}

const TOKEN_KEY = "token";
const REFRESH_KEY = "refresh_token";
const USER_KEY = "user";

function stripBearer(t) {
  return typeof t === "string" ? t.replace(/^Bearer\s+/i, "").trim() : null;
}

function base64UrlToJson(b64url) {
  try {
    let b64 = String(b64url).replace(/-/g, "+").replace(/_/g, "/");
    while (b64.length % 4) b64 += "=";
    const binary = atob(b64);
    const bytes = Uint8Array.from(binary, (c) => c.charCodeAt(0));
    const text = new TextDecoder().decode(bytes);
    return JSON.parse(text);
  } catch {
    return null;
  }
}

function decodeJwtPayload(token) {
  const raw = stripBearer(token);
  if (!raw) return null;
  const parts = raw.split(".");
  if (parts.length < 2) return null;
  return base64UrlToJson(parts[1]);
}

function jwtType(token) {
  const p = decodeJwtPayload(token);
  return p?.type || p?.typ || p?.token_type || p?.tokenType || null;
}

export function getAccessToken() {
  const t = stripBearer(localStorage.getItem(TOKEN_KEY));
  if (!t || t === "undefined" || t === "null") return null;
  if (jwtType(t) === "refresh") return null;
  return t;
}

export function getRefreshToken() {
  const t = stripBearer(localStorage.getItem(REFRESH_KEY));
  if (!t || t === "undefined" || t === "null") return null;
  return t;
}

export function setAuthTokens({ accessToken, refreshToken, user } = {}) {
  let a = stripBearer(accessToken);
  let r = stripBearer(refreshToken);

  const aType = a ? jwtType(a) : null;
  const rType = r ? jwtType(r) : null;
  if (a && r && aType === "refresh" && rType === "access") {
    const tmp = a;
    a = r;
    r = tmp;
  }

  if (a && jwtType(a) === "refresh") {
    r = r || a;
    a = null;
  }

  if (a) localStorage.setItem(TOKEN_KEY, a);
  if (r) localStorage.setItem(REFRESH_KEY, r);

  if (user !== undefined) {
    try {
      localStorage.setItem(USER_KEY, JSON.stringify(user));
    } catch {}
  }
}

export function clearAuthTokens() {
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(REFRESH_KEY);
  localStorage.removeItem(USER_KEY);
}

function extractBackendMessage(responseData) {
  if (!responseData) return null;

  if (typeof responseData === "string") {
    const s = responseData.trim();
    return s ? s : null;
  }

  if (typeof responseData === "object") {
    if (typeof responseData.detail === "string" && responseData.detail.trim()) return responseData.detail.trim();
    if (typeof responseData.message === "string" && responseData.message.trim()) return responseData.message.trim();
    if (typeof responseData.error === "string" && responseData.error.trim()) return responseData.error.trim();

    if (responseData.error && typeof responseData.error === "object") {
      if (typeof responseData.error.message === "string" && responseData.error.message.trim()) return responseData.error.message.trim();
      if (typeof responseData.error.code === "string" && responseData.error.code.trim()) return responseData.error.code.trim();
      if (typeof responseData.error.type === "string" && responseData.error.type.trim()) return responseData.error.type.trim();
    }

    if (responseData.data && typeof responseData.data === "object") {
      const nested = extractBackendMessage(responseData.data);
      if (nested) return nested;
    }

    if (responseData.result && typeof responseData.result === "object") {
      const nested = extractBackendMessage(responseData.result);
      if (nested) return nested;
    }

    try {
      return JSON.stringify(responseData);
    } catch {
      return null;
    }
  }

  return null;
}

function looksLikeQuotaOrBillingIssue(msg) {
  const s = String(msg || "").toLowerCase();
  return s.includes("insufficient_quota") || s.includes("exceeded your current quota") || s.includes("check your plan and billing") || s.includes("billing") || s.includes("quota");
}

function looksLikeRateLimit(msg) {
  const s = String(msg || "").toLowerCase();
  return s.includes("rate limit") || s.includes("too many requests");
}

const apiClient = axios.create({ baseURL: API_BASE_URL, timeout: 60000, withCredentials: false });
const uploadClient = axios.create({ baseURL: API_BASE_URL, timeout: 300000, withCredentials: false });

let _refreshPromise = null;

function unwrapResponse(input) {
  if (input && typeof input === "object" && "status" in input && "data" in input) {
    return unwrapResponse(input.data);
  }

  if (input && typeof input === "object" && "success" in input) {
    if (input.success === false) {
      throw new Error(input.error || input.message || input.detail || "Request failed");
    }
    if ("data" in input) return input.data;
    return input;
  }

  return input;
}

async function refreshAccessToken() {
  if (_refreshPromise) return _refreshPromise;

  const refresh_token = getRefreshToken();
  if (!refresh_token) throw new Error("Missing refresh token");

  _refreshPromise = apiClient
    .post("/api/auth/refresh", { refresh_token }, { _skipAuth: true })
    .then((res) => unwrapResponse(res.data))
    .then((data) => {
      const access = data?.access_token || data?.tokens?.access_token;
      const refresh = data?.refresh_token || data?.tokens?.refresh_token || refresh_token;
      if (!access) throw new Error("Refresh did not return access_token");
      setAuthTokens({ accessToken: access, refreshToken: refresh });
      return access;
    })
    .finally(() => {
      _refreshPromise = null;
    });

  return _refreshPromise;
}

[apiClient, uploadClient].forEach((client) => {
  client.interceptors.request.use(
    (config) => {
      if (config?._skipAuth) return config;

      const token = getAccessToken();
      if (token) {
        config.headers = config.headers || {};
        config.headers.Authorization = `Bearer ${token}`;
      }
      return config;
    },
    (error) => Promise.reject(error)
  );

  client.interceptors.response.use(
    (response) => response,
    async (error) => {
      const status = error?.response?.status;
      const original = error?.config;

      const url = String(original?.url || "");
      const isAuthCall = url.includes("/api/auth/login") || url.includes("/api/auth/refresh");

      if (status === 401 && original && !original._retry && !isAuthCall) {
        original._retry = true;
        try {
          const newAccess = await refreshAccessToken();
          original.headers = original.headers || {};
          original.headers.Authorization = `Bearer ${newAccess}`;
          return client.request(original);
        } catch {
          clearAuthTokens();
        }
      }

      return Promise.reject(error);
    }
  );
});

export { apiClient, uploadClient };

function handleAxiosError(error, { logoutOn401 = true } = {}) {
  const status = error?.response?.status;
  const responseData = error?.response?.data;

  const backendMsg = extractBackendMessage(responseData);
  let message = (backendMsg && backendMsg.trim ? backendMsg.trim() : null) || error?.message || "Request failed. Please try again.";

  if (looksLikeQuotaOrBillingIssue(message)) {
    message = "AI provider quota exceeded. Check billing/quota and try again.";
  } else if (looksLikeRateLimit(message)) {
    message = "AI provider rate limited. Try again in a moment.";
  }

  const hasBackendMessage = Boolean(backendMsg && typeof backendMsg === "string" && backendMsg.trim());

  if (status === 401 && logoutOn401) {
    if (!getRefreshToken()) clearAuthTokens();
    if (!hasBackendMessage) message = "Session expired. Please log in again.";
  } else if (status === 402 && !hasBackendMessage) {
    message = "Payment/quota required for the AI provider. Check billing/quota and try again.";
  } else if (status === 403 && !hasBackendMessage) {
    message = "You do not have permission to access this resource.";
  } else if (status === 404 && !hasBackendMessage) {
    message = "Resource not found.";
  } else if (status === 413 && !hasBackendMessage) {
    message = "File too large. Please try a smaller file.";
  } else if (status === 429 && !hasBackendMessage) {
    message = "Too many requests. Please try again later.";
  } else if (status >= 500 && !hasBackendMessage) {
    message = "Server error. Please try again later.";
  }

  if (status === 422 && responseData && typeof responseData === "object" && Array.isArray(responseData.detail)) {
    const first = responseData.detail[0] || null;
    const loc = Array.isArray(first?.loc) ? first.loc.join(".") : "";
    const msg = first?.msg || "";
    message = `Validation error${loc ? ` (${loc})` : ""}${msg ? `: ${msg}` : ""}`;
  }

  if (error?.message?.includes("Network Error") || error?.code === "ERR_NETWORK") {
    message = "Cannot connect to the server. Please check if the backend is running.";
  }

  if (error?.code === "ECONNABORTED") {
    message = "Request timed out. Please try again.";
  }

  throw new Error(message);
}

export async function checkBackendConnection() {
  try {
    const res = await fetch(`${API_BASE_URL}/health`);
    return res.ok ? { ok: true, status: res.status, data: await res.json() } : { ok: false, status: res.status };
  } catch (error) {
    return { ok: false, error: error?.message || String(error) };
  }
}

export const authAPI = {
  login: async (credentials) => {
    try {
      const res = await apiClient.post("/api/auth/login", credentials, { _skipAuth: true });
      const data = res.data;

      if (data?.success && data?.tokens && data?.user) {
        const out = {
          access_token: data.tokens.access_token,
          refresh_token: data.tokens.refresh_token,
          user: data.user,
          token_type: data.tokens.token_type || "bearer",
          expires_in: data.tokens.expires_in,
        };
        setAuthTokens({ accessToken: out.access_token, refreshToken: out.refresh_token, user: out.user });
        return out;
      }

      if (data?.access_token && data?.user) {
        setAuthTokens({ accessToken: data.access_token, refreshToken: data.refresh_token, user: data.user });
        return data;
      }

      if (data?.data?.tokens && data?.data?.user) {
        const out = {
          access_token: data.data.tokens.access_token,
          refresh_token: data.data.tokens.refresh_token,
          user: data.data.user,
        };
        setAuthTokens({ accessToken: out.access_token, refreshToken: out.refresh_token, user: out.user });
        return out;
      }

      throw new Error("Invalid login response format");
    } catch (error) {
      handleAxiosError(error, { logoutOn401: false });
    }
  },

  register: async (userData) => {
    try {
      const res = await apiClient.post("/api/auth/register", userData, { _skipAuth: true });
      const data = res.data;
      if (data?.success) return { success: true, message: data?.message || "Registration successful. Please sign in." };
      if (data?.data?.success) return { success: true, message: data?.data?.message || "Registration successful. Please sign in." };
      throw new Error(data?.error || data?.message || "Registration failed");
    } catch (error) {
      handleAxiosError(error, { logoutOn401: false });
    }
  },

  getCurrentUser: async () => {
    try {
      const res = await apiClient.get("/api/auth/me");
      return unwrapResponse(res.data);
    } catch (error) {
      handleAxiosError(error);
    }
  },

  logout: async () => {
    try {
      const res = await apiClient.post("/api/auth/logout");
      clearAuthTokens();
      return unwrapResponse(res.data);
    } catch (error) {
      clearAuthTokens();
      handleAxiosError(error);
    }
  },

  refreshToken: async () => {
    try {
      const refresh_token = getRefreshToken();
      if (!refresh_token) throw new Error("Missing refresh token. Please log in again.");

      const res = await apiClient.post("/api/auth/refresh", { refresh_token }, { _skipAuth: true });
      const data = unwrapResponse(res.data);

      const access = data?.access_token || data?.tokens?.access_token;
      const refresh = data?.refresh_token || data?.tokens?.refresh_token;

      if (!access) throw new Error("Invalid refresh response: missing access token.");

      setAuthTokens({ accessToken: access, refreshToken: refresh || refresh_token });
      return data;
    } catch (error) {
      handleAxiosError(error, { logoutOn401: true });
    }
  },
};

const MAX_PAGE_SIZE = 100;

function clampInt(v, fallback) {
  const n = Number(v);
  if (!Number.isFinite(n)) return fallback;
  return Math.floor(n);
}

function normalizeListParams(params = {}) {
  const p = { ...(params || {}) };

  if (p.page_size != null) p.page_size = Math.min(MAX_PAGE_SIZE, Math.max(1, clampInt(p.page_size, MAX_PAGE_SIZE)));
  if (p.limit != null) p.limit = Math.min(MAX_PAGE_SIZE, Math.max(1, clampInt(p.limit, MAX_PAGE_SIZE)));

  if (p.page != null) p.page = Math.max(1, clampInt(p.page, 1));

  return p;
}

function extractSpeechArray(payload) {
  if (!payload) return [];
  if (Array.isArray(payload)) return payload;

  if (typeof payload === "object") {
    const cands = [
      payload.speeches,
      payload.items,
      payload.results,
      payload.data?.speeches,
      payload.data?.items,
      payload.data?.results,
    ];
    for (const c of cands) {
      if (Array.isArray(c)) return c;
    }
  }
  return [];
}

export async function listSpeeches(params = {}) {
  const fetchWith = async (p) => {
    const res = await apiClient.get("/api/speeches/", { params: normalizeListParams(p) });
    const payload = unwrapResponse(res.data);
    return extractSpeechArray(payload);
  };

  try {
    return await fetchWith(params);
  } catch (error) {
    if (error?.response?.status === 422) {
      try {
        return await fetchWith({});
      } catch (e2) {
        handleAxiosError(e2);
      }
    }
    handleAxiosError(error);
  }
}

export async function listSpeechesAll({ pageSize = 100, max = 10000 } = {}) {
  const size = Math.min(100, Math.max(1, clampInt(pageSize, 100)));
  const out = [];
  const seen = new Set();

  let page = 1;
  let guard = 0;

  while (out.length < max && guard < 200) {
    guard += 1;

    let batch = [];
    try {
      batch = await listSpeeches({ page, page_size: size });
    } catch {
      batch = [];
    }

    if (!Array.isArray(batch) || batch.length === 0) break;

    let added = 0;
    for (const s of batch) {
      const k = String(s?.id ?? s?.speech_id ?? s?.speechId ?? s?._id ?? "").trim();
      if (!k || seen.has(k)) continue;
      seen.add(k);
      out.push(s);
      added += 1;
      if (out.length >= max) break;
    }

    if (added === 0) break;
    if (batch.length < size) break;

    page += 1;
  }

  return out;
}

export async function getSpeechesStats() {
  try {
    const res = await apiClient.get("/api/speeches/stats");
    return unwrapResponse(res.data);
  } catch (error) {
    handleAxiosError(error);
  }
}

export async function searchSpeeches({ q, search_in = "all", include_public = false, limit = 20 } = {}) {
  const qq = String(q ?? "").trim();
  if (qq.length < 2) return [];

  const safeLimit = Math.min(100, Math.max(1, clampInt(limit, 20)));

  try {
    const res = await apiClient.get("/api/speeches/search", {
      params: { q: qq, search_in, include_public, limit: safeLimit },
    });
    const payload = unwrapResponse(res.data);
    return extractSpeechArray(payload);
  } catch (error) {
    handleAxiosError(error);
  }
}

export async function createSpeech(payload) {
  try {
    const res = await apiClient.post("/api/speeches/", payload);
    return unwrapResponse(res.data);
  } catch (error) {
    handleAxiosError(error);
  }
}

export async function uploadSpeech({
  file,
  title,
  speaker,
  topic,
  date,
  date_str,
  location,
  event,
  language = "en",
  is_public = false,
  llm_provider = "openai",
  llm_model = "gpt-4o-mini",
  party,
  role,
  use_research_analysis = false,
} = {}) {
  try {
    if (!file) throw new Error("Missing file");
    if (!title) throw new Error("Missing title");
    if (!speaker) throw new Error("Missing speaker");

    const form = new FormData();
    form.append("file", file);
    form.append("title", title);
    form.append("speaker", speaker);
    if (topic) form.append("topic", topic);
    if (date_str || date) form.append("date_str", date_str || date);
    if (location) form.append("location", location);
    if (event) form.append("event", event);
    if (language) form.append("language", language);
    form.append("is_public", String(Boolean(is_public)));
    if (llm_provider) form.append("llm_provider", llm_provider);
    if (llm_model) form.append("llm_model", llm_model);
    
    // Research analysis fields
    if (party) form.append("party", party);
    if (role) form.append("role", role);
    form.append("use_research_analysis", String(Boolean(use_research_analysis)));

    const res = await uploadClient.post("/api/speeches/upload", form);
    return unwrapResponse(res.data);
  } catch (error) {
    handleAxiosError(error);
  }
}

export async function getSpeech(speechId, { include_text = true, include_analysis = true } = {}) {
  try {
    const res = await apiClient.get(`/api/speeches/${encodeURIComponent(String(speechId))}`, {
      params: { include_text, include_analysis },
    });
    return unwrapResponse(res.data);
  } catch (error) {
    handleAxiosError(error);
  }
}

export async function getSpeechFull(speechId) {
  try {
    const res = await apiClient.get(`/api/speeches/${encodeURIComponent(String(speechId))}/full`);
    return unwrapResponse(res.data);
  } catch (error) {
    handleAxiosError(error);
  }
}

export async function getSpeechStatus(speechId) {
  try {
    const res = await apiClient.get(`/api/speeches/${encodeURIComponent(String(speechId))}/status`);
    return unwrapResponse(res.data);
  } catch (error) {
    handleAxiosError(error);
  }
}

export async function getSpeechMedia(speechId) {
  try {
    const res = await apiClient.get(`/api/speeches/${encodeURIComponent(String(speechId))}/media`);
    return unwrapResponse(res.data);
  } catch (error) {
    handleAxiosError(error);
  }
}

export function getMediaUrl(speechId, speech) {
  const raw = speech?.media_url ?? speech?.mediaUrl ?? "";
  if (!raw) return null;

  if (/^https?:\/\//i.test(raw)) return raw;
  if (raw.startsWith("/")) return `${API_BASE_URL}${raw}`;

  return `${API_BASE_URL}/api/speeches/${encodeURIComponent(String(speechId))}/media`;
}

export async function getAnalysis(speechId, { media_duration_seconds } = {}) {
  const id = encodeURIComponent(String(speechId));

  const candidates = [
    { url: `/api/speeches/${id}/analysis`, params: media_duration_seconds != null ? { media_duration_seconds } : undefined },
    { url: `/api/speeches/${id}/analysis_summary` },
    { url: `/api/speeches/${id}/analysis-summary` },
    { url: `/api/analysis/${id}` },
    { url: `/api/analysis`, params: { speech_id: id } },
  ];

  let lastErr = null;

  for (const c of candidates) {
    try {
      const res = await apiClient.get(c.url, c.params ? { params: c.params } : undefined);
      return unwrapResponse(res.data);
    } catch (err) {
      lastErr = err;
      const status = err?.response?.status;
      if (status && status !== 404 && status !== 405) {
        handleAxiosError(err);
      }
    }
  }

  if (lastErr) handleAxiosError(lastErr);
  throw new Error("Failed to load analysis.");
}

export async function getAnalysisLegacy(speechId) {
  return getAnalysis(speechId);
}

export async function getKeyStatements(speechId, { media_duration_seconds } = {}) {
  try {
    const res = await apiClient.get(`/api/speeches/${encodeURIComponent(String(speechId))}/key-statements`, {
      params: media_duration_seconds != null ? { media_duration_seconds } : undefined,
    });
    return unwrapResponse(res.data);
  } catch (error) {
    handleAxiosError(error);
  }
}

export async function getSections(speechId, { media_duration_seconds } = {}) {
  try {
    const res = await apiClient.get(`/api/speeches/${encodeURIComponent(String(speechId))}/sections`, {
      params: media_duration_seconds != null ? { media_duration_seconds } : undefined,
    });
    return unwrapResponse(res.data);
  } catch (error) {
    handleAxiosError(error);
  }
}

export async function analyzeSpeech(speechId, { force = false } = {}) {
  try {
    const res = await apiClient.post(`/api/speeches/${encodeURIComponent(String(speechId))}/analyze`, null, {
      params: { force: Boolean(force) },
    });
    return unwrapResponse(res.data);
  } catch (error) {
    handleAxiosError(error);
  }
}

export async function reanalyzeSpeechAnalysis(speechId, opts = {}) {
  const id = encodeURIComponent(String(speechId));

  const body = {
    ...opts,
    use_semantic: opts.useSemantic ?? opts.use_semantic,
    include_questions: opts.includeQuestions ?? opts.include_questions,
    question_types: opts.questionTypes ?? opts.question_types,
    max_questions: opts.maxQuestions ?? opts.max_questions,
    media_duration_seconds: opts.mediaDurationSeconds ?? opts.media_duration_seconds ?? (opts.mediaDurationSeconds === null ? null : undefined),
  };

  const candidates = [
    { method: "post", url: `/api/speeches/${id}/reanalyze`, data: body },
    { method: "post", url: `/api/speeches/${id}/analysis/reanalyze`, data: body },
    { method: "post", url: `/api/speeches/${id}/analyze`, data: body },
    { method: "post", url: `/api/analysis/${id}/reanalyze`, data: body },
  ];

  let lastErr = null;

  for (const c of candidates) {
    try {
      const res = await apiClient.request({ method: c.method, url: c.url, data: c.data });
      return unwrapResponse(res.data);
    } catch (e) {
      lastErr = e;
      const st = e?.response?.status;
      if (st && st !== 404 && st !== 405) handleAxiosError(e);
    }
  }

  if (lastErr) handleAxiosError(lastErr);
  throw new Error("Failed to re-analyze speech.");
}

export async function deleteSpeech(speechId) {
  try {
    const res = await apiClient.delete(`/api/speeches/${encodeURIComponent(String(speechId))}`);
    return unwrapResponse(res.data);
  } catch (error) {
    handleAxiosError(error);
  }
}

export async function updateSpeech(speechId, updates) {
  try {
    const res = await apiClient.put(`/api/speeches/${encodeURIComponent(String(speechId))}`, updates);
    return unwrapResponse(res.data);
  } catch (error) {
    handleAxiosError(error);
  }
}

export async function generateQuestions(arg1, arg2 = {}) {
  try {
    let speechId;
    let question_type;
    let max_questions;
    let llm_provider;
    let llm_model;

    if (arg1 && typeof arg1 === "object") {
      speechId = arg1.speechId ?? arg1.speech_id;
      question_type = arg1.questionType ?? arg1.question_type ?? "journalistic";
      max_questions = arg1.maxQuestions ?? arg1.max_questions ?? 5;
      llm_provider = arg1.llmProvider ?? arg1.llm_provider;
      llm_model = arg1.llmModel ?? arg1.llm_model;
    } else {
      speechId = arg1;
      question_type = arg2.question_type ?? arg2.questionType ?? "journalistic";
      max_questions = arg2.max_questions ?? arg2.maxQuestions ?? 5;
      llm_provider = arg2.llm_provider ?? arg2.llmProvider;
      llm_model = arg2.llm_model ?? arg2.llmModel;
    }

    if (speechId === undefined || speechId === null) throw new Error("Missing speechId");

    const id = encodeURIComponent(String(speechId));
    const body = { question_type, max_questions, llm_provider, llm_model };

    const res = await apiClient.post(`/api/speeches/${id}/questions/generate`, body);
    return unwrapResponse(res.data);
  } catch (error) {
    handleAxiosError(error);
  }
}

const _sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const _normProgressForUI = (p) => {
  const n = Number(p);
  if (!Number.isFinite(n)) return null;
  if (n > 0 && n <= 1) return n * 100;
  if (n >= 0 && n <= 100) return n;
  return null;
};

const _statusLooksDone = (s) => {
  const v = String(s || "").toLowerCase();
  return v.includes("done") || v.includes("complete") || v.includes("success") || v === "finished";
};

const _statusLooksFailed = (s) => {
  const v = String(s || "").toLowerCase();
  return v.includes("fail") || v.includes("error");
};

export async function uploadSpeechWithPolling({
  file,
  title,
  speaker,
  topic = "",
  date = "",
  location = "",
  event = "",
  llmProvider = "openai",
  llmModel = "gpt-4o-mini",
  isPublic = false,
  party = "",
  role = "",
  useResearchAnalysis = false,
  onProgress,
  pollIntervalMs = 2500,
  timeoutMs = 20 * 60 * 1000,
} = {}) {
  try {
    if (!file) throw new Error("Missing file");
    if (!title) throw new Error("Missing title");
    if (!speaker) throw new Error("Missing speaker");

    const form = new FormData();
    form.append("file", file);
    form.append("title", title);
    form.append("speaker", speaker);
    if (topic) form.append("topic", topic);
    if (date) form.append("date_str", date);
    if (location) form.append("location", location);
    if (event) form.append("event", event);
    form.append("is_public", String(Boolean(isPublic)));
    form.append("llm_provider", llmProvider);
    form.append("llm_model", llmModel);
    
    // Research analysis fields
    if (party) form.append("party", party);
    if (role) form.append("role", role);
    form.append("use_research_analysis", String(Boolean(useResearchAnalysis)));

    onProgress?.("Preparing upload…", 1);

    const res = await uploadClient.post("/api/speeches/upload", form, {
      onUploadProgress: (evt) => {
        const total = evt.total || 0;
        const loaded = evt.loaded || 0;
        if (total > 0) {
          const frac = Math.max(0, Math.min(1, loaded / total));
          onProgress?.("Uploading…", Math.round(frac * 25));
        } else {
          onProgress?.("Uploading…", 5);
        }
      },
    });

    const payload = unwrapResponse(res.data) || {};
    const speechId = payload.speech_id ?? payload.speechId ?? payload.id ?? payload?.speech?.id ?? null;

    if (!speechId) throw new Error("Upload succeeded but response did not include speechId.");

    onProgress?.("Upload complete. Processing…", 30);

    const startPolling = async () => {
      const start = Date.now();

      while (Date.now() - start < timeoutMs) {
        try {
          const st = await getSpeechStatus(speechId);
          const status = st?.status ?? st?.analysis_status ?? st?.state;
          const msg = st?.message ?? st?.detail ?? "";
          const p = _normProgressForUI(st?.progress ?? st?.analysis_progress ?? st?.pct ?? st?.percent);

          if (_statusLooksFailed(status)) throw new Error(msg || "Analysis failed.");

          if (_statusLooksDone(status) || st?.analysis_ready === true) {
            onProgress?.("Analysis complete!", 100);
            return true;
          }

          onProgress?.(msg || "Processing…", p != null ? Math.max(30, Math.min(95, p)) : 60);
          await _sleep(pollIntervalMs);
          continue;
        } catch (e) {
          const httpStatus = e?.response?.status;
          if (httpStatus && httpStatus !== 404 && httpStatus !== 405) throw e;
        }

        try {
          const a = await getAnalysis(speechId);
          if (a && typeof a === "object") {
            onProgress?.("Analysis complete!", 100);
            return true;
          }
        } catch {}

        onProgress?.("Processing…", 70);
        await _sleep(pollIntervalMs);
      }

      throw new Error("Polling timed out. The server may still be processing.");
    };

    return { speechId, startPolling };
  } catch (error) {
    handleAxiosError(error, { logoutOn401: true });
  }
}

export function isAuthenticated() {
  return Boolean(getAccessToken());
}

export default apiClient;