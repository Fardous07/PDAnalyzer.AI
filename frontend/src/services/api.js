
// frontend/src/services/api.js
import axios from "axios";

// Backend base URL (no trailing slash)
export const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || "http://localhost:8000").replace(/\/+$/, "");

console.log("API Base URL:", API_BASE_URL);

// -------------------------------------------------------
// Time Utility Functions
// -------------------------------------------------------

export function formatTime(seconds) {
  if (seconds === null || seconds === undefined || Number.isNaN(seconds)) return "00:00";
  const s = Math.max(0, Number(seconds));
  const mins = Math.floor(s / 60);
  const secs = Math.floor(s % 60);
  return `${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;
}

export function jumpToTime(seconds) {
  const videoPlayer = document.getElementById("video-player");
  if (videoPlayer) {
    videoPlayer.currentTime = Number(seconds) || 0;
    videoPlayer.play?.();
  }
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

// -------------------------------------------------------
// Auth token helpers
// -------------------------------------------------------

const TOKEN_KEY = "token";
const REFRESH_KEY = "refresh_token";
const USER_KEY = "user";

export function getAccessToken() {
  const t = localStorage.getItem(TOKEN_KEY);
  if (!t || t === "undefined" || t === "null") return null;
  return t;
}

export function getRefreshToken() {
  const t = localStorage.getItem(REFRESH_KEY);
  if (!t || t === "undefined" || t === "null") return null;
  return t;
}

export function setAuthTokens({ accessToken, refreshToken, user } = {}) {
  if (accessToken) localStorage.setItem(TOKEN_KEY, accessToken);
  if (refreshToken) localStorage.setItem(REFRESH_KEY, refreshToken);

  if (user !== undefined) {
    try {
      localStorage.setItem(USER_KEY, JSON.stringify(user));
    } catch {
      // ignore
    }
  }
}

export function clearAuthTokens() {
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(REFRESH_KEY);
  localStorage.removeItem(USER_KEY);
}

// -------------------------------------------------------
// Axios clients
// -------------------------------------------------------

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 60000,
  withCredentials: false,
});

const uploadClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 300000,
  withCredentials: false,
});

// Attach auth token to requests
[apiClient, uploadClient].forEach((client) => {
  client.interceptors.request.use(
    (config) => {
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
    (error) => {
      console.error("API Error:", {
        message: error.message,
        status: error.response?.status,
        data: error.response?.data,
        url: error.config?.url,
      });
      return Promise.reject(error);
    }
  );
});

// -------------------------------------------------------
// Response unwrap helpers
// -------------------------------------------------------

function unwrapResponse(data) {
  // Common backend shape: { success, data, error, message, timestamp }
  if (data && typeof data === "object" && "success" in data) {
    if (data.success === false) {
      const msg = data.error || data.message || data.detail || "Request failed";
      throw new Error(msg);
    }
    if ("data" in data) return data.data;
    return data;
  }
  return data;
}

function safeArray(v) {
  return Array.isArray(v) ? v : [];
}

// -------------------------------------------------------
// Ideology normalization (Centrist-only; NO Neutral)
// -------------------------------------------------------

const LIB = "Libertarian";
const AUTH = "Authoritarian";
const CENTRIST = "Centrist";

function normalizeScores(scores) {
  const s = scores && typeof scores === "object" ? scores : {};
  const lib = Number(s.Libertarian ?? 0);
  const auth = Number(s.Authoritarian ?? 0);

  // Centrist must be provided by backend; otherwise compute complement.
  let cen = Number(s.Centrist ?? 0);
  if (!Number.isFinite(cen) || (cen === 0 && !("Centrist" in s))) {
    const comp = 100 - lib - auth;
    cen = Number.isFinite(comp) ? Math.max(0, comp) : 0;
  }

  return {
    Libertarian: Number.isFinite(lib) ? lib : 0,
    Authoritarian: Number.isFinite(auth) ? auth : 0,
    Centrist: Number.isFinite(cen) ? cen : 0,
  };
}

// Deep sanitizer to remove legacy "Neutral", normalize family/subtype recursively
function purgeLegacyNeutralEverywhere(obj) {
  if (Array.isArray(obj)) return obj.map(purgeLegacyNeutralEverywhere);

  if (obj && typeof obj === "object") {
    const out = {};
    for (const [k, v] of Object.entries(obj)) {
      out[k] = purgeLegacyNeutralEverywhere(v);
    }

    // Sanitize any 'scores' dict
    if (out.scores && typeof out.scores === "object") {
      const s = { ...out.scores };
      if ("Neutral" in s && !("Centrist" in s)) s.Centrist = s.Neutral;
      delete s.Neutral;
      out.scores = normalizeScores(s);
    }

    // Normalize common family/subtype fields
    const famKeys = ["ideology_family", "dominant_family", "family"];
    const subKeys = ["ideology_subtype", "dominant_subtype", "subtype"];

    const famKey = famKeys.find((fk) => fk in out);
    if (famKey) {
      const famIn = String(out[famKey] || "").trim();
      const fam = famIn === "Neutral" ? CENTRIST : famIn;
      const normalizedFam = [LIB, AUTH, CENTRIST].includes(fam) ? fam : CENTRIST;
      out[famKey] = normalizedFam;

      subKeys.forEach((sk) => {
        if (sk in out) {
          const sub = String(out[sk] || "").trim();
          out[sk] = normalizedFam === CENTRIST ? null : (sub || null);
        }
      });
    }

    return out;
  }

  return obj;
}

function normalizeIdeologyPayload(payload) {
  if (!payload || typeof payload !== "object") return payload;

  // Deep sanitize first (covers nested data too)
  payload = purgeLegacyNeutralEverywhere(payload);

  // Idempotent: ensure top-level normalization
  if (payload.scores && typeof payload.scores === "object") {
    payload.scores = normalizeScores(payload.scores);
  }

  if (payload.speech_level && typeof payload.speech_level === "object") {
    if (payload.speech_level.scores && typeof payload.speech_level.scores === "object") {
      payload.speech_level.scores = normalizeScores(payload.speech_level.scores);
    }
    if (payload.speech_level.dominant_family === CENTRIST) payload.speech_level.dominant_subtype = null;
  }

  if (payload.ideology_family === CENTRIST) payload.ideology_subtype = null;

  return payload;
}

// -------------------------------------------------------
// Centralized error handler
// -------------------------------------------------------

function handleAxiosError(error, { logoutOn401 = true } = {}) {
  console.error("Axios error details:", {
    code: error.code,
    message: error.message,
    response: error.response?.data,
    status: error.response?.status,
  });

  const status = error?.response?.status;
  const responseData = error?.response?.data;

  let backendDetail = null;
  if (responseData) {
    if (typeof responseData === "object") {
      backendDetail = responseData.detail || responseData.error || responseData.message;
    } else if (typeof responseData === "string") {
      backendDetail = responseData;
    }
  }

  let message =
    (typeof backendDetail === "string" && backendDetail.trim() ? backendDetail : null) ||
    error?.message ||
    "Request failed. Please try again.";

  const hasBackendMessage = typeof backendDetail === "string" && backendDetail.trim();

  if (status === 401 && logoutOn401) {
    clearAuthTokens();
    if (!hasBackendMessage) message = "Session expired. Please log in again.";
  } else if (status === 403) {
    if (!hasBackendMessage) message = "You do not have permission to access this resource.";
  } else if (status === 404) {
    if (!hasBackendMessage) message = "Resource not found.";
  } else if (status === 413) {
    if (!hasBackendMessage) message = "File too large. Please try a smaller file.";
  } else if (status === 429) {
    if (!hasBackendMessage) message = "Too many requests. Please try again later.";
  } else if (status >= 500) {
    if (!hasBackendMessage) message = "Server error. Please try again later.";
  }

  // Browser CORS/network failures show up as "Network Error"
  if (error.message?.includes("Network Error") || error.code === "ERR_NETWORK") {
    message = "Cannot connect to the server. Please check if the backend is running.";
  }

  if (error?.code === "ECONNABORTED") {
    message = "Request timed out. Please try again.";
  }

  throw new Error(message);
}

// -------------------------------------------------------
// Health Check
// -------------------------------------------------------

export async function checkBackendConnection() {
  try {
    const res = await fetch(`${API_BASE_URL}/health`);
    return res.ok ? { ok: true, status: res.status, data: await res.json() } : { ok: false, status: res.status };
  } catch (error) {
    return { ok: false, error: error.message };
  }
}

// -------------------------------------------------------
// Authentication API
// -------------------------------------------------------

export const authAPI = {
  login: async (credentials) => {
    try {
      const res = await apiClient.post("/api/auth/login", credentials);
      const data = res.data;

      // preferred shape: { success, tokens, user }
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

      // fallback shapes
      if (data?.access_token && data?.user) {
        setAuthTokens({ accessToken: data.access_token, refreshToken: data.refresh_token, user: data.user });
        return data;
      }

      if (data?.data) {
        if (data.data.tokens && data.data.user) {
          const out = {
            access_token: data.data.tokens.access_token,
            refresh_token: data.data.tokens.refresh_token,
            user: data.data.user,
            token_type: data.data.tokens.token_type || "bearer",
            expires_in: data.data.tokens.expires_in,
          };
          setAuthTokens({ accessToken: out.access_token, refreshToken: out.refresh_token, user: out.user });
          return out;
        }
        if (data.data.access_token && data.data.user) {
          setAuthTokens({
            accessToken: data.data.access_token,
            refreshToken: data.data.refresh_token,
            user: data.data.user,
          });
          return data.data;
        }
      }

      throw new Error("Invalid login response format");
    } catch (error) {
      handleAxiosError(error, { logoutOn401: false });
    }
  },

  // ✅ IMPORTANT: DO NOT AUTO-LOGIN ON REGISTER
  register: async (userData) => {
    try {
      const res = await apiClient.post("/api/auth/register", userData);
      const data = res.data;

      // backend likely returns { success, user, tokens } but we do NOT store tokens here
      if (data?.success) {
        return {
          success: true,
          message: data?.message || "Registration successful. Please sign in.",
        };
      }

      // wrapped response fallback
      if (data?.data?.success) {
        return {
          success: true,
          message: data?.data?.message || "Registration successful. Please sign in.",
        };
      }

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

      const res = await apiClient.post("/api/auth/refresh", { refresh_token });
      const data = unwrapResponse(res.data);

      const access = data?.access_token || data?.tokens?.access_token;
      const refresh = data?.refresh_token || data?.tokens?.refresh_token;

      if (!access) throw new Error("Invalid refresh response: missing access token.");

      setAuthTokens({ accessToken: access, RefreshToken: refresh || refresh_token });
      return data;
    } catch (error) {
      handleAxiosError(error, { logoutOn401: true });
    }
  },
};

// -------------------------------------------------------
// Speech APIs (use trailing slash to avoid 307)
// -------------------------------------------------------

export async function listSpeeches(params = {}) {
  try {
    // IMPORTANT: backend route is "/api/speeches/" (router has @router.get("/"))
    const res = await apiClient.get("/api/speeches/", { params });
    const payload = unwrapResponse(res.data);

    if (payload && typeof payload === "object") {
      if (Array.isArray(payload.speeches)) return payload.speeches;
      if (Array.isArray(payload.data?.speeches)) return payload.data.speeches;
    }
    if (Array.isArray(payload)) return payload;
    return [];
  } catch (error) {
    handleAxiosError(error);
  }
}

export async function getSpeech(speechId, params = {}) {
  try {
    const res = await apiClient.get(`/api/speeches/${speechId}`, { params });
    return unwrapResponse(res.data);
  } catch (error) {
    handleAxiosError(error);
  }
}

export async function getSpeechFull(speechId) {
  try {
    const res = await apiClient.get(`/api/speeches/${speechId}/full`);
    return unwrapResponse(res.data);
  } catch (error) {
    handleAxiosError(error);
  }
}

export async function getSpeechStatus(speechId) {
  try {
    const res = await apiClient.get(`/api/speeches/${speechId}/status`);
    return unwrapResponse(res.data);
  } catch (error) {
    handleAxiosError(error);
  }
}

// -------------------------------------------------------
// Analysis APIs
// -------------------------------------------------------

export async function getAnalysis(speechId, { mediaDurationSeconds } = {}) {
  try {
    const params = {};
    if (Number.isFinite(mediaDurationSeconds) && Number(mediaDurationSeconds) > 0) {
      params.media_duration_seconds = Number(mediaDurationSeconds);
    }
    const res = await apiClient.get(`/api/analysis/speech/${speechId}`, { params });
    const data = unwrapResponse(res.data);
    return normalizeIdeologyPayload(data);
  } catch (error) {
    handleAxiosError(error);
  }
}

export async function reanalyzeSpeechAnalysis(
  speechId,
  {
    useSemantic = true,
    threshold = 0.6,
    includeQuestions = true,
    questionTypes = ["journalistic", "technical"],
    maxQuestions = 6,
    mediaDurationSeconds = null,
    llmProvider = null,
    llmModel = null,
  } = {}
) {
  try {
    const body = {
      use_semantic: Boolean(useSemantic),
      threshold: Number(threshold),
      include_questions: Boolean(includeQuestions),
      question_types: Array.isArray(questionTypes) ? questionTypes : ["journalistic", "technical"],
      max_questions: Math.max(1, Math.min(8, Number(maxQuestions) || 6)),
      media_duration_seconds:
        Number.isFinite(mediaDurationSeconds) && Number(mediaDurationSeconds) > 0
          ? Number(mediaDurationSeconds)
          : null,
      llm_provider: llmProvider || null,
      llm_model: llmModel || null,
    };

    const res = await apiClient.post(`/api/analysis/speech/${speechId}/reanalyze`, body);
    const data = unwrapResponse(res.data);
    return normalizeIdeologyPayload(data);
  } catch (error) {
    handleAxiosError(error);
  }
}

export async function generateQuestions({
  speechId,
  questionType = "journalistic",
  maxQuestions = 5,
  llmProvider = null,
  llmModel = null,
} = {}) {
  try {
    const body = {
      speech_id: Number(speechId),
      question_type: String(questionType || "journalistic"),
      max_questions: Math.max(1, Math.min(8, Number(maxQuestions) || 5)),
      llm_provider: llmProvider || null,
      llm_model: llmModel || null,
    };
    const res = await apiClient.post(`/api/analysis/questions/generate`, body);
    return unwrapResponse(res.data);
  } catch (error) {
    handleAxiosError(error);
  }
}

// -------------------------------------------------------
// Upload + Polling
// -------------------------------------------------------

export async function uploadSpeechWithPolling({
  file,
  title,
  speaker,
  topic = "",
  date = "",
  location = "",
  event = "",
  language = "en",
  llmProvider = "openai",
  llmModel = "gpt-4o-mini",
  isPublic = false,
  onProgress = () => {},
}) {
  onProgress("Preparing file upload...", 10);

  const formData = new FormData();
  formData.append("file", file);
  formData.append("title", title.trim());
  formData.append("speaker", speaker.trim());
  if (topic) formData.append("topic", topic.trim());
  if (date) formData.append("date_str", date);
  if (location) formData.append("location", location);
  if (event) formData.append("event", event);
  formData.append("language", language);
  formData.append("llm_provider", llmProvider);
  formData.append("llm_model", llmModel);
  formData.append("is_public", isPublic.toString());

  let uploadResponse;
  try {
    onProgress("Uploading file to server...", 30);

    const res = await uploadClient.post("/api/speeches/upload", formData, {
      headers: { "Content-Type": "multipart/form-data" },
    });

    uploadResponse = res.data;
    onProgress("File uploaded successfully. Starting analysis...", 70);
  } catch (error) {
    if (error?.response?.status === 401) throw new Error("Session expired. Please log in again.");
    if (error?.response?.status === 413) throw new Error("File too large. Please try a smaller file.");
    if (error?.code === "ECONNABORTED") throw new Error("Upload timeout. The file may be too large.");

    let errorMessage = "Upload failed";
    if (error?.response?.data) {
      const data = error.response.data;
      errorMessage = data.detail || data.error || data.message || errorMessage;
    }
    throw new Error(errorMessage);
  }

  const data = uploadResponse;
  const speechId = data?.speech_id || data?.id || data?.data?.speech_id || data?.data?.id;

  if (!speechId) {
    console.error("Upload response:", uploadResponse);
    throw new Error("Upload succeeded but no speech ID returned");
  }

  onProgress("Analysis started...", 80);

  async function startPolling({ intervalMs = 5000, maxAttempts = 360 } = {}) {
    let attempt = 0;

    while (attempt < maxAttempts) {
      attempt += 1;

      let statusRes;
      try {
        statusRes = await apiClient.get(`/api/speeches/${speechId}/status`);
      } catch (error) {
        if (error?.response?.status === 401) throw new Error("Session expired. Please log in again.");
        if (error?.response?.status === 404) throw new Error("Speech not found.");
        handleAxiosError(error);
      }

      const statusPayload = unwrapResponse(statusRes?.data);
      const statusVal = statusPayload?.status;

      if (statusVal === "completed") {
        onProgress("Analysis complete!", 100);
        return await getSpeech(speechId, { include_text: true, include_analysis: true });
      }

      if (statusVal === "failed") {
        onProgress("Analysis failed.", 100);
        const msg = statusPayload?.error_message || statusPayload?.detail || "Analysis failed";
        throw new Error(msg);
      }

      const estimatedProgress = Math.min(99, 80 + (attempt / maxAttempts) * 19);
      const statusMessages = {
        pending: "Queued for analysis...",
        processing: "Analyzing speech...",
        uploaded: "Starting analysis...",
      };

      onProgress(statusMessages[statusVal] || "Processing...", estimatedProgress);
      await new Promise((resolve) => setTimeout(resolve, intervalMs));
    }

    // fallback
    try {
      return await getSpeech(speechId, { include_text: true, include_analysis: true });
    } catch {
      throw new Error("Analysis taking longer than expected. Check your dashboard.");
    }
  }

  return { speechId, startPolling };
}

// -------------------------------------------------------
// Media URL helper
// -------------------------------------------------------

export function getMediaUrl(_speechId, speechData = null) {
  const url = speechData?.media_url || speechData?.source_url;
  if (!url || typeof url !== "string") return null;

  if (url.startsWith("http://") || url.startsWith("https://")) return url;
  if (url.startsWith("/")) return `${API_BASE_URL}${url}`;

  return `${API_BASE_URL}/media/uploads/${url}`;
}

// -------------------------------------------------------
// Update / Delete
// -------------------------------------------------------

export async function deleteSpeech(speechId) {
  try {
    const res = await apiClient.delete(`/api/speeches/${speechId}`);
    return unwrapResponse(res.data);
  } catch (error) {
    handleAxiosError(error);
  }
}

export async function updateSpeech(speechId, updates) {
  try {
    const res = await apiClient.put(`/api/speeches/${speechId}`, updates);
    return unwrapResponse(res.data);
  } catch (error) {
    handleAxiosError(error);
  }
}

// -------------------------------------------------------
// Utils
// -------------------------------------------------------

export function isAuthenticated() {
  return Boolean(getAccessToken());
}

export function getUserFromStorage() {
  try {
    const userStr = localStorage.getItem(USER_KEY);
    return userStr ? JSON.parse(userStr) : null;
  } catch {
    return null;
  }
}

export default apiClient;
