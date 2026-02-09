// frontend/src/pages/CreateProjectPage.jsx
import React, { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  Upload,
  FileVideo,
  FileAudio,
  FileText,
  Brain,
  AlertCircle,
  ChevronDown,
  Loader2,
  CheckCircle,
  XCircle,
} from "lucide-react";
import { useAuth } from "../context/AuthContext";
import { uploadSpeechWithPolling } from "../services/api";

const DEFAULT_MAX_SIZE = 100 * 1024 * 1024; // 100MB
const DEFAULT_MAX_SPEECHES = 50;

const CreateProjectPage = ({ refreshProjects }) => {
  const { token, user, refreshProjects: refreshProjectsFromContext } = useAuth();
  const navigate = useNavigate();

  const refreshFn =
    typeof refreshProjects === "function" ? refreshProjects : refreshProjectsFromContext;

  const [title, setTitle] = useState("");
  const [speaker, setSpeaker] = useState("");
  const [topic, setTopic] = useState("");
  const [date, setDate] = useState("");
  const [location, setLocation] = useState("");
  const [event, setEvent] = useState("");
  const [file, setFile] = useState(null);

  const [llmProvider, setLlmProvider] = useState("openai");
  const [llmModel, setLlmModel] = useState("gpt-4o-mini");

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [uploadProgress, setUploadProgress] = useState(0);
  const [currentStatus, setCurrentStatus] = useState("");
  const [isProcessing, setIsProcessing] = useState(false);
  const [pollingError, setPollingError] = useState("");

  const modelOptions = useMemo(
    () => ({
      openai: [
        {
          value: "gpt-4o-mini",
          label: "GPT-4o Mini",
          description: "Fast, accurate, cost-effective",
        },
        { value: "gpt-4o", label: "GPT-4o", description: "Higher accuracy, richer reasoning" },
        { value: "gpt-4", label: "GPT-4", description: "Legacy high-precision model" },
      ],
      groq: [
        {
          value: "mixtral-8x7b-32768",
          label: "Mixtral 8x7b",
          description: "High-performance open-source analysis",
        },
        {
          value: "llama3-70b-8192",
          label: "Llama 3 70B",
          description: "Deep contextual understanding",
        },
        {
          value: "llama3-8b-8192",
          label: "Llama 3 8B",
          description: "Fast inference for rapid analysis",
        },
      ],
    }),
    []
  );

  const normalizeProgress = (p) => {
    const num = Number(p);
    if (!Number.isFinite(num)) return 0;
    if (num > 0 && num <= 1) return Math.round(num * 100);
    return Math.max(0, Math.min(100, Math.round(num)));
  };

  const getExt = (name) => (name || "").split(".").pop()?.toLowerCase() || "";

  const isDocLike = (ext) => ["pdf", "doc", "docx"].includes(ext);
  const isTextFile = (ext) => ["txt", "md", "json", "csv", "pdf", "doc", "docx"].includes(ext);
  const isVideoFile = (ext) => ["mp4", "mov", "avi", "mkv", "webm", "flv", "wmv"].includes(ext);
  const isAudioFile = (ext) => ["mp3", "wav", "m4a", "aac", "flac", "ogg"].includes(ext);

  const maxSize =
    user?.max_file_size && !Number.isNaN(Number(user.max_file_size))
      ? Number(user.max_file_size)
      : DEFAULT_MAX_SIZE;

  const maxSpeeches =
    user?.max_speeches && !Number.isNaN(Number(user.max_speeches))
      ? Number(user.max_speeches)
      : DEFAULT_MAX_SPEECHES;

  const usedCountRaw =
    (typeof user?.speech_count === "number" ? user.speech_count : null) ??
    (typeof user?.usage?.speech_count === "number" ? user.usage.speech_count : null);

  const hasUsageCount = typeof usedCountRaw === "number" && Number.isFinite(usedCountRaw);
  const usedCount = hasUsageCount ? usedCountRaw : null;
  const remainingAnalyses = hasUsageCount ? Math.max(0, maxSpeeches - usedCount) : null;

  const handleFileChange = (e) => {
    const f = e.target.files?.[0];
    if (!f) return;

    if (f.size > maxSize) {
      setError(
        `File size exceeds limit (${(f.size / (1024 * 1024)).toFixed(2)} MB). ` +
          `Maximum allowed: ${(maxSize / (1024 * 1024)).toFixed(0)} MB.`
      );
      setFile(null);
      setUploadProgress(0);
      setCurrentStatus("");
      setIsProcessing(false);
      setPollingError("");
      return;
    }

    const ext = getExt(f.name);
    const isVideo = isVideoFile(ext);
    const isAudio = isAudioFile(ext);
    const isText = isTextFile(ext);

    if (!isVideo && !isAudio && !isText) {
      setError(
        "Unsupported file type. Upload video (.mp4, .mov, .avi), audio (.mp3, .wav, .m4a), " +
          "or text (.txt, .md, .pdf, .doc, .docx) files."
      );
      setFile(null);
      return;
    }

    // Informational warnings only
    if (isVideo || isAudio) {
      setError(
        `Note: You selected a ${isVideo ? "video" : "audio"} file. ` +
          "The backend will transcribe it before analysis. Large files may take longer."
      );
    } else if (isDocLike(ext)) {
      setError(`Note: You selected a .${ext.toUpperCase()} file. If parsing fails, try converting it to .txt.`);
    } else {
      setError("");
    }

    setFile(f);
    setUploadProgress(0);
    setCurrentStatus("");
    setIsProcessing(false);
    setPollingError("");
  };

  const getFileIcon = () => {
    if (!file) return <FileVideo className="w-6 h-6 text-indigo-400" />;
    const ext = getExt(file.name);
    if (isVideoFile(ext)) return <FileVideo className="w-6 h-6 text-indigo-400" />;
    if (isAudioFile(ext)) return <FileAudio className="w-6 h-6 text-emerald-400" />;
    return <FileText className="w-6 h-6 text-purple-400" />;
  };

  const handleProviderChange = (newProvider) => {
    setLlmProvider(newProvider);
    setLlmModel(modelOptions[newProvider]?.[0]?.value || "gpt-4o-mini");
  };

  const extractErrorMessage = (err) => {
    const fromResponse =
      err?.response?.data?.detail ||
      err?.response?.data?.error ||
      err?.response?.data?.message ||
      err?.response?.data?.error_message;

    let msg = fromResponse || err?.message || "Analysis failed. Please try again.";
    const lower = String(msg).toLowerCase();

    if (lower.includes("openai_api_key") && lower.includes("missing")) {
      return "OPENAI_API_KEY missing — configure your OpenAI API key in the backend.";
    }
    if (lower.includes("groq_api_key") && lower.includes("missing")) {
      return "GROQ_API_KEY missing — configure your Groq API key in the backend.";
    }
    if (err?.code === "ECONNABORTED" || lower.includes("timeout") || lower.includes("timed out")) {
      return "Request timed out. Try a smaller file or check your server/network.";
    }
    if (lower.includes("cors") || err?.code === "ERR_NETWORK") {
      return "Cannot connect to the server. Check that the backend is running and CORS is configured.";
    }
    if (err?.response?.status === 401) {
      return "Session expired. Please log in again.";
    }

    return String(msg);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    setError("");
    setUploadProgress(0);
    setCurrentStatus("");
    setIsProcessing(false);
    setPollingError("");

    if (!token) {
      setError("Authentication required. Please log in to continue.");
      navigate("/login");
      return;
    }

    if (!title.trim() || !speaker.trim()) {
      setError("Title and speaker are required.");
      return;
    }

    if (!file) {
      setError("Please select a speech file to analyze.");
      return;
    }

    if (remainingAnalyses !== null && remainingAnalyses <= 0) {
      setError("You have reached your analysis limit.");
      return;
    }

    setLoading(true);

    let startedPolling = false;

    try {
      const { speechId, startPolling } = await uploadSpeechWithPolling({
        file,
        title: title.trim(),
        speaker: speaker.trim(),
        topic: topic.trim(),
        date,
        location: location.trim(),
        event: event.trim(),
        llmProvider,
        llmModel,
        isPublic: false,
        onProgress: (message, progress) => {
          setCurrentStatus(message || "Processing...");
          setUploadProgress(normalizeProgress(progress));
        },
      });

      if (typeof refreshFn === "function") {
        try {
          await refreshFn();
        } catch {
          // ignore
        }
      }

      startedPolling = true;
      setIsProcessing(true);
      setCurrentStatus("Polling for analysis completion...");

      startPolling()
        .then(async () => {
          setCurrentStatus("Analysis complete! Redirecting...");
          setUploadProgress(100);
          setIsProcessing(false);

          if (typeof refreshFn === "function") {
            try {
              await refreshFn();
            } catch {
              // ignore
            }
          }

          setTimeout(() => navigate(`/analysis/${speechId}`), 600);
        })
        .catch((pollErr) => {
          const errMsg = extractErrorMessage(pollErr);
          setPollingError(errMsg);
          setCurrentStatus("Analysis encountered an issue.");
          setUploadProgress(100);
          setIsProcessing(false);
        });
    } catch (err) {
      let errorMessage = extractErrorMessage(err);

      if (errorMessage.toLowerCase().includes("session expired")) {
        localStorage.removeItem("token");
        localStorage.removeItem("user");
        setError("Session expired. Please log in again.");
        setTimeout(() => navigate("/login"), 600);
        return;
      }

      if (errorMessage.toLowerCase().includes("timed out")) {
        errorMessage =
          "Upload/analysis timed out. If the server is still running, the analysis may still finish. " +
          "Check your dashboard for status.";
      }

      setError(errorMessage);
      setIsProcessing(false);
      setCurrentStatus("");
    } finally {
      setLoading(false);
      if (!startedPolling) setIsProcessing(false);
    }
  };

  const getStatusIcon = () => {
    if (pollingError) return <XCircle className="w-5 h-5 text-rose-400" />;
    if (uploadProgress === 100 && !pollingError && !isProcessing)
      return <CheckCircle className="w-5 h-5 text-emerald-400" />;
    if (loading || isProcessing) return <Loader2 className="w-5 h-5 text-indigo-400 animate-spin" />;
    return null;
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900">
      <div className="max-w-5xl mx-auto space-y-6 p-4 sm:p-6 lg:p-8">
        <div className="flex items-start justify-between gap-4">
          <div>
            <h1 className="text-2xl font-bold text-white">Create Analysis</h1>
            <p className="text-sm text-gray-400 mt-1">
              Upload a speech and generate ideological analysis results.
            </p>
          </div>

          <button
            type="button"
            onClick={() => navigate("/dashboard")}
            className="px-4 py-2 rounded-xl bg-gray-800/60 hover:bg-gray-700/60 border border-gray-700/60 text-gray-200 text-sm transition-colors"
          >
            Back to Dashboard
          </button>
        </div>

        {error && (
          <div className="border border-rose-700/40 bg-rose-900/20 text-sm text-rose-200 rounded-xl p-4 flex items-start gap-3">
            <AlertCircle className="w-5 h-5 flex-shrink-0 mt-0.5 text-rose-300" />
            <span>{error}</span>
          </div>
        )}

        {pollingError && (
          <div className="border border-amber-700/40 bg-amber-900/20 text-sm text-amber-200 rounded-xl p-4 flex items-start gap-3">
            <AlertCircle className="w-5 h-5 flex-shrink-0 mt-0.5 text-amber-300" />
            <div>
              <span>{pollingError}</span>
              <p className="text-xs mt-1 text-amber-200/80">
                The speech was uploaded. If the server stays running, analysis may still complete.
                Check the dashboard.
              </p>
            </div>
          </div>
        )}

        {(loading || isProcessing || currentStatus) && (
          <div className="bg-gray-800/50 border border-gray-700/50 rounded-2xl p-4 shadow-sm">
            <div className="flex items-center gap-3 mb-3">
              {getStatusIcon()}
              <div className="flex-1">
                <div className="flex justify-between text-sm text-gray-200 mb-1">
                  <span className="font-medium">{currentStatus || "Initializing..."}</span>
                  <span className="font-semibold">{uploadProgress}%</span>
                </div>
                <div className="w-full bg-gray-900/60 rounded-full h-2">
                  <div
                    className={`h-2 rounded-full transition-all duration-300 ${
                      pollingError
                        ? "bg-amber-500"
                        : uploadProgress === 100
                        ? "bg-emerald-500"
                        : "bg-gradient-to-r from-indigo-500 to-purple-500"
                    }`}
                    style={{ width: `${uploadProgress}%` }}
                  />
                </div>
              </div>
            </div>

            <div className="text-xs text-gray-400 mt-2 flex items-center justify-between">
              <span>{file && `File: ${file.name}`}</span>
              <span>
                {uploadProgress < 100
                  ? currentStatus?.toLowerCase().includes("upload")
                    ? "Uploading..."
                    : "Processing..."
                  : pollingError
                  ? "Queued"
                  : "Complete"}
              </span>
            </div>

            {pollingError && (
              <div className="mt-3">
                <button
                  onClick={() => navigate("/dashboard")}
                  className="w-full text-sm py-2.5 rounded-xl bg-indigo-900/30 text-indigo-200 hover:bg-indigo-900/40 border border-indigo-700/40 transition-colors"
                  type="button"
                >
                  Go to Dashboard to Check Status
                </button>
              </div>
            )}
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Metadata */}
            <section className="lg:col-span-1 bg-gray-800/50 border border-gray-700/50 rounded-2xl p-5 space-y-5 shadow-sm">
              <div className="flex items-center gap-3">
                <div className="h-10 w-10 rounded-xl bg-indigo-500/15 border border-indigo-500/20 flex items-center justify-center">
                  <Brain className="w-5 h-5 text-indigo-300" />
                </div>
                <div>
                  <h2 className="text-sm font-semibold text-white">Details</h2>
                  <p className="text-xs text-gray-400">Saved with the speech record</p>
                </div>
              </div>

              <div className="space-y-4">
                <div>
                  <label className="block text-xs font-medium text-gray-300 mb-2">
                    Speech Title *
                  </label>
                  <input
                    type="text"
                    value={title}
                    onChange={(e) => setTitle(e.target.value)}
                    placeholder="e.g., State of the Union Address"
                    className="w-full text-sm rounded-xl border border-gray-700/60 bg-gray-900/40 text-white px-3 py-2.5 focus:outline-none focus:ring-2 focus:ring-indigo-500/40 focus:border-indigo-500/40"
                    disabled={loading || isProcessing}
                    required
                  />
                </div>

                <div>
                  <label className="block text-xs font-medium text-gray-300 mb-2">
                    Speaker *
                  </label>
                  <input
                    type="text"
                    value={speaker}
                    onChange={(e) => setSpeaker(e.target.value)}
                    placeholder="e.g., Prime Minister..."
                    className="w-full text-sm rounded-xl border border-gray-700/60 bg-gray-900/40 text-white px-3 py-2.5 focus:outline-none focus:ring-2 focus:ring-indigo-500/40 focus:border-indigo-500/40"
                    disabled={loading || isProcessing}
                    required
                  />
                </div>

                <div>
                  <label className="block text-xs font-medium text-gray-300 mb-2">
                    Topic (optional)
                  </label>
                  <input
                    type="text"
                    value={topic}
                    onChange={(e) => setTopic(e.target.value)}
                    placeholder="e.g., economy, foreign policy"
                    className="w-full text-sm rounded-xl border border-gray-700/60 bg-gray-900/40 text-white px-3 py-2.5 focus:outline-none focus:ring-2 focus:ring-indigo-500/40 focus:border-indigo-500/40"
                    disabled={loading || isProcessing}
                  />
                </div>

                <div>
                  <label className="block text-xs font-medium text-gray-300 mb-2">
                    Date (optional)
                  </label>
                  <input
                    type="date"
                    value={date}
                    onChange={(e) => setDate(e.target.value)}
                    className="w-full text-sm rounded-xl border border-gray-700/60 bg-gray-900/40 text-white px-3 py-2.5 focus:outline-none focus:ring-2 focus:ring-indigo-500/40 focus:border-indigo-500/40"
                    disabled={loading || isProcessing}
                  />
                </div>

                <div>
                  <label className="block text-xs font-medium text-gray-300 mb-2">
                    Location (optional)
                  </label>
                  <input
                    type="text"
                    value={location}
                    onChange={(e) => setLocation(e.target.value)}
                    placeholder="e.g., Oslo"
                    className="w-full text-sm rounded-xl border border-gray-700/60 bg-gray-900/40 text-white px-3 py-2.5 focus:outline-none focus:ring-2 focus:ring-indigo-500/40 focus:border-indigo-500/40"
                    disabled={loading || isProcessing}
                  />
                </div>

                <div>
                  <label className="block text-xs font-medium text-gray-300 mb-2">
                    Event (optional)
                  </label>
                  <input
                    type="text"
                    value={event}
                    onChange={(e) => setEvent(e.target.value)}
                    placeholder="e.g., party congress"
                    className="w-full text-sm rounded-xl border border-gray-700/60 bg-gray-900/40 text-white px-3 py-2.5 focus:outline-none focus:ring-2 focus:ring-indigo-500/40 focus:border-indigo-500/40"
                    disabled={loading || isProcessing}
                  />
                </div>
              </div>

              {user && (
                <div className="pt-4 border-t border-gray-700/50">
                  <div className="text-xs text-gray-300 space-y-2">
                    <div className="flex justify-between">
                      <span className="text-gray-400">Max analyses:</span>
                      <span className="font-medium text-gray-200">{maxSpeeches}</span>
                    </div>

                    {hasUsageCount ? (
                      <>
                        <div className="flex justify-between">
                          <span className="text-gray-400">Used:</span>
                          <span className="font-medium text-gray-200">{usedCount}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-400">Remaining:</span>
                          <span
                            className={`font-medium ${
                              remainingAnalyses === 0 ? "text-rose-300" : "text-emerald-300"
                            }`}
                          >
                            {remainingAnalyses}
                          </span>
                        </div>
                        <div className="h-1.5 bg-gray-900/60 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-gradient-to-r from-emerald-500 to-teal-500 rounded-full"
                            style={{
                              width: `${Math.min(((usedCount / maxSpeeches) * 100) || 0, 100)}%`,
                            }}
                          />
                        </div>
                      </>
                    ) : (
                      <p className="text-xs text-gray-400">
                        Usage count not provided by backend. (Quota will not be enforced in UI.)
                      </p>
                    )}
                  </div>
                </div>
              )}
            </section>

            {/* Upload + Model */}
            <section className="lg:col-span-2 bg-gray-800/50 border border-gray-700/50 rounded-2xl p-5 space-y-5 shadow-sm">
              {/* Upload */}
              <div className="space-y-3">
                <div className="flex items-center gap-2">
                  <div className="h-8 w-8 rounded-lg bg-gray-900/50 border border-gray-700/50 flex items-center justify-center">
                    <Upload className="w-4 h-4 text-gray-300" />
                  </div>
                  <div>
                    <h2 className="text-sm font-semibold text-white">Speech File</h2>
                    <p className="text-xs text-gray-400">Video/audio will be transcribed by backend</p>
                  </div>
                </div>

                <div
                  className={`border-2 border-dashed rounded-2xl p-5 flex flex-col items-center justify-center text-center transition-colors ${
                    file
                      ? "border-emerald-500/40 bg-emerald-900/10"
                      : "border-gray-700/70 bg-gray-900/30 hover:border-indigo-500/50"
                  }`}
                >
                  <div className="mb-3">{getFileIcon()}</div>

                  {file ? (
                    <>
                      <p className="text-sm font-medium text-white mb-1">{file.name}</p>
                      <p className="text-xs text-gray-400 mb-3">
                        Size: {(file.size / (1024 * 1024)).toFixed(2)} MB • Format:{" "}
                        {getExt(file.name).toUpperCase()}
                      </p>
                      <div className="flex gap-2">
                        <label className="cursor-pointer text-xs px-3 py-2 rounded-xl bg-gray-900/50 border border-gray-700/50 text-gray-200 hover:bg-gray-900/70 transition-colors">
                          Change File
                          <input
                            type="file"
                            className="hidden"
                            accept=".mp4,.mov,.avi,.mkv,.webm,.flv,.wmv,.mp3,.wav,.m4a,.aac,.flac,.ogg,.txt,.md,.pdf,.doc,.docx,.json,.csv"
                            onChange={handleFileChange}
                            disabled={loading || isProcessing}
                          />
                        </label>
                        <button
                          type="button"
                          onClick={() => {
                            setFile(null);
                            setError("");
                            setUploadProgress(0);
                            setCurrentStatus("");
                            setIsProcessing(false);
                            setPollingError("");
                          }}
                          className="text-xs px-3 py-2 rounded-xl bg-rose-900/20 border border-rose-700/40 text-rose-200 hover:bg-rose-900/30 transition-colors"
                          disabled={loading || isProcessing}
                        >
                          Remove
                        </button>
                      </div>
                    </>
                  ) : (
                    <>
                      <p className="text-sm text-gray-300 mb-3">Click to browse and select a file</p>
                      <label className="cursor-pointer inline-flex items-center gap-2 px-4 py-2.5 rounded-xl bg-gradient-to-r from-indigo-600 to-purple-600 text-white text-xs font-semibold hover:opacity-90 transition-opacity">
                        <Upload className="w-4 h-4" />
                        Select File
                        <input
                          type="file"
                          className="hidden"
                          accept=".mp4,.mov,.avi,.mkv,.webm,.flv,.wmv,.mp3,.wav,.m4a,.aac,.flac,.ogg,.txt,.md,.pdf,.doc,.docx,.json,.csv"
                          onChange={handleFileChange}
                          disabled={loading || isProcessing}
                        />
                      </label>
                      <p className="text-xs text-gray-400 mt-3">
                        Max file size: {(maxSize / (1024 * 1024)).toFixed(0)} MB
                      </p>
                    </>
                  )}
                </div>
              </div>

              {/* Model */}
              <div className="space-y-3">
                <div className="flex items-center gap-2">
                  <div className="h-8 w-8 rounded-lg bg-blue-500/15 border border-blue-500/20 flex items-center justify-center">
                    <Brain className="w-4 h-4 text-blue-300" />
                  </div>
                  <div>
                    <h2 className="text-sm font-semibold text-white">Model</h2>
                    <p className="text-xs text-gray-400">Used by backend analysis pipeline</p>
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-xs font-medium text-gray-300 mb-2">
                      Provider
                    </label>
                    <div className="relative">
                      <select
                        value={llmProvider}
                        onChange={(e) => handleProviderChange(e.target.value)}
                        className="w-full text-sm rounded-xl border border-gray-700/60 px-3 py-2.5 bg-gray-900/40 text-white focus:outline-none focus:ring-2 focus:ring-blue-500/40 focus:border-blue-500/40 appearance-none pr-10 disabled:opacity-60"
                        disabled={loading || isProcessing}
                      >
                        <option value="openai">OpenAI</option>
                        <option value="groq">Groq</option>
                      </select>
                      <ChevronDown className="absolute right-3 top-3.5 w-4 h-4 text-gray-400 pointer-events-none" />
                    </div>
                  </div>

                  <div>
                    <label className="block text-xs font-medium text-gray-300 mb-2">
                      Model
                    </label>
                    <div className="relative">
                      <select
                        value={llmModel}
                        onChange={(e) => setLlmModel(e.target.value)}
                        className="w-full text-sm rounded-xl border border-gray-700/60 px-3 py-2.5 bg-gray-900/40 text-white focus:outline-none focus:ring-2 focus:ring-blue-500/40 focus:border-blue-500/40 appearance-none pr-10 disabled:opacity-60"
                        disabled={loading || isProcessing}
                      >
                        {(modelOptions[llmProvider] || []).map((m) => (
                          <option key={m.value} value={m.value}>
                            {m.label}
                          </option>
                        ))}
                      </select>
                      <ChevronDown className="absolute right-3 top-3.5 w-4 h-4 text-gray-400 pointer-events-none" />
                    </div>
                    <p className="text-xs text-gray-400 mt-1">
                      {modelOptions[llmProvider]?.find((m) => m.value === llmModel)?.description ||
                        "Select a model"}
                    </p>
                  </div>
                </div>
              </div>

              <div className="pt-4 border-t border-gray-700/50">
                <button
                  type="submit"
                  disabled={
                    loading ||
                    isProcessing ||
                    !file ||
                    !title.trim() ||
                    !speaker.trim() ||
                    (remainingAnalyses !== null && remainingAnalyses <= 0)
                  }
                  className="w-full inline-flex items-center justify-center px-5 py-3 rounded-2xl bg-gradient-to-r from-indigo-600 to-purple-600 text-white text-sm font-semibold hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 shadow-md hover:shadow-lg"
                >
                  {loading || isProcessing ? (
                    <>
                      <span className="mr-3 h-5 w-5 border-2 border-white/70 border-t-transparent rounded-full animate-spin" />
                      {loading ? "Uploading..." : "Processing..."}
                    </>
                  ) : (
                    <>
                      <Brain className="w-5 h-5 mr-2" />
                      {remainingAnalyses !== null && remainingAnalyses <= 0
                        ? "Analysis Limit Reached"
                        : "Start Analysis"}
                    </>
                  )}
                </button>

                <div className="mt-3 text-xs text-gray-400 space-y-1">
                  <p>Processing: Transcription → Segmentation → Ideology scoring → Key statements → Questions</p>
                </div>
              </div>
            </section>
          </div>
        </form>
      </div>
    </div>
  );
};

export default CreateProjectPage;