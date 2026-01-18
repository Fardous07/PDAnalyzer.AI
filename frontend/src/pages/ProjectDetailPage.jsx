// frontend/src/pages/ProjectDetailPage.jsx
import React, { useEffect, useState } from "react";
import { useParams, Link } from "react-router-dom";
import {
  MessageSquare,
  Loader2,
  Brain,
  Zap,
  ArrowLeft,
  FileText,
  Info,
} from "lucide-react";
import { getSpeech, getOverview, generateQuestions } from "../services/api";

const ProjectDetailPage = () => {
  const { id } = useParams();

  const [speech, setSpeech] = useState(null);
  const [overview, setOverview] = useState(null);
  const [loading, setLoading] = useState(true);
  const [loadingQuestions, setLoadingQuestions] = useState(false);
  const [error, setError] = useState(null);

  const [questions, setQuestions] = useState([]);
  const [provider, setProvider] = useState("openai");
  const [model, setModel] = useState("gpt-4o-mini");

  const [questionTypes, setQuestionTypes] = useState([
    "clarification",
    "challenge",
    "follow_up",
    "accountability",
  ]);
  const [focusAreas, setFocusAreas] = useState([
    "ideology",
    "policy",
    "consistency",
  ]);
  const [tone, setTone] = useState("professional");
  const [numQuestions, setNumQuestions] = useState(6);

  useEffect(() => {
    const load = async () => {
      setLoading(true);
      setError(null);
      try {
        const [speechData, overviewData] = await Promise.all([
          getSpeech(id),
          getOverview(id),
        ]);
        setSpeech(speechData);
        setOverview(overviewData);
      } catch (e) {
        console.error("Failed to load project details", e);
        setError("Failed to load project details.");
      } finally {
        setLoading(false);
      }
    };
    load();
  }, [id]);

  const handleGenerateQuestions = async () => {
    setLoadingQuestions(true);
    try {
      const res = await generateQuestions(id, provider, model, {
        question_types: questionTypes.join(","),
        focus_areas: focusAreas.join(","),
        tone: tone,
        num_questions: numQuestions,
      });
      setQuestions(res.questions || []);
    } catch (e) {
      console.error("Failed to generate questions", e);
      alert("Failed to generate questions");
    } finally {
      setLoadingQuestions(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="flex flex-col items-center gap-3 text-slate-600">
          <div className="w-10 h-10 border-4 border-slate-900 border-t-transparent rounded-full animate-spin" />
          <p>Loading project…</p>
        </div>
      </div>
    );
  }

  if (error || !speech) {
    return (
      <div className="max-w-3xl mx-auto py-10 px-4">
        <Link
          to="/"
          className="inline-flex items-center text-sm text-slate-600 hover:text-slate-900 mb-4"
        >
          <ArrowLeft className="w-4 h-4 mr-1" />
          Back to projects
        </Link>
        <div className="bg-white rounded-2xl shadow p-6 text-center">
          <h2 className="text-xl font-semibold text-slate-900 mb-2">
            Project not found
          </h2>
          <p className="text-slate-600">
            {error || "The requested speech analysis is not available."}
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-5xl mx-auto py-8 px-4 space-y-6">
      {/* Top nav/back + header */}
      <div className="flex items-center justify-between">
        <Link
          to="/"
          className="inline-flex items-center text-sm text-slate-600 hover:text-slate-900"
        >
          <ArrowLeft className="w-4 h-4 mr-1" />
          Back to projects
        </Link>
        <Link
          to={`/analysis/${id}`}
          className="inline-flex items-center text-xs px-3 py-1.5 rounded-full bg-indigo-50 text-indigo-700 hover:bg-indigo-100"
        >
          <FileText className="w-3 h-3 mr-1" />
          View full analysis
        </Link>
      </div>

      {/* Project summary card */}
      <div className="bg-white rounded-2xl shadow p-5 flex flex-col gap-3">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <p className="text-xs uppercase tracking-wide text-slate-500 mb-1">
              Project • {speech.status === "done" ? "Analyzed" : "Processing"}
            </p>
            <h1 className="text-xl font-semibold text-slate-900">
              {speech.title || "Untitled speech"}
            </h1>
            <p className="text-xs text-slate-600 mt-1">
              {speech.speaker_name && <span>{speech.speaker_name}</span>}
              {speech.topic && <span> • {speech.topic}</span>}
              {speech.date && (
                <span> • {new Date(speech.date).toLocaleDateString()}</span>
              )}
            </p>
          </div>

          {overview && (
            <div className="flex flex-wrap gap-3 text-xs">
              <ScorePill
                label="Libertarian"
                color="emerald"
                value={overview.overall_lib_score || 0}
              />
              <ScorePill
                label="Authoritarian"
                color="rose"
                value={overview.overall_auth_score || 0}
              />
              <ScorePill
                label="Neutral"
                color="slate"
                value={overview.overall_neutral_score || 0}
              />
            </div>
          )}
        </div>

        {overview && (
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs mt-2">
            <SummaryCard
              label="Segments analyzed"
              value={overview.segment_count}
              helper="Text segments scored"
            />
            <SummaryCard
              label="Word count"
              value={overview.total_words}
              helper="Approx. transcript length"
            />
            <SummaryCard
              label="Complexity"
              value={overview.complexity}
              helper="Language complexity"
            />
            <SummaryCard
              label="Key statements"
              value={overview.key_segments_count}
              helper="Ideological highlights"
            />
          </div>
        )}
      </div>

      {/* Question generation card */}
      <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-4">
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-sm font-semibold text-gray-900 flex items-center gap-2">
            <MessageSquare className="w-4 h-4 text-blue-500" />
            Generate Interview Questions
          </h2>
          <div className="text-xs text-gray-500">
            Customize question generation
          </div>
        </div>

        {/* Question generation controls */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
          {/* Question Types */}
          <div className="space-y-2">
            <label className="block text-xs font-medium text-gray-700">
              Question Types
            </label>
            <div className="flex flex-wrap gap-1">
              {[
                "clarification",
                "challenge",
                "follow_up",
                "accountability",
                "context",
                "consistency",
              ].map((type) => (
                <label key={type} className="inline-flex items-center">
                  <input
                    type="checkbox"
                    checked={questionTypes.includes(type)}
                    onChange={(e) => {
                      if (e.target.checked) {
                        setQuestionTypes([...questionTypes, type]);
                      } else {
                        setQuestionTypes(
                          questionTypes.filter((t) => t !== type)
                        );
                      }
                    }}
                    className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                  />
                  <span className="ml-1 text-xs text-gray-700 capitalize">
                    {type.replace("_", " ")}
                  </span>
                </label>
              ))}
            </div>
          </div>

          {/* Focus Areas */}
          <div className="space-y-2">
            <label className="block text-xs font-medium text-gray-700">
              Focus Areas
            </label>
            <div className="flex flex-wrap gap-1">
              {[
                "ideology",
                "policy",
                "consistency",
                "implications",
                "rhetoric",
                "values",
              ].map((area) => (
                <label key={area} className="inline-flex items-center">
                  <input
                    type="checkbox"
                    checked={focusAreas.includes(area)}
                    onChange={(e) => {
                      if (e.target.checked) {
                        setFocusAreas([...focusAreas, area]);
                      } else {
                        setFocusAreas(
                          focusAreas.filter((a) => a !== area)
                        );
                      }
                    }}
                    className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                  />
                  <span className="ml-1 text-xs text-gray-700 capitalize">
                    {area}
                  </span>
                </label>
              ))}
            </div>
          </div>

          {/* Tone and Count */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <label className="text-xs font-medium text-gray-700">Tone</label>
              <select
                value={tone}
                onChange={(e) => setTone(e.target.value)}
                className="text-xs border border-gray-300 rounded px-2 py-1"
              >
                <option value="professional">Professional</option>
                <option value="challenging">Challenging</option>
                <option value="investigative">Investigative</option>
                <option value="balanced">Balanced</option>
                <option value="conversational">Conversational</option>
              </select>
            </div>

            <div className="flex items-center justify-between">
              <label className="text-xs font-medium text-gray-700">
                Number
              </label>
              <select
                value={numQuestions}
                onChange={(e) =>
                  setNumQuestions(parseInt(e.target.value, 10))
                }
                className="text-xs border border-gray-300 rounded px-2 py-1"
              >
                {[3, 4, 5, 6, 7, 8, 9, 10].map((n) => (
                  <option key={n} value={n}>
                    {n}
                  </option>
                ))}
              </select>
            </div>
          </div>
        </div>

        {/* LLM Settings */}
        <div className="flex items-center gap-2 mb-4">
          <select
            value={provider}
            onChange={(e) => setProvider(e.target.value)}
            className="border border-gray-300 rounded-md px-2 py-1 text-xs bg-white"
          >
            <option value="openai">OpenAI</option>
            <option value="groq">Groq</option>
          </select>
          <input
            value={model}
            onChange={(e) => setModel(e.target.value)}
            className="border border-gray-300 rounded-md px-2 py-1 text-xs w-32"
            placeholder="Model (gpt-4o-mini, etc.)"
          />
          <button
            onClick={handleGenerateQuestions}
            disabled={loadingQuestions}
            className="inline-flex items-center px-3 py-1.5 rounded-md bg-blue-600 text-white text-xs hover:bg-blue-700 disabled:opacity-50 ml-auto"
          >
            {loadingQuestions ? (
              <>
                <Loader2 className="w-3 h-3 animate-spin mr-1" />
                Generating…
              </>
            ) : (
              <>
                <Brain className="w-3 h-3 mr-1" />
                Generate Questions
              </>
            )}
          </button>
        </div>

        {/* Generated Questions */}
        {questions.length > 0 && (
          <div className="mt-4 border-t border-gray-100 pt-4">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-xs font-semibold text-gray-900">
                Generated Questions ({questions.length})
              </h3>
              <button
                onClick={() => setQuestions([])}
                className="text-xs text-gray-500 hover:text-gray-700"
              >
                Clear
              </button>
            </div>

            <div className="space-y-3 max-h-64 overflow-y-auto pr-1">
              {questions.map((q, idx) => (
                <div
                  key={idx}
                  className="border border-blue-100 bg-blue-50 rounded-lg px-3 py-2 text-xs"
                >
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2">
                      <span className="px-2 py-0.5 rounded-full bg-white text-blue-700 text-[10px] font-semibold">
                        Q{idx + 1}
                      </span>
                      <span
                        className={`px-2 py-0.5 rounded-full text-[10px] font-semibold capitalize ${
                          q.type === "challenge"
                            ? "bg-red-100 text-red-700"
                            : q.type === "clarification"
                            ? "bg-blue-100 text-blue-700"
                            : q.type === "follow_up"
                            ? "bg-purple-100 text-purple-700"
                            : q.type === "accountability"
                            ? "bg-amber-100 text-amber-700"
                            : "bg-emerald-100 text-emerald-700"
                        }`}
                      >
                        {q.type}
                      </span>
                      {q.priority === "high" && (
                        <span className="px-2 py-0.5 rounded-full bg-red-100 text-red-700 text-[10px] font-semibold">
                          High Priority
                        </span>
                      )}
                    </div>
                    <Zap className="w-3 h-3 text-yellow-500" />
                  </div>

                  <p className="font-medium text-gray-900 mb-2">
                    {q.question}
                  </p>

                  {q.context && (
                    <div className="mb-2">
                      <span className="text-[11px] font-medium text-gray-700">
                        Context:
                      </span>
                      <p className="text-[11px] text-gray-600">{q.context}</p>
                    </div>
                  )}

                  {q.based_on && (
                    <div className="text-[11px] text-gray-500">
                      <span className="font-medium">Based on:</span>{" "}
                      {q.based_on}
                    </div>
                  )}

                  {q.follow_up && (
                    <div className="mt-2 pt-2 border-t border-blue-200">
                      <span className="text-[11px] font-medium text-gray-700">
                        Follow-up:
                      </span>
                      <p className="text-[11px] text-gray-600 italic">
                        {q.follow_up}
                      </p>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {questions.length === 0 && !loadingQuestions && (
          <div className="flex items-center gap-2 text-[11px] text-slate-500 mt-2">
            <Info className="w-3 h-3" />
            No questions generated yet. Adjust the settings and click
            &quot;Generate Questions&quot;.
          </div>
        )}
      </div>
    </div>
  );
};

const ScorePill = ({ label, color, value }) => {
  const colorMap = {
    emerald: "bg-emerald-50 text-emerald-700",
    rose: "bg-rose-50 text-rose-700",
    slate: "bg-slate-50 text-slate-700",
  };
  return (
    <div className={`px-3 py-1 rounded-full ${colorMap[color] || ""}`}>
      <span className="text-[11px] font-medium mr-1">{label}</span>
      <span className="font-semibold text-xs">
        {typeof value === "number" ? value.toFixed(1) : "0.0"}%
      </span>
    </div>
  );
};

const SummaryCard = ({ label, value, helper }) => (
  <div className="bg-slate-50 rounded-xl p-3">
    <div className="text-xs text-slate-500 mb-1">{label}</div>
    <div className="text-lg font-semibold text-slate-900">
      {value ?? "–"}
    </div>
    <div className="text-[11px] text-slate-500 mt-1">{helper}</div>
  </div>
);

export default ProjectDetailPage;
