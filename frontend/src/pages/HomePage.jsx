// frontend/src/pages/HomePage.jsx
import React from "react";
import {
  Brain,
  LineChart,
  Sparkles,
  BarChart3,
  FileAudio,
  Video,
} from "lucide-react";

const HomePage = () => {
  return (
    <div className="h-full flex flex-col gap-6">
      {/* Top intro card */}
      <section className="bg-white rounded-2xl border border-gray-200 shadow-sm p-6 lg:p-8">
        <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-6">
          <div className="space-y-4 max-w-2xl">
            <div className="inline-flex items-center px-3 py-1 rounded-full bg-indigo-50 text-indigo-700 text-xs font-semibold">
              <Sparkles className="w-3 h-3 mr-1.5" />
              PDAnalyzer · AI Speech Insights
            </div>
            <h1 className="text-2xl lg:text-3xl font-bold text-gray-900">
              See what political speeches are really saying.
            </h1>
            <p className="text-sm lg:text-base text-gray-600 leading-relaxed">
              PDAnalyzer takes a speech, turns it into text, and highlights the
              most important political and ideological parts for you. Instead of
              reading the whole speech by hand, you get a clear overview in a
              few seconds.
            </p>
            <p className="text-sm lg:text-base text-gray-600 leading-relaxed">
              Upload a speech, watch or read it in the app, and explore key
              statements, themes, and follow-up questions that help you think
              more critically about what is being promised.
            </p>
          </div>

          <div className="flex-shrink-0 w-full lg:w-72">
            <div className="rounded-xl bg-gradient-to-br from-indigo-600 via-violet-600 to-fuchsia-500 text-white p-4 shadow-lg">
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                  <Brain className="w-5 h-5" />
                  <span className="text-sm font-semibold">
                    Analysis status
                  </span>
                </div>
                <span className="text-xs bg-white/15 px-2 py-0.5 rounded-full">
                  Online
                </span>
              </div>
              <div className="space-y-2 text-xs text-white/90">
                <div className="flex items-center justify-between">
                  <span>Transcription</span>
                  <span className="font-semibold">Whisper (OpenAI)</span>
                </div>
                <div className="flex items-center justify-between">
                  <span>Ideology scoring</span>
                  <span className="font-semibold">Custom model</span>
                </div>
                <div className="flex items-center justify-between">
                  <span>Question generator</span>
                  <span className="font-semibold">GPT / Groq</span>
                </div>
              </div>
              <div className="mt-3 h-1.5 rounded-full bg-white/20 overflow-hidden">
                <div className="h-full w-4/5 bg-white/90 rounded-full" />
              </div>
              <p className="mt-3 text-[11px] text-white/80 leading-snug">
                Upload a speech and the system will handle transcription,
                scoring, and questions automatically.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Pipeline overview */}
      <section className="grid grid-cols-1 lg:grid-cols-4 gap-4">
        {/* Step 1: Ingestion */}
        <div className="lg:col-span-1 bg-white border border-gray-200 rounded-2xl p-5">
          <div className="flex items-center gap-2 mb-3">
            <FileAudio className="w-5 h-5 text-indigo-500" />
            <h2 className="text-sm font-semibold text-gray-900">
              1. Add a speech
            </h2>
          </div>
          <p className="text-xs text-gray-600 mb-3">
            Start by adding a speech you want to analyze. You can upload a video,
            an audio file, or paste in prepared text.
          </p>
          <ul className="text-xs text-gray-600 list-disc list-inside space-y-1">
            <li>Supports common formats (MP4, MP3, WAV, MOV, TXT)</li>
            <li>Add speaker, party, date, and title</li>
            <li>Each speech is saved as its own project</li>
          </ul>
        </div>

        {/* Step 2: Transcription */}
        <div className="lg:col-span-1 bg-white border border-gray-200 rounded-2xl p-5">
          <div className="flex items-center gap-2 mb-3">
            <Video className="w-5 h-5 text-emerald-500" />
            <h2 className="text-sm font-semibold text-gray-900">
              2. Turn speech into text
            </h2>
          </div>
          <p className="text-xs text-gray-600 mb-3">
            The audio is automatically turned into a written transcript and
            split into smaller, readable pieces.
          </p>
          <ul className="text-xs text-gray-600 list-disc list-inside space-y-1">
            <li>Automatic transcription using AI</li>
            <li>Sentence-level segments for easier reading</li>
            <li>Full transcript available in the analysis view</li>
          </ul>
        </div>

        {/* Step 3: Ideology Scoring */}
        <div className="lg:col-span-1 bg-white border border-gray-200 rounded-2xl p-5">
          <div className="flex items-center gap-2 mb-3">
            <BarChart3 className="w-5 h-5 text-amber-500" />
            <h2 className="text-sm font-semibold text-gray-900">
              3. Score the content
            </h2>
          </div>
          <p className="text-xs text-gray-600 mb-3">
            Each part of the speech is scored on how libertarian, authoritarian,
            or neutral it sounds, based on a set of political themes.
          </p>
          <ul className="text-xs text-gray-600 list-disc list-inside space-y-1">
            <li>Ideology scores for each segment</li>
            <li>Counts of themes like democracy, law and order, etc.</li>
            <li>Overall ideological profile for the full speech</li>
          </ul>
        </div>

        {/* Step 4: Analysis & Questions */}
        <div className="lg:col-span-1 bg-white border border-gray-200 rounded-2xl p-5">
          <div className="flex items-center gap-2 mb-3">
            <LineChart className="w-5 h-5 text-rose-500" />
            <h2 className="text-sm font-semibold text-gray-900">
              4. Explore & ask questions
            </h2>
          </div>
          <p className="text-xs text-gray-600 mb-3">
            See where the speech is most ideological, compare different
            speeches, and get suggested follow-up questions you could ask.
          </p>
          <ul className="text-xs text-gray-600 list-disc list-inside space-y-1">
            <li>Highlighted key statements with labels</li>
            <li>Auto-generated journalist-style questions</li>
            <li>Comparison view across multiple speeches</li>
          </ul>
        </div>
      </section>
    </div>
  );
};

export default HomePage;
