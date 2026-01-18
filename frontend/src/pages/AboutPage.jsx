import React from "react";

const AboutPage = () => {
  return (
    <div className="space-y-4">
      {/* Intro section */}
      <section className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6">
        <h1 className="text-2xl font-bold text-gray-900 mb-2">
          About PDAnalyzer
        </h1>
        <p className="text-sm text-gray-600 max-w-3xl leading-relaxed">
          PDAnalyzer (Political Discourse Analyzer) is an AI tool for exploring
          political speeches. It helps you see which parts of a speech lean more
          toward libertarian or authoritarian thinking, and highlights the key
          promises, themes, and tensions in the speaker’s message.
        </p>
      </section>

      {/* How it works section */}
      <section className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6 text-sm text-gray-700 space-y-3 leading-relaxed">
        <p>
          When you upload a speech, PDAnalyzer turns the audio or video into
          text, breaks it into smaller pieces, and scores each part using a
          MARPOR-inspired framework. This looks at ideas like democracy and
          participation, law and order, centralization of power, social
          freedoms, economic control, and more.
        </p>
        <p>
          Behind the scenes, the system uses a FastAPI backend with PostgreSQL
          for storing speeches and analysis results. Transcription is handled by
          OpenAI Whisper, and large language models (OpenAI / Groq) are used for
          ideological scoring and question generation.
        </p>
        <p>
          The interface you are using is built as a project workspace: on the
          left you see your list of speeches, and in the main area you can watch
          or read each speech with synchronized media, transcript, ideological
          scores, key statements, and suggested follow-up questions. The goal is
          to make it easier to understand not just what was said, but what it
          actually means.
        </p>
      </section>
    </div>
  );
};

export default AboutPage;
