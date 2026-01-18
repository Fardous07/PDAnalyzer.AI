import React from 'react';
import { Link } from 'react-router-dom';
import { Brain, LineChart, Shield, Users, ArrowRight, CheckCircle, BarChart3, FileText, Globe, Lock } from 'lucide-react';

const LandingPage = () => {
  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-50 to-white">
      {/* Navigation */}
      <nav className="px-6 py-4 flex justify-between items-center max-w-7xl mx-auto">
        <div className="flex items-center gap-2">
          <img src="/favicon.svg" alt="Logo" className="h-10 w-10" />
          <span className="text-xl font-bold text-slate-900">PDAnalyzer.AI</span>
        </div>
        <div className="flex gap-4">
          <Link to="/login" className="px-4 py-2 text-sm font-medium text-slate-700 hover:text-slate-900">
            Sign In
          </Link>
          <Link 
            to="/register" 
            className="px-4 py-2 bg-blue-600 text-white rounded-lg text-sm font-medium hover:bg-blue-700 transition-colors"
          >
            Get Started Free
          </Link>
        </div>
      </nav>

      {/* Hero Section */}
      <div className="max-w-7xl mx-auto px-6 py-16 md:py-24 text-center">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-blue-100 text-blue-700 text-sm font-medium mb-6">
          <Shield className="w-4 h-4" />
          Secure • Private • Research-Grade
        </div>
        
        <h1 className="text-4xl md:text-6xl font-bold text-slate-900 mb-6 leading-tight">
          Analyze Political Speech
          <span className="block text-blue-600">with AI Intelligence</span>
        </h1>
        
        <p className="text-xl text-slate-600 max-w-3xl mx-auto mb-10">
          Upload political speeches, get instant analysis of ideological positioning,
          track trends over time, and generate insightful questions—all in your private workspace.
        </p>
        
        <div className="flex flex-col sm:flex-row gap-4 justify-center mb-16">
          <Link 
            to="/register" 
            className="px-8 py-3 bg-blue-600 text-white rounded-xl text-lg font-semibold hover:bg-blue-700 transition-colors flex items-center justify-center gap-2"
          >
            Start Analyzing Free
            <ArrowRight className="w-5 h-5" />
          </Link>
          <Link 
            to="/login" 
            className="px-8 py-3 bg-white text-slate-700 border border-slate-300 rounded-xl text-lg font-semibold hover:bg-slate-50 transition-colors"
          >
            Sign In
          </Link>
        </div>

        {/* Trust Badges */}
        <div className="flex flex-wrap justify-center gap-8 mb-16 opacity-70">
          <div className="flex items-center gap-2 text-sm text-slate-500">
            <Lock className="w-4 h-4" />
            <span>End-to-end encryption</span>
          </div>
          <div className="flex items-center gap-2 text-sm text-slate-500">
            <Globe className="w-4 h-4" />
            <span>Global compliance</span>
          </div>
          <div className="flex items-center gap-2 text-sm text-slate-500">
            <CheckCircle className="w-4 h-4" />
            <span>No credit card required</span>
          </div>
        </div>
      </div>

      {/* Features Grid */}
      <div className="max-w-7xl mx-auto px-6 pb-16 md:pb-24">
        <div className="text-center mb-12">
          <h2 className="text-3xl font-bold text-slate-900 mb-4">Why Political Analysts Choose Us</h2>
          <p className="text-lg text-slate-600 max-w-2xl mx-auto">
            From journalists to researchers, we provide the tools to understand political discourse at scale.
          </p>
        </div>

        <div className="grid md:grid-cols-3 gap-8 mb-16">
          <div className="bg-white p-6 rounded-2xl shadow-lg border border-slate-200">
            <div className="w-12 h-12 rounded-xl bg-blue-100 flex items-center justify-center mb-4">
              <Brain className="w-6 h-6 text-blue-600" />
            </div>
            <h3 className="text-xl font-bold text-slate-900 mb-2">AI-Powered Analysis</h3>
            <p className="text-slate-600 mb-4">
              Uses advanced LLMs to detect libertarian vs authoritarian tendencies with MARPOR framework.
              Get insights beyond simple keyword matching.
            </p>
            <ul className="space-y-2 text-sm text-slate-600">
              <li className="flex items-center gap-2">
                <CheckCircle className="w-4 h-4 text-emerald-500" />
                <span>Advanced MARPOR classification</span>
              </li>
              <li className="flex items-center gap-2">
                <CheckCircle className="w-4 h-4 text-emerald-500" />
                <span>Context-aware scoring</span>
              </li>
              <li className="flex items-center gap-2">
                <CheckCircle className="w-4 h-4 text-emerald-500" />
                <span>Multi-model validation</span>
              </li>
            </ul>
          </div>
          
          <div className="bg-white p-6 rounded-2xl shadow-lg border border-slate-200">
            <div className="w-12 h-12 rounded-xl bg-emerald-100 flex items-center justify-center mb-4">
              <LineChart className="w-6 h-6 text-emerald-600" />
            </div>
            <h3 className="text-xl font-bold text-slate-900 mb-2">Trend Tracking</h3>
            <p className="text-slate-600 mb-4">
              Compare speeches over time, track ideological shifts, and visualize political evolution with comprehensive dashboards.
            </p>
            <ul className="space-y-2 text-sm text-slate-600">
              <li className="flex items-center gap-2">
                <CheckCircle className="w-4 h-4 text-emerald-500" />
                <span>Time-series analysis</span>
              </li>
              <li className="flex items-center gap-2">
                <CheckCircle className="w-4 h-4 text-emerald-500" />
                <span>Cross-speaker comparison</span>
              </li>
              <li className="flex items-center gap-2">
                <CheckCircle className="w-4 h-4 text-emerald-500" />
                <span>Exportable reports</span>
              </li>
            </ul>
          </div>
          
          <div className="bg-white p-6 rounded-2xl shadow-lg border border-slate-200">
            <div className="w-12 h-12 rounded-xl bg-purple-100 flex items-center justify-center mb-4">
              <Users className="w-6 h-6 text-purple-600" />
            </div>
            <h3 className="text-xl font-bold text-slate-900 mb-2">Secure Workspace</h3>
            <p className="text-slate-600 mb-4">
              Your data stays private. Each user gets isolated storage with role-based access control and enterprise-grade security.
            </p>
            <ul className="space-y-2 text-sm text-slate-600">
              <li className="flex items-center gap-2">
                <CheckCircle className="w-4 h-4 text-emerald-500" />
                <span>Individual data isolation</span>
              </li>
              <li className="flex items-center gap-2">
                <CheckCircle className="w-4 h-4 text-emerald-500" />
                <span>Public/private sharing</span>
              </li>
              <li className="flex items-center gap-2">
                <CheckCircle className="w-4 h-4 text-emerald-500" />
                <span>GDPR compliant</span>
              </li>
            </ul>
          </div>
        </div>

        {/* How It Works */}
        <div className="bg-gradient-to-r from-blue-50 to-indigo-50 rounded-2xl p-8 md:p-12">
          <h2 className="text-3xl font-bold text-slate-900 mb-8 text-center">How It Works</h2>
          
          <div className="grid md:grid-cols-4 gap-6">
            <div className="text-center">
              <div className="h-12 w-12 rounded-full bg-blue-600 text-white flex items-center justify-center text-lg font-bold mx-auto mb-4">
                1
              </div>
              <h4 className="font-bold text-slate-900 mb-2">Upload</h4>
              <p className="text-sm text-slate-600">
                Upload video, audio, or text files
              </p>
            </div>
            
            <div className="text-center">
              <div className="h-12 w-12 rounded-full bg-blue-600 text-white flex items-center justify-center text-lg font-bold mx-auto mb-4">
                2
              </div>
              <h4 className="font-bold text-slate-900 mb-2">Transcribe</h4>
              <p className="text-sm text-slate-600">
                AI automatically converts speech to text
              </p>
            </div>
            
            <div className="text-center">
              <div className="h-12 w-12 rounded-full bg-blue-600 text-white flex items-center justify-center text-lg font-bold mx-auto mb-4">
                3
              </div>
              <h4 className="font-bold text-slate-900 mb-2">Analyze</h4>
              <p className="text-sm text-slate-600">
                Get ideology scores and key insights
              </p>
            </div>
            
            <div className="text-center">
              <div className="h-12 w-12 rounded-full bg-blue-600 text-white flex items-center justify-center text-lg font-bold mx-auto mb-4">
                4
              </div>
              <h4 className="font-bold text-slate-900 mb-2">Visualize</h4>
              <p className="text-sm text-slate-600">
                Explore trends and generate reports
              </p>
            </div>
          </div>
        </div>

        {/* CTA Section */}
        <div className="mt-16 text-center">
          <h2 className="text-3xl font-bold text-slate-900 mb-4">
            Ready to analyze political discourse?
          </h2>
          <p className="text-lg text-slate-600 mb-8 max-w-2xl mx-auto">
            Join researchers, journalists, and analysts who use PDAnalyzer to understand political speech.
          </p>
          <Link 
            to="/register" 
            className="inline-flex items-center justify-center gap-2 px-8 py-3 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-xl text-lg font-semibold hover:from-blue-700 hover:to-purple-700 transition-all shadow-lg"
          >
            Start Your Free Trial
            <ArrowRight className="w-5 h-5" />
          </Link>
          <p className="text-sm text-slate-500 mt-4">
            No credit card required • 50 free speech analyses • Cancel anytime
          </p>
        </div>
      </div>
    </div>
  );
};

export default LandingPage;