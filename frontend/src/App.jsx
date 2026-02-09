// frontend/src/App.jsx
import React, { useEffect, useRef, useState, useCallback } from "react";
import { BrowserRouter, Routes, Route, useNavigate, useLocation, Navigate } from "react-router-dom";

import { AuthProvider, useAuth } from "./context/AuthContext";
import AuthGuard from "./components/AuthGuard";
import AppHeader from "./components/Header.jsx";
import Sidebar from "./components/Sidebar.jsx";
import UserProfile from "./components/UserProfile.jsx";

import HomePage from "./pages/HomePage.jsx";
import AboutPage from "./pages/AboutPage.jsx";
import ComparisonPage from "./pages/ComparisonPage.jsx";
import AnalysisPage from "./pages/AnalysisPage.jsx";
import CreateProjectPage from "./pages/CreateProjectPage.jsx";
import LoginPage from "./pages/LoginPage.jsx";
import RegisterPage from "./pages/RegisterPage.jsx";
import LandingPage from "./pages/LandingPage.jsx";
import TermsPage from "./pages/TermsPage";
import PrivacyPage from "./pages/PrivacyPage";

import { listSpeechesAll, deleteSpeech, updateSpeech, isAuthenticated } from "./services/api";

const AppShell = () => {
  const { user, loading, logout, authInitialized } = useAuth();

  const [projects, setProjects] = useState([]);
  const [searchQuery, setSearchQuery] = useState("");
  const [activeProjectId, setActiveProjectId] = useState(null);
  const [loadingProjects, setLoadingProjects] = useState(false);
  const [apiReady, setApiReady] = useState(false);

  const navigate = useNavigate();
  const location = useLocation();

  const refreshInFlightRef = useRef(false);

  useEffect(() => {
    const checkBackend = async () => {
      try {
        const base = (import.meta.env.VITE_API_BASE_URL || "http://localhost:8000").replace(/\/+$/, "");
        const response = await fetch(`${base}/health`);
        setApiReady(Boolean(response.ok));
      } catch (error) {
        console.error("Cannot connect to backend:", error);
        setApiReady(false);
      }
    };

    checkBackend();
  }, []);

  const refreshProjects = useCallback(async () => {
    if (!user || !authInitialized || !isAuthenticated() || !apiReady) return;
    if (refreshInFlightRef.current) return;

    refreshInFlightRef.current = true;
    setLoadingProjects(true);

    try {
      const speeches = await listSpeechesAll({ pageSize: 100 });
      const arr = Array.isArray(speeches) ? speeches : [];

      const mapped = arr.map((s) => ({
        ...s,
        title: s.title || "Untitled speech",
        speaker_name: s.speaker_name || s.speaker || "",
      }));

      setProjects(mapped);
      console.log("Projects loaded:", mapped.length);
    } catch (e) {
      console.error("Failed to load projects:", e?.message);

      const msg = String(e?.message || "");
      if (msg.includes("401") || msg.includes("Could not validate credentials") || msg.includes("Unauthorized")) {
        logout();
        navigate("/login", { state: { message: "Session expired. Please log in again." } });
      } else {
        setProjects([]);
      }
    } finally {
      setLoadingProjects(false);
      refreshInFlightRef.current = false;
    }
  }, [user, authInitialized, apiReady, logout, navigate]);

  useEffect(() => {
    const shouldLoadProjects = user && authInitialized && isAuthenticated() && apiReady;
    if (shouldLoadProjects) refreshProjects();
  }, [user, authInitialized, apiReady, refreshProjects]);

  useEffect(() => {
    const match = location.pathname.match(/^\/analysis\/(\d+)/);
    if (match && match[1]) {
      const idFromPath = Number(match[1]);
      if (!Number.isNaN(idFromPath) && idFromPath !== activeProjectId) {
        setActiveProjectId(idFromPath);
      }
    }
  }, [location.pathname, activeProjectId]);

  const handleSelectProject = (projectId) => {
    setActiveProjectId(projectId);
    if (projectId != null) navigate(`/analysis/${projectId}`);
  };

  const handleCreateProject = () => navigate("/projects/new");

  const handleRenameProject = async (projectId, newTitle) => {
    try {
      await updateSpeech(projectId, { title: newTitle });
      setProjects((prev) => prev.map((p) => (p.id === projectId ? { ...p, title: newTitle } : p)));
    } catch (e) {
      console.error("Failed to rename project:", e);
      alert("Failed to rename project");
    }
  };

  const handleDeleteProject = async (projectId) => {
    try {
      await deleteSpeech(projectId);
      setProjects((prev) => prev.filter((p) => p.id !== projectId));
      if (activeProjectId === projectId) {
        setActiveProjectId(null);
        navigate("/");
      }
    } catch (e) {
      console.error("Failed to delete project:", e);
      alert("Failed to delete project");
    }
  };

  const isPublicRoute =
    location.pathname === "/landing" ||
    location.pathname === "/login" ||
    location.pathname === "/register" ||
    location.pathname === "/terms" ||
    location.pathname === "/privacy";

  if (loading || !authInitialized) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50">
        <div className="flex flex-col items-center gap-4">
          <div className="w-12 h-12 border-4 border-indigo-600 border-t-transparent rounded-full animate-spin" />
          <p className="text-gray-600 font-medium">Loading application...</p>
        </div>
      </div>
    );
  }

  if (!apiReady && !isPublicRoute) {
    const base = (import.meta.env.VITE_API_BASE_URL || "http://localhost:8000").replace(/\/+$/, "");
    return (
      <div className="flex items-center justify-center min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50">
        <div className="max-w-md p-8 bg-white rounded-xl shadow-lg border border-red-200">
          <div className="text-center">
            <div className="w-16 h-16 bg-red-100 rounded-full flex items-center justify-center mx-auto mb-4">
              <svg className="w-8 h-8 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
            <h2 className="text-2xl font-bold text-gray-800 mb-2">Cannot Connect to Backend</h2>
            <p className="text-gray-600 mb-6">The application cannot reach the backend server. Please ensure:</p>
            <ul className="text-left text-gray-600 mb-6 space-y-2">
              <li className="flex items-start">
                <span className="text-green-500 mr-2">✓</span>
                Backend server is running at {base}
              </li>
              <li className="flex items-start">
                <span className="text-green-500 mr-2">✓</span>
                CORS is properly configured
              </li>
              <li className="flex items-start">
                <span className="text-green-500 mr-2">✓</span>
                No firewall is blocking the connection
              </li>
            </ul>
            <button onClick={() => window.location.reload()} className="px-6 py-3 bg-indigo-600 text-white font-semibold rounded-lg hover:bg-indigo-700 transition-colors" type="button">
              Retry Connection
            </button>
          </div>
        </div>
      </div>
    );
  }

  if (isPublicRoute) {
    return (
      <Routes>
        <Route path="/landing" element={user ? <Navigate to="/" replace /> : <LandingPage />} />
        <Route path="/login" element={user ? <Navigate to="/" replace /> : <LoginPage />} />
        <Route path="/register" element={user ? <Navigate to="/" replace /> : <RegisterPage />} />
        <Route path="/terms" element={<TermsPage />} />
        <Route path="/privacy" element={<PrivacyPage />} />
        <Route path="*" element={<Navigate to={user ? "/" : "/landing"} replace />} />
      </Routes>
    );
  }

  if (!user) return <Navigate to="/login" replace />;

  return (
    <div className="h-screen flex flex-col">
      <AppHeader user={user} searchQuery={searchQuery} setSearchQuery={setSearchQuery} projects={projects} />

      <div className="flex flex-1 overflow-hidden">
        <Sidebar
          projects={projects}
          activeProjectId={activeProjectId}
          onSelectProject={handleSelectProject}
          onCreateProject={handleCreateProject}
          onRenameProject={handleRenameProject}
          onDeleteProject={handleDeleteProject}
          loadingProjects={loadingProjects}
          user={user}
          refreshProjects={refreshProjects}
        />

        <main className="flex-1 overflow-y-auto bg-gray-50 px-4 py-4 lg:px-6 lg:py-6">
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/dashboard" element={<HomePage />} />

            <Route
              path="/projects/new"
              element={
                <AuthGuard>
                  <CreateProjectPage refreshProjects={refreshProjects} />
                </AuthGuard>
              }
            />

            <Route
              path="/comparison"
              element={
                <AuthGuard>
                  <ComparisonPage projects={projects} loading={loadingProjects} refreshProjects={refreshProjects} />
                </AuthGuard>
              }
            />

            <Route
              path="/analysis/:id"
              element={
                <AuthGuard>
                  <AnalysisPage refreshProjects={refreshProjects} />
                </AuthGuard>
              }
            />

            <Route path="/about" element={<AboutPage />} />

            <Route
              path="/profile"
              element={
                <AuthGuard>
                  <UserProfile userProjects={projects} />
                </AuthGuard>
              }
            />

            <Route path="/terms" element={<TermsPage />} />
            <Route path="/privacy" element={<PrivacyPage />} />

            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </main>
      </div>
    </div>
  );
};

const App = () => (
  <AuthProvider>
    <BrowserRouter>
      <AppShell />
    </BrowserRouter>
  </AuthProvider>
);

export default App;