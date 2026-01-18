import React, { useMemo, useState } from "react";
import { NavLink, useNavigate, useLocation } from "react-router-dom";
import {
  Plus,
  MoreVertical,
  Edit2,
  Trash2,
  Loader2,
  Sparkles,
  Folder,
  ChevronDown,
  ChevronUp,
  User,
  Settings,
  LogOut,
  Shield,
  HelpCircle,
  CreditCard,
  FileText,
  MessageSquare,
  Mic,
  Volume2,
} from "lucide-react";

import { deleteSpeech, updateSpeech } from "../services/api";
import { useAuth } from "../context/AuthContext";

const Sidebar = ({ projects = [], loading = false, refreshProjects }) => {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();

  // Rename state (separate from menu state)
  const [editingId, setEditingId] = useState(null);
  const [editingTitle, setEditingTitle] = useState("");

  // Menu open state (kebab per row)
  const [menuOpenId, setMenuOpenId] = useState(null);

  const [busyId, setBusyId] = useState(null);
  const [confirmDeleteId, setConfirmDeleteId] = useState(null);

  // Keep your user panel behavior as-is
  const [showUserPanel, setShowUserPanel] = useState(false);

  const handleCreate = () => {
    navigate("/projects/new");
  };

  // Status dot helper (live/created vs not created)
  const getStatusColorClass = (project) => {
    const s = (project?.status || project?.analysis_status || project?.state || "").toLowerCase();

    // Adjust these values to match your backend
    if (["done", "completed", "ready", "created", "live"].includes(s)) return "bg-emerald-500";
    if (["processing", "running", "pending", "queued"].includes(s)) return "bg-amber-400";
    if (["error", "failed"].includes(s)) return "bg-red-500";

    // Not created / draft / unknown
    return "bg-gray-300";
  };

  const getProjectIcon = (project) => {
    // Choose icon based on file type or content
    const fileType = project?.file_type?.toLowerCase();
    
    if (fileType?.includes('audio') || fileType?.includes('mp3') || fileType?.includes('wav')) {
      return <Mic className="w-4 h-4" />;
    } else if (fileType?.includes('video') || fileType?.includes('mp4')) {
      return <Volume2 className="w-4 h-4" />;
    } else if (fileType?.includes('text') || fileType?.includes('txt') || fileType?.includes('pdf')) {
      return <FileText className="w-4 h-4" />;
    } else {
      // Default icon for unknown types
      return <MessageSquare className="w-4 h-4" />;
    }
  };

  const startRename = (project) => {
    setEditingId(project.id);
    setEditingTitle(project.title || "Untitled project");
    setMenuOpenId(null);
  };

  const saveRename = async (id) => {
    const next = editingTitle.trim();
    if (!next) {
      setEditingId(null);
      return;
    }
    try {
      setBusyId(id);
      await updateSpeech(id, { title: next });
      await refreshProjects?.();
    } catch (e) {
      console.error("Rename failed", e);
      alert("Failed to rename project");
    } finally {
      setBusyId(null);
      setEditingId(null);
    }
  };

  const handleDelete = (id) => {
    setConfirmDeleteId(id);
    setMenuOpenId(null);
  };

  const confirmDelete = async () => {
    if (!confirmDeleteId) return;

    try {
      setBusyId(confirmDeleteId);
      await deleteSpeech(confirmDeleteId);
      await refreshProjects?.();

      if (location.pathname === `/analysis/${confirmDeleteId}`) {
        navigate("/");
      }
    } catch (e) {
      console.error("Delete failed", e);
      alert("Failed to delete project");
    } finally {
      setBusyId(null);
      setConfirmDeleteId(null);
    }
  };

  // Active project id from URL
  let activeProjectId = null;
  const analysisMatch = location.pathname.match(/^\/analysis\/(\d+)/);
  if (analysisMatch) activeProjectId = parseInt(analysisMatch[1], 10);

  // Usage stats (unchanged concept, safer math)
  const stats = useMemo(() => {
    if (!user) return null;

    const usedSpeeches = user.speech_count || projects.length;
    const totalSpeeches = user.max_speeches || 50;
    const safeTotal = Math.max(Number(totalSpeeches) || 0, 0);
    const safeUsed = Math.max(Number(usedSpeeches) || 0, 0);
    const percentage = safeTotal > 0 ? (safeUsed / safeTotal) * 100 : 0;

    return {
      used: safeUsed,
      total: safeTotal,
      percentage,
      remaining: Math.max(safeTotal - safeUsed, 0),
    };
  }, [user, projects.length]);

  return (
    <aside className="w-72 border-r border-gray-200 bg-white flex flex-col h-screen relative">
      {/* Top Section: Create Project (KEPT) */}
      <div className="p-4 border-b border-gray-100">
        <button
          onClick={handleCreate}
          className="w-full inline-flex items-center justify-center gap-2 rounded-lg bg-indigo-600 hover:bg-indigo-700 text-white py-2.5 text-sm font-medium transition-colors"
        >
          <Plus className="w-4 h-4" />
          <span>Create Project</span>
        </button>

        {/* REMOVED: Advanced Analysis button section */}
      </div>

      {/* Project History Section */}
      <div className="flex-1 overflow-y-auto min-h-0">
        <div className="px-4 pt-3 pb-2 text-[11px] font-semibold text-gray-400 uppercase tracking-wide">
          Project History
        </div>

        <div className="px-3 pb-4">
          {loading && (
            <div className="flex items-center justify-center py-6 text-gray-400 text-sm">
              <Loader2 className="w-4 h-4 animate-spin mr-2" />
              Loading…
            </div>
          )}

          {!loading && projects.length === 0 && (
            <div className="px-2 py-6 text-xs text-gray-400 text-center">
              No projects yet. Click{" "}
              <span className="font-semibold text-gray-600">Create Project</span>{" "}
              to start.
            </div>
          )}

          {!loading && projects.length > 0 && (
            <div className="space-y-1">
              {projects.map((project) => {
                const isEditing = editingId === project.id;
                const isActive = project.id === activeProjectId;
                const isBusy = busyId === project.id;

                return (
                  <div
                    key={project.id}
                    className={`group flex items-center gap-2 rounded-lg px-2 py-2 ${
                      isActive
                        ? "bg-indigo-50 border border-indigo-100"
                        : "hover:bg-gray-50"
                    }`}
                  >
                    {/* Left: Link with Icon and Text */}
                    <NavLink
                      to={`/analysis/${project.id}`}
                      className="flex-1 flex items-center gap-3 min-w-0"
                      onClick={() => {
                        setMenuOpenId(null);
                      }}
                    >
                      {/* Project Icon (changed from Folder) */}
                      <div className="flex items-center justify-center w-8 h-8 rounded-md bg-gradient-to-br from-blue-50 to-indigo-50 text-indigo-600 shrink-0">
                        {getProjectIcon(project)}
                      </div>

                      <div className="flex-1 min-w-0">
                        {isEditing ? (
                          <input
                            autoFocus
                            value={editingTitle}
                            onChange={(e) => setEditingTitle(e.target.value)}
                            onBlur={() => saveRename(project.id)}
                            onKeyDown={(e) => {
                              if (e.key === "Enter") saveRename(project.id);
                              if (e.key === "Escape") setEditingId(null);
                            }}
                            className="w-full text-sm bg-white border border-indigo-200 rounded px-2 py-1 outline-none focus:ring-1 focus:ring-indigo-500"
                          />
                        ) : (
                          <>
                            <div className="text-sm font-medium text-gray-900 truncate">
                              {project.title || "Untitled project"}
                            </div>
                            <div className="text-xs text-gray-500 truncate">
                              {project.speaker_name || "Unknown speaker"}
                            </div>
                          </>
                        )}
                      </div>
                    </NavLink>

                    {/* Right: Status Dot (moved to right side), Busy, Menu */}
                    <div className="relative flex items-center gap-2 shrink-0">
                      {/* Status dot - now on the right side */}
                      <span
                        className={`inline-flex h-2.5 w-2.5 rounded-full ${getStatusColorClass(
                          project
                        )}`}
                        title={
                          project.status ||
                          project.analysis_status ||
                          project.state ||
                          "unknown"
                        }
                      />

                      {isBusy ? (
                        <Loader2 className="w-4 h-4 animate-spin text-gray-400" />
                      ) : (
                        <>
                          <button
                            type="button"
                            onClick={(e) => {
                              e.preventDefault();
                              e.stopPropagation();
                              setMenuOpenId((cur) =>
                                cur === project.id ? null : project.id
                              );
                            }}
                            className="rounded-md p-1 text-gray-400 hover:text-gray-600 hover:bg-gray-100"
                            aria-label="Project actions"
                          >
                            <MoreVertical className="w-4 h-4" />
                          </button>

                          {menuOpenId === project.id && (
                            <div
                              className="absolute right-0 top-full mt-1 w-36 bg-white rounded-md shadow-lg border border-gray-200 z-20 text-xs py-1"
                              onClick={(e) => {
                                e.preventDefault();
                                e.stopPropagation();
                              }}
                            >
                              <button
                                type="button"
                                onClick={() => startRename(project)}
                                className="w-full flex items-center gap-2 px-3 py-2 hover:bg-gray-50 text-left text-gray-700"
                              >
                                <Edit2 className="w-3.5 h-3.5" />
                                <span>Rename</span>
                              </button>

                              <button
                                type="button"
                                onClick={() => handleDelete(project.id)}
                                className="w-full flex items-center gap-2 px-3 py-2 hover:bg-red-50 text-left text-red-600"
                              >
                                <Trash2 className="w-3.5 h-3.5" />
                                <span>Delete</span>
                              </button>
                            </div>
                          )}
                        </>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>

      {/* ===========================
          User Profile Section
          =========================== */}
      {user && (
        <div className="border-t border-gray-200 bg-gradient-to-r from-slate-50 to-white">
          {/* User Profile Card (Always Visible) */}
          <div className="p-4">
            <div className="flex items-center gap-3 mb-4">
              <div className="h-10 w-10 rounded-full bg-gradient-to-br from-indigo-500 to-purple-500 flex items-center justify-center text-white text-sm font-bold">
                {user.username?.charAt(0).toUpperCase()}
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-sm font-semibold text-gray-900 truncate">
                  {user.full_name || user.username}
                </p>
                <p className="text-xs text-gray-500">Welcome, {user.username}</p>
              </div>
              <button
                onClick={() => setShowUserPanel(!showUserPanel)}
                className="p-1.5 rounded-lg hover:bg-gray-100 text-gray-500"
              >
                {showUserPanel ? (
                  <ChevronUp className="w-4 h-4" />
                ) : (
                  <ChevronDown className="w-4 h-4" />
                )}
              </button>
            </div>

            {/* Speech Usage Stats (Always Visible) */}
            {stats && (
              <div className="space-y-3 mb-4">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-gray-600 font-medium">Speech Usage</span>
                  <span className="font-semibold text-gray-900">
                    {stats.used} / {stats.total}
                  </span>
                </div>
                <div className="h-1.5 bg-gray-200 rounded-full overflow-hidden">
                  <div
                    className={`h-full rounded-full ${
                      stats.percentage > 90
                        ? "bg-red-500"
                        : stats.percentage > 70
                        ? "bg-amber-500"
                        : "bg-emerald-500"
                    }`}
                    style={{ width: `${Math.min(stats.percentage, 100)}%` }}
                  />
                </div>
                <p className="text-xs text-gray-500">
                  {stats.remaining} analyses remaining
                </p>
              </div>
            )}

            {/* Quick Actions (Always Visible) */}
            <div className="flex gap-2">
              <button
                onClick={() => navigate("/profile")}
                className="flex-1 flex items-center justify-center gap-2 px-3 py-2 text-xs bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-lg transition-colors"
              >
                <User className="w-3 h-3" />
                Profile
              </button>
              <button
                onClick={() => {
                  logout();
                  navigate("/login");
                }}
                className="flex-1 flex items-center justify-center gap-2 px-3 py-2 text-xs bg-red-50 hover:bg-red-100 text-red-600 rounded-lg transition-colors"
              >
                <LogOut className="w-3 h-3" />
                Logout
              </button>
            </div>
          </div>

          {/* Advanced User Panel (Appears Below) */}
          {showUserPanel && (
            <div className="border-t border-gray-200 bg-white px-4 py-4 animate-slideDown">
              <div className="space-y-4">
                {/* Account Info */}
                <div className="pb-3 border-b border-gray-100">
                  <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-3">
                    Account Details
                  </h4>
                  <div className="space-y-2">
                    <div className="flex items-center justify-between text-xs">
                      <span className="text-gray-600">Email</span>
                      <span className="font-medium text-gray-900 truncate ml-2">
                        {user.email}
                      </span>
                    </div>
                    <div className="flex items-center justify-between text-xs">
                      <span className="text-gray-600">Plan</span>
                      <span className="font-medium text-gray-900 capitalize">
                        {user.subscription_tier}
                      </span>
                    </div>
                    <div className="flex items-center justify-between text-xs">
                      <span className="text-gray-600">Member Since</span>
                      <span className="font-medium text-gray-900">
                        {new Date(user.created_at).toLocaleDateString()}
                      </span>
                    </div>
                  </div>
                </div>

                {/* Quick Navigation */}
                <div className="pb-3 border-b border-gray-100">
                  <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-3">
                    Quick Access
                  </h4>
                  <div className="space-y-2">
                    <button
                      onClick={() => {
                        navigate("/profile");
                        setShowUserPanel(false);
                      }}
                      className="w-full flex items-center gap-3 px-2 py-2 text-sm text-gray-700 hover:bg-gray-50 rounded-lg"
                    >
                      <User className="w-4 h-4 text-gray-400" />
                      <span className="flex-1 text-left">Profile Settings</span>
                    </button>
                    <button
                      onClick={() => {
                        navigate("/settings");
                        setShowUserPanel(false);
                      }}
                      className="w-full flex items-center gap-3 px-2 py-2 text-sm text-gray-700 hover:bg-gray-50 rounded-lg"
                    >
                      <Settings className="w-4 h-4 text-gray-400" />
                      <span className="flex-1 text-left">Preferences</span>
                    </button>
                    <button
                      onClick={() => {
                        navigate("/security");
                        setShowUserPanel(false);
                      }}
                      className="w-full flex items-center gap-3 px-2 py-2 text-sm text-gray-700 hover:bg-gray-50 rounded-lg"
                    >
                      <Shield className="w-4 h-4 text-gray-400" />
                      <span className="flex-1 text-left">Security</span>
                    </button>
                  </div>
                </div>

                {/* Account Actions */}
                <div>
                  <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-3">
                    Account Actions
                  </h4>
                  <div className="space-y-2">
                    {user.subscription_tier === "free" && (
                      <button
                        onClick={() => navigate("/upgrade")}
                        className="w-full flex items-center gap-3 px-2 py-2 text-sm text-blue-600 hover:bg-blue-50 rounded-lg"
                      >
                        <CreditCard className="w-4 h-4" />
                        <span className="flex-1 text-left">Upgrade to Pro</span>
                        <span className="px-2 py-0.5 text-xs bg-blue-100 text-blue-800 rounded">
                          Recommended
                        </span>
                      </button>
                    )}
                    <button
                      onClick={() => navigate("/help")}
                      className="w-full flex items-center gap-3 px-2 py-2 text-sm text-gray-700 hover:bg-gray-50 rounded-lg"
                    >
                      <HelpCircle className="w-4 h-4 text-gray-400" />
                      <span className="flex-1 text-left">Help & Support</span>
                    </button>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Delete Confirmation Modal */}
      {confirmDeleteId && (
        <div className="absolute inset-0 bg-black/50 flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-xl p-5 max-w-sm w-full">
            <div className="mb-4">
              <h3 className="text-lg font-semibold text-gray-900 mb-2">
                Delete Project
              </h3>
              <p className="text-sm text-gray-600">
                Are you sure you want to delete this project? All analysis data
                will be permanently removed.
              </p>
            </div>
            <div className="flex gap-3">
              <button
                onClick={() => setConfirmDeleteId(null)}
                className="flex-1 py-2 text-sm border border-gray-300 text-gray-700 hover:bg-gray-50 rounded-lg"
              >
                Cancel
              </button>
              <button
                onClick={confirmDelete}
                disabled={busyId === confirmDeleteId}
                className="flex-1 py-2 text-sm bg-red-600 text-white hover:bg-red-700 rounded-lg disabled:opacity-60"
              >
                {busyId === confirmDeleteId ? "Deleting..." : "Delete"}
              </button>
            </div>
          </div>
        </div>
      )}
    </aside>
  );
};

export default Sidebar;