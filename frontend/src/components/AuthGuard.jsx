// frontend/src/components/AuthGuard.jsx
import React, { useEffect, useRef, useState } from "react";
import { Navigate, useLocation } from "react-router-dom";
import { useAuth } from "../context/AuthContext";

/**
 * AuthGuard
 * - Protects routes unless authenticated.
 * - If token exists but user is missing, tries refreshUser() once.
 *
 * Props:
 * - requireRole: string | string[]  (e.g. "admin" or ["admin","editor"])
 * - requirePro: boolean
 */
const AuthGuard = ({ children, requireRole, requirePro = false }) => {
  const location = useLocation();

  const {
    user,
    loading,
    authInitialized,
    refreshUser,
    isAuthenticated, // computed from localStorage token in your AuthContext
    isPro,
  } = useAuth();

  const rehydrateAttemptedRef = useRef(false);
  const [rehydrating, setRehydrating] = useState(false);

  useEffect(() => {
    let alive = true;

    const run = async () => {
      // Don't do anything until auth provider finished initializing
      if (!authInitialized) return;
      if (loading) return;

      const hasToken = Boolean(isAuthenticated);

      // Token exists but user is missing => fetch user once
      if (hasToken && !user && !rehydrateAttemptedRef.current && typeof refreshUser === "function") {
        rehydrateAttemptedRef.current = true;
        if (alive) setRehydrating(true);
        try {
          await refreshUser();
        } finally {
          if (alive) setRehydrating(false);
        }
      }
    };

    run();
    return () => {
      alive = false;
    };
  }, [authInitialized, loading, isAuthenticated, user, refreshUser]);

  // Show loader while auth is initializing or while we try to rehydrate user once
  if (!authInitialized || loading || rehydrating) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="flex flex-col items-center gap-4">
          <div className="w-8 h-8 border-2 border-indigo-600 border-t-transparent rounded-full animate-spin" />
          <p className="text-gray-600">Loading...</p>
        </div>
      </div>
    );
  }

  // After init + possible rehydrate attempt: still no user => go login
  if (!user) {
    const from = location.pathname + location.search;
    return <Navigate to="/login" replace state={{ from }} />;
  }

  // Optional: role guard
  if (requireRole) {
    const allowed = Array.isArray(requireRole) ? requireRole : [requireRole];
    if (!allowed.includes(user?.role)) {
      return <Navigate to="/dashboard" replace />;
    }
  }

  // Optional: subscription guard
  if (requirePro && !isPro) {
    return <Navigate to="/dashboard" replace />;
  }

  return children;
};

export default AuthGuard;