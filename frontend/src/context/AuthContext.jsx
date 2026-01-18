// frontend/src/context/AuthContext.jsx
import React, { createContext, useState, useContext, useEffect, useCallback } from "react";
import {
  authAPI,
  listSpeeches,
  getAccessToken,
  setAuthTokens,
  clearAuthTokens,
} from "../services/api";

// Create the context
export const AuthContext = createContext(null);

// Hook for using the auth context
export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) throw new Error("useAuth must be used within AuthProvider");
  return context;
};

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);

  // token state is kept mainly for UI reactions; source of truth is localStorage
  const [token, setToken] = useState(getAccessToken());

  const [loading, setLoading] = useState(true);
  const [userProjects, setUserProjects] = useState([]);
  const [loadingProjects, setLoadingProjects] = useState(false);
  const [authInitialized, setAuthInitialized] = useState(false);

  const hardRedirectToLogin = () => {
    window.location.href = "/login";
  };

  const logout = useCallback(async () => {
    console.log("Logging out...");

    // Clear local state immediately
    setToken(null);
    setUser(null);
    setUserProjects([]);

    // Clear storage
    clearAuthTokens();

    // Best effort: tell backend (ignore failures)
    try {
      await authAPI.logout();
    } catch (e) {
      console.warn("Backend logout failed (ignored):", e?.message || e);
    }

    hardRedirectToLogin();
  }, []);

  const loadUserProjects = useCallback(async () => {
    const t = getAccessToken();
    if (!t) {
      console.log("No valid token available, skipping project load");
      return;
    }

    setLoadingProjects(true);
    try {
      const speeches = await listSpeeches();
      setUserProjects(Array.isArray(speeches) ? speeches : []);
      console.log("Projects loaded successfully:", Array.isArray(speeches) ? speeches.length : 0);
    } catch (error) {
      console.error("Failed to load user projects:", error?.message || error);

      const msg = String(error?.message || "");
      if (
        msg.includes("Authentication") ||
        msg.includes("Session") ||
        msg.includes("401") ||
        msg.includes("Could not validate credentials")
      ) {
        await logout();
      }
    } finally {
      setLoadingProjects(false);
    }
  }, [logout]);

  // Initialize auth from localStorage
  useEffect(() => {
    const initAuth = async () => {
      const storedToken = getAccessToken();
      const storedUser = localStorage.getItem("user");

      console.log("Initializing auth:", {
        storedToken: Boolean(storedToken),
        storedUser: Boolean(storedUser),
      });

      if (storedToken && storedUser) {
        try {
          const userData = JSON.parse(storedUser);

          // optimistic UI
          setToken(storedToken);
          setUser(userData);

          // validate token against backend
          try {
            console.log("Attempting to validate token...");
            const currentUser = await authAPI.getCurrentUser();
            console.log("Token validation successful:", currentUser?.email || "ok");

            const updatedUser = { ...userData, ...(currentUser || {}) };
            localStorage.setItem("user", JSON.stringify(updatedUser));
            setUser(updatedUser);

            await loadUserProjects();
          } catch (error) {
            console.error("Token validation failed:", error?.message || error);

            // If access token invalid, try refresh once
            try {
              console.log("Trying refresh token...");
              await authAPI.refreshToken();

              const refreshedUser = await authAPI.getCurrentUser();
              localStorage.setItem("user", JSON.stringify(refreshedUser));
              setUser(refreshedUser);
              setToken(getAccessToken());

              await loadUserProjects();
            } catch (refreshErr) {
              console.error("Refresh failed:", refreshErr?.message || refreshErr);
              await logout();
              return;
            }
          }
        } catch (error) {
          console.error("Auth initialization failed:", error);
          await logout();
          return;
        }
      } else {
        console.log("No stored auth found");
      }

      setLoading(false);
      setAuthInitialized(true);
    };

    initAuth();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [loadUserProjects, logout]);

  const login = async (credentials) => {
    try {
      console.log("Attempting login with credentials:", {
        email: credentials?.email,
        password: credentials?.password ? "***" : "missing",
      });

      const response = await authAPI.login(credentials);

      if (!response?.access_token || !response?.user) {
        console.error("Invalid login response structure:", response);
        throw new Error("Invalid login response from server");
      }

      // IMPORTANT: store both access + refresh token
      setAuthTokens({
        accessToken: response.access_token,
        refreshToken: response.refresh_token || null,
        user: response.user,
      });

      setToken(response.access_token);
      setUser(response.user);

      await loadUserProjects();

      return { success: true, user: response.user };
    } catch (error) {
      console.error("Login failed:", error);
      return {
        success: false,
        error: error?.message || "Login failed. Please check your credentials.",
      };
    }
  };

  const register = async (userData) => {
    try {
      console.log("Attempting registration...");
      const response = await authAPI.register(userData);

      // If backend auto-logins after register
      if (response?.access_token && response?.user) {
        setAuthTokens({
          accessToken: response.access_token,
          refreshToken: response.refresh_token || null,
          user: response.user,
        });

        setToken(response.access_token);
        setUser(response.user);

        await loadUserProjects();

        return { success: true, user: response.user };
      }

      // Fallback (no auto-login)
      return { success: true, user: response?.user || null };
    } catch (error) {
      console.error("Registration failed:", error);
      return {
        success: false,
        error: error?.message || "Registration failed. Please try again.",
      };
    }
  };

  const updateUser = async (userData) => {
    try {
      return await new Promise((resolve) => {
        setTimeout(() => {
          const newUser = { ...(user || {}), ...(userData || {}) };
          localStorage.setItem("user", JSON.stringify(newUser));
          setUser(newUser);
          resolve(newUser);
        }, 300);
      });
    } catch (error) {
      console.error("Update failed:", error);
      throw error;
    }
  };

  const refreshUser = async () => {
    const t = getAccessToken();
    if (!t) return null;

    try {
      const userData = await authAPI.getCurrentUser();
      localStorage.setItem("user", JSON.stringify(userData));
      setUser(userData);
      return userData;
    } catch (error) {
      console.error("Failed to refresh user:", error?.message || error);

      // attempt refresh once, then retry
      try {
        await authAPI.refreshToken();
        const userData = await authAPI.getCurrentUser();
        localStorage.setItem("user", JSON.stringify(userData));
        setUser(userData);
        setToken(getAccessToken());
        return userData;
      } catch (e) {
        await logout();
        return null;
      }
    }
  };

  const refreshProjects = async () => {
    await loadUserProjects();
  };

  const value = {
    user,
    token,
    loading,
    userProjects,
    loadingProjects,
    authInitialized,

    login,
    register,
    logout,

    updateUser,
    refreshUser,
    refreshProjects,

    isAuthenticated: Boolean(getAccessToken()),
    isAdmin: user?.role === "admin",
    isPro: user?.subscription_tier === "pro" || user?.subscription_tier === "enterprise",
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};
