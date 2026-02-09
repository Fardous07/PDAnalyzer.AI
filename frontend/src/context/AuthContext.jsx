import React, { createContext, useState, useContext, useEffect, useCallback, useMemo } from "react";
import { authAPI, listSpeeches, getAccessToken, setAuthTokens, clearAuthTokens } from "../services/api";

export const AuthContext = createContext(null);

export const useAuth = () => {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within AuthProvider");
  return ctx;
};

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(() => {
    try {
      const stored = localStorage.getItem("user");
      return stored ? JSON.parse(stored) : null;
    } catch {
      return null;
    }
  });

  const [token, setToken] = useState(() => getAccessToken());
  const [loading, setLoading] = useState(true);
  const [userProjects, setUserProjects] = useState([]);
  const [loadingProjects, setLoadingProjects] = useState(false);
  const [authInitialized, setAuthInitialized] = useState(false);

  const hardRedirectToLogin = useCallback(() => {
    window.location.href = "/login";
  }, []);

  const logout = useCallback(async () => {
    setToken(null);
    setUser(null);
    setUserProjects([]);
    clearAuthTokens();

    try {
      await authAPI.logout();
    } catch (e) {
      console.warn("Backend logout failed (ignored):", e?.message || e);
    }

    hardRedirectToLogin();
  }, [hardRedirectToLogin]);

  const loadUserProjects = useCallback(async () => {
    const t = getAccessToken();
    if (!t) return;

    setLoadingProjects(true);
    try {
      const speeches = await listSpeeches();
      setUserProjects(Array.isArray(speeches) ? speeches : []);
    } catch (error) {
      const msg = String(error?.message || "");
      if (
        msg.includes("Authentication") ||
        msg.includes("Session") ||
        msg.includes("401") ||
        msg.includes("Could not validate credentials") ||
        msg.includes("Unauthorized")
      ) {
        await logout();
      } else {
        setUserProjects([]);
      }
    } finally {
      setLoadingProjects(false);
    }
  }, [logout]);

  useEffect(() => {
    let cancelled = false;

    const initAuth = async () => {
      try {
        const storedToken = getAccessToken();

        let storedUser = null;
        try {
          const u = localStorage.getItem("user");
          storedUser = u ? JSON.parse(u) : null;
        } catch {
          storedUser = null;
        }

        if (storedToken) {
          if (!cancelled) {
            setToken(storedToken);
            if (storedUser) setUser(storedUser);
          }

          try {
            const currentUser = await authAPI.getCurrentUser();
            const mergedUser = { ...(storedUser || {}), ...(currentUser || {}) };
            localStorage.setItem("user", JSON.stringify(mergedUser));

            if (!cancelled) {
              setUser(mergedUser);
              setToken(getAccessToken());
            }

            await loadUserProjects();
          } catch (error) {
            try {
              await authAPI.refreshToken();
              const refreshedUser = await authAPI.getCurrentUser();
              localStorage.setItem("user", JSON.stringify(refreshedUser));

              if (!cancelled) {
                setUser(refreshedUser);
                setToken(getAccessToken());
              }

              await loadUserProjects();
            } catch {
              if (!cancelled) await logout();
              return;
            }
          }
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
          setAuthInitialized(true);
        }
      }
    };

    initAuth();

    return () => {
      cancelled = true;
    };
  }, [loadUserProjects, logout]);

  const login = useCallback(
    async (credentials) => {
      try {
        const response = await authAPI.login(credentials);
        if (!response?.access_token || !response?.user) {
          throw new Error("Invalid login response from server");
        }

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
        return {
          success: false,
          error: error?.message || "Login failed. Please check your credentials.",
        };
      }
    },
    [loadUserProjects]
  );

  const register = useCallback(
    async (userData) => {
      try {
        const response = await authAPI.register(userData);

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

        return { success: true, user: response?.user || null, message: response?.message };
      } catch (error) {
        return {
          success: false,
          error: error?.message || "Registration failed. Please try again.",
        };
      }
    },
    [loadUserProjects]
  );

  const updateUser = useCallback(
    async (userData) => {
      const next = { ...(user || {}), ...(userData || {}) };
      localStorage.setItem("user", JSON.stringify(next));
      setUser(next);
      return next;
    },
    [user]
  );

  const refreshUser = useCallback(async () => {
    const t = getAccessToken();
    if (!t) return null;

    try {
      const userData = await authAPI.getCurrentUser();
      localStorage.setItem("user", JSON.stringify(userData));
      setUser(userData);
      return userData;
    } catch (error) {
      try {
        await authAPI.refreshToken();
        const userData = await authAPI.getCurrentUser();
        localStorage.setItem("user", JSON.stringify(userData));
        setUser(userData);
        setToken(getAccessToken());
        return userData;
      } catch {
        await logout();
        return null;
      }
    }
  }, [logout]);

  const refreshProjects = useCallback(async () => {
    await loadUserProjects();
  }, [loadUserProjects]);

  const value = useMemo(
    () => ({
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
    }),
    [
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
    ]
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};

export default AuthContext;