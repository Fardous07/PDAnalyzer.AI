import React, { useState } from "react";
import { NavLink, useNavigate } from "react-router-dom";
import {
  Search,
  Menu,
  X,
} from "lucide-react";

const AppHeader = ({ user, searchQuery, setSearchQuery, projects = [] }) => {
  const navigate = useNavigate();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  // Filter projects based on search query
  const filteredProjects = searchQuery
    ? projects.filter(project =>
        project.title?.toLowerCase().includes(searchQuery.toLowerCase()) ||
        project.speaker_name?.toLowerCase().includes(searchQuery.toLowerCase()) ||
        project.content?.toLowerCase().includes(searchQuery.toLowerCase())
      )
    : [];

  return (
    <header className="h-16 border-b border-gray-200 bg-white/90 backdrop-blur flex items-center justify-between px-4 lg:px-6 sticky top-0 z-50">
      {/* Left section: Logo and Brand */}
      <div className="flex items-center gap-4">
        <button
          className="md:hidden p-2"
          onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
        >
          {mobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
        </button>
        
        <div
          className="flex items-center gap-2 cursor-pointer"
          onClick={() => navigate("/")}
        >
          <img
            src="/favicon.svg"
            alt="Political Discourse Analyzer logo"
            className="h-9 w-9 rounded-lg"
          />
          <div className="flex flex-col">
            <span className="text-sm font-bold text-gray-900">
              PDAnalyzer.AI
            </span>
            <span className="text-xs text-gray-500">
              Political Discourse Analyzer
            </span>
          </div>
        </div>
      </div>

      {/* Mobile Menu */}
      {mobileMenuOpen && (
        <div className="absolute top-16 left-0 right-0 bg-white border-b border-gray-200 md:hidden py-3 px-4">
          <nav className="flex flex-col gap-3">
            <NavLink
              to="/"
              className={({ isActive }) =>
                `px-3 py-2 rounded-lg ${isActive ? "bg-indigo-50 text-indigo-600" : "text-gray-700 hover:bg-gray-50"}`
              }
              onClick={() => setMobileMenuOpen(false)}
            >
              Home
            </NavLink>
            <NavLink
              to="/comparison"
              className={({ isActive }) =>
                `px-3 py-2 rounded-lg ${isActive ? "bg-indigo-50 text-indigo-600" : "text-gray-700 hover:bg-gray-50"}`
              }
              onClick={() => setMobileMenuOpen(false)}
            >
              Comparison
            </NavLink>
            <NavLink
              to="/about"
              className={({ isActive }) =>
                `px-3 py-2 rounded-lg ${isActive ? "bg-indigo-50 text-indigo-600" : "text-gray-700 hover:bg-gray-50"}`
              }
              onClick={() => setMobileMenuOpen(false)}
            >
              About
            </NavLink>
            {!user && (
              <>
                <NavLink
                  to="/login"
                  className="px-3 py-2 text-gray-700 hover:bg-gray-50 rounded-lg"
                  onClick={() => setMobileMenuOpen(false)}
                >
                  Sign In
                </NavLink>
                <NavLink
                  to="/register"
                  className="px-3 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 text-center"
                  onClick={() => setMobileMenuOpen(false)}
                >
                  Get Started
                </NavLink>
              </>
            )}
          </nav>
        </div>
      )}

      {/* Center: Search Bar with Results */}
      <div className="flex-1 max-w-2xl mx-4">
        <div className="relative">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
            <input
              type="text"
              placeholder="Search speeches..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-10 pr-4 py-2 text-sm border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500"
            />
          </div>

          {/* Search Results Dropdown */}
          {searchQuery && filteredProjects.length > 0 && (
            <div className="absolute top-full left-0 right-0 mt-1 bg-white border border-gray-200 rounded-lg shadow-lg max-h-96 overflow-y-auto z-50">
              {filteredProjects.map((project) => (
                <div
                  key={project.id}
                  className="px-4 py-3 hover:bg-gray-50 cursor-pointer border-b border-gray-100 last:border-0"
                  onClick={() => {
                    navigate(`/analysis/${project.id}`);
                    setSearchQuery("");
                  }}
                >
                  <p className="text-sm font-medium text-gray-900">
                    {project.title || "Untitled speech"}
                  </p>
                  <p className="text-xs text-gray-500 truncate">
                    {project.speaker_name || "Unknown speaker"}
                  </p>
                  {project.content && (
                    <p className="text-xs text-gray-400 mt-1 line-clamp-2">
                      {project.content.substring(0, 100)}...
                    </p>
                  )}
                </div>
              ))}
            </div>
          )}

          {searchQuery && filteredProjects.length === 0 && (
            <div className="absolute top-full left-0 right-0 mt-1 bg-white border border-gray-200 rounded-lg shadow-lg p-4 z-50">
              <p className="text-sm text-gray-500">No speeches found</p>
            </div>
          )}
        </div>
      </div>

      {/* Right section: Desktop Navigation */}
      <nav className="hidden md:flex items-center gap-2 text-sm font-medium">
        <NavLink
          to="/"
          className={({ isActive }) =>
            `px-3 py-2 rounded-lg ${isActive ? "text-indigo-600 bg-indigo-50" : "text-gray-600 hover:text-gray-900 hover:bg-gray-100"}`
          }
        >
          Home
        </NavLink>
        <NavLink
          to="/comparison"
          className={({ isActive }) =>
            `px-3 py-2 rounded-lg ${isActive ? "text-indigo-600 bg-indigo-50" : "text-gray-600 hover:text-gray-900 hover:bg-gray-100"}`
          }
        >
          Comparison
        </NavLink>
        <NavLink
          to="/about"
          className={({ isActive }) =>
            `px-3 py-2 rounded-lg ${isActive ? "text-indigo-600 bg-indigo-50" : "text-gray-600 hover:text-gray-900 hover:bg-gray-100"}`
          }
        >
          About
        </NavLink>
        
        {!user && (
          <>
            <NavLink
              to="/login"
              className="px-3 py-2 text-gray-700 hover:text-gray-900"
            >
              Sign In
            </NavLink>
            <NavLink
              to="/register"
              className="px-3 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700"
            >
              Get Started
            </NavLink>
          </>
        )}
      </nav>
    </header>
  );
};

export default AppHeader;