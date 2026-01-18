import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { 
  User, Mail, Building, Calendar, 
  FileText, Shield, LogOut,
  Settings, CreditCard, HelpCircle,
  Edit2, Check, X, AlertCircle,
  Sparkles, HardDrive
} from 'lucide-react';
import { useAuth } from '../context/AuthContext';

const UserProfile = ({ userProjects = [] }) => { // Receive userProjects as prop
  const { user, logout, updateUser } = useAuth();
  const navigate = useNavigate();
  
  const [isEditing, setIsEditing] = useState(false);
  const [editForm, setEditForm] = useState({
    fullName: user?.full_name || '',
    organization: user?.organization || '',
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');

  if (!user) return null;

  const handleEditSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setSuccess('');

    try {
      const result = await updateUser({
        full_name: editForm.fullName,
        organization: editForm.organization,
      });

      if (result.success) {
        setSuccess('Profile updated successfully');
        setIsEditing(false);
      } else {
        setError(result.error);
      }
    } catch (err) {
      setError('Failed to update profile');
    } finally {
      setLoading(false);
    }
  };

  const formatDate = (dateString) => {
    if (!dateString) return 'N/A';
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'long',
      day: 'numeric'
    });
  };

  const formatFileSize = (bytes) => {
    if (bytes >= 1_000_000_000) {
      return `${(bytes / 1_000_000_000).toFixed(1)} GB`;
    } else if (bytes >= 1_000_000) {
      return `${(bytes / 1_000_000).toFixed(1)} MB`;
    } else if (bytes >= 1_000) {
      return `${(bytes / 1_000).toFixed(1)} KB`;
    }
    return `${bytes} bytes`;
  };

  const getTierColor = (tier) => {
    switch (tier) {
      case 'enterprise': return 'bg-purple-100 text-purple-800';
      case 'pro': return 'bg-blue-100 text-blue-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const getTierLabel = (tier) => {
    switch (tier) {
      case 'enterprise': return 'Enterprise';
      case 'pro': return 'Professional';
      default: return 'Free';
    }
  };

  // Calculate speech count from projects if available, otherwise use user.speech_count
  const speechCount = userProjects.length > 0 ? userProjects.length : (user.speech_count || 0);
  const remainingSpeeches = user.max_speeches - speechCount;
  const usagePercentage = (speechCount / user.max_speeches) * 100;

  return (
    <div className="min-h-screen bg-gray-50 py-8 px-4 sm:px-6 lg:px-8">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-2xl font-bold text-gray-900">Profile Settings</h1>
          <p className="text-gray-600">Manage your account and preferences</p>
        </div>

        {/* Error/Success Messages */}
        {error && (
          <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-xl flex items-start gap-3">
            <AlertCircle className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" />
            <p className="text-sm text-red-700">{error}</p>
          </div>
        )}

        {success && (
          <div className="mb-6 p-4 bg-emerald-50 border border-emerald-200 rounded-xl flex items-start gap-3">
            <Check className="w-5 h-5 text-emerald-600 flex-shrink-0 mt-0.5" />
            <p className="text-sm text-emerald-700">{success}</p>
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Left Column - Profile Info */}
          <div className="lg:col-span-2 space-y-6">
            {/* Profile Card */}
            <div className="bg-white rounded-2xl border border-gray-200 p-6">
              <div className="flex items-start justify-between mb-6">
                <div className="flex items-center gap-4">
                  <div className="h-16 w-16 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-500 flex items-center justify-center text-white text-2xl font-bold">
                    {user.username.charAt(0).toUpperCase()}
                  </div>
                  <div>
                    <h3 className="text-lg font-bold text-gray-900">
                      {user.full_name || user.username}
                    </h3>
                    <p className="text-sm text-gray-600">@{user.username}</p>
                    <div className="flex items-center gap-2 mt-1">
                      <span className={`px-2 py-1 rounded-full text-xs font-medium ${getTierColor(user.subscription_tier)}`}>
                        {getTierLabel(user.subscription_tier)} Plan
                      </span>
                      {user.subscription_tier === 'free' && (
                        <button
                          onClick={() => navigate('/upgrade')}
                          className="flex items-center gap-1 text-xs text-indigo-600 hover:text-indigo-700"
                        >
                          <Sparkles className="w-3 h-3" />
                          Upgrade
                        </button>
                      )}
                    </div>
                  </div>
                </div>
                
                {!isEditing ? (
                  <button
                    onClick={() => setIsEditing(true)}
                    className="flex items-center gap-2 px-3 py-1.5 text-sm text-gray-700 hover:text-gray-900 hover:bg-gray-100 rounded-lg"
                  >
                    <Edit2 className="w-4 h-4" />
                    Edit Profile
                  </button>
                ) : (
                  <div className="flex items-center gap-2">
                    <button
                      onClick={handleEditSubmit}
                      disabled={loading}
                      className="flex items-center gap-1 px-3 py-1.5 text-sm bg-indigo-600 text-white hover:bg-indigo-700 rounded-lg disabled:opacity-60"
                    >
                      <Check className="w-4 h-4" />
                      {loading ? 'Saving...' : 'Save'}
                    </button>
                    <button
                      onClick={() => {
                        setIsEditing(false);
                        setEditForm({
                          fullName: user.full_name || '',
                          organization: user.organization || '',
                        });
                      }}
                      className="flex items-center gap-1 px-3 py-1.5 text-sm text-gray-700 hover:text-gray-900 hover:bg-gray-100 rounded-lg"
                    >
                      <X className="w-4 h-4" />
                      Cancel
                    </button>
                  </div>
                )}
              </div>

              {isEditing ? (
                <form className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Full Name
                    </label>
                    <input
                      type="text"
                      value={editForm.fullName}
                      onChange={(e) => setEditForm({ ...editForm, fullName: e.target.value })}
                      className="w-full px-4 py-2.5 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                      placeholder="Your full name"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Organization
                    </label>
                    <input
                      type="text"
                      value={editForm.organization}
                      onChange={(e) => setEditForm({ ...editForm, organization: e.target.value })}
                      className="w-full px-4 py-2.5 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                      placeholder="Company, University, etc."
                    />
                  </div>
                </form>
              ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="flex items-center gap-3">
                    <Mail className="w-5 h-5 text-gray-400" />
                    <div>
                      <p className="text-sm text-gray-500">Email</p>
                      <p className="font-medium text-gray-900">{user.email}</p>
                    </div>
                  </div>
                  <div className="flex items-center gap-3">
                    <User className="w-5 h-5 text-gray-400" />
                    <div>
                      <p className="text-sm text-gray-500">Username</p>
                      <p className="font-medium text-gray-900">@{user.username}</p>
                    </div>
                  </div>
                  {user.organization && (
                    <div className="flex items-center gap-3">
                      <Building className="w-5 h-5 text-gray-400" />
                      <div>
                        <p className="text-sm text-gray-500">Organization</p>
                        <p className="font-medium text-gray-900">{user.organization}</p>
                      </div>
                    </div>
                  )}
                  <div className="flex items-center gap-3">
                    <Calendar className="w-5 h-5 text-gray-400" />
                    <div>
                      <p className="text-sm text-gray-500">Member since</p>
                      <p className="font-medium text-gray-900">{formatDate(user.created_at)}</p>
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Usage Stats - Updated with correct speech count */}
            <div className="bg-white rounded-2xl border border-gray-200 p-6">
              <h3 className="text-lg font-bold text-gray-900 mb-4">Usage Statistics</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <FileText className="w-5 h-5 text-indigo-500" />
                      <span className="text-sm text-gray-700">Speech Analyses</span>
                    </div>
                    <span className="text-lg font-bold text-gray-900">
                      {speechCount} / {user.max_speeches}
                    </span>
                  </div>
                  <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
                    <div 
                      className={`h-full rounded-full ${
                        usagePercentage > 90 ? 'bg-red-500' : 
                        usagePercentage > 70 ? 'bg-amber-500' : 'bg-emerald-500'
                      }`}
                      style={{ width: `${Math.min(usagePercentage, 100)}%` }}
                    />
                  </div>
                  <p className="text-xs text-gray-500">
                    {remainingSpeeches} analyses remaining
                  </p>
                </div>

                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <HardDrive className="w-5 h-5 text-emerald-500" />
                      <span className="text-sm text-gray-700">File Size Limit</span>
                    </div>
                    <span className="text-lg font-bold text-gray-900">
                      {formatFileSize(user.max_file_size)}
                    </span>
                  </div>
                  <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
                    <div 
                      className="h-full bg-gradient-to-r from-emerald-500 to-teal-500 rounded-full"
                      style={{ width: '100%' }}
                    />
                  </div>
                  <p className="text-xs text-gray-500">
                    Maximum file upload size
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* Right Column - Account Actions */}
          <div className="space-y-6">
            {/* Security */}
            <div className="bg-white rounded-2xl border border-gray-200 p-6">
              <h3 className="text-lg font-bold text-gray-900 mb-4">Security</h3>
              <div className="space-y-3">
                <button
                  onClick={() => navigate('/change-password')}
                  className="w-full flex items-center justify-between p-3 border border-gray-300 rounded-lg hover:bg-gray-50"
                >
                  <div className="flex items-center gap-3">
                    <Shield className="w-5 h-5 text-gray-400" />
                    <span className="text-sm text-gray-700">Change Password</span>
                  </div>
                </button>
                <div className="text-xs text-gray-500">
                  Last updated: {formatDate(user.updated_at)}
                </div>
              </div>
            </div>

            {/* Account Actions */}
            <div className="bg-white rounded-2xl border border-gray-200 p-6">
              <h3 className="text-lg font-bold text-gray-900 mb-4">Account</h3>
              <div className="space-y-2">
                <button 
                  onClick={() => navigate('/settings')}
                  className="w-full flex items-center gap-3 px-4 py-2.5 text-left text-gray-700 hover:bg-gray-50 rounded-lg transition-colors"
                >
                  <Settings className="w-4 h-4" />
                  <span className="flex-1">Settings</span>
                </button>
                
                {user.subscription_tier === 'free' && (
                  <button 
                    onClick={() => navigate('/upgrade')}
                    className="w-full flex items-center gap-3 px-4 py-2.5 text-left text-gray-700 hover:bg-gray-50 rounded-lg transition-colors"
                  >
                    <CreditCard className="w-4 h-4" />
                    <span className="flex-1">Upgrade to Pro</span>
                    <span className="px-2 py-0.5 text-xs bg-amber-100 text-amber-800 rounded">Get More</span>
                  </button>
                )}
                
                <button 
                  onClick={() => navigate('/help')}
                  className="w-full flex items-center gap-3 px-4 py-2.5 text-left text-gray-700 hover:bg-gray-50 rounded-lg transition-colors"
                >
                  <HelpCircle className="w-4 h-4" />
                  <span className="flex-1">Help & Support</span>
                </button>
                
                <button
                  onClick={() => {
                    logout();
                    navigate('/login');
                  }}
                  className="w-full flex items-center gap-3 px-4 py-2.5 text-left text-red-600 hover:bg-red-50 rounded-lg transition-colors"
                >
                  <LogOut className="w-4 h-4" />
                  <span className="flex-1">Logout</span>
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default UserProfile;