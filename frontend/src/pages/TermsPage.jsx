import React from 'react';

const TermsPage = () => {
  return (
    <div className="min-h-screen bg-gray-50 py-12 px-4">
      <div className="max-w-3xl mx-auto bg-white rounded-xl shadow p-8">
        <h1 className="text-3xl font-bold mb-6">Terms of Service</h1>
        <div className="prose max-w-none">
          <p className="mb-4">
            Welcome to PDAnalyzer.AI. By using our service, you agree to these terms.
          </p>
          <h2 className="text-xl font-semibold mt-6 mb-3">1. Acceptance of Terms</h2>
          <p className="mb-4">
            By accessing and using PDAnalyzer.AI, you accept and agree to be bound by the terms 
            and provision of this agreement.
          </p>
          <h2 className="text-xl font-semibold mt-6 mb-3">2. Use License</h2>
          <p className="mb-4">
            Permission is granted to temporarily use PDAnalyzer.AI for personal, 
            non-commercial transitory viewing only.
          </p>
          <h2 className="text-xl font-semibold mt-6 mb-3">3. User Account</h2>
          <p className="mb-4">
            You are responsible for maintaining the confidentiality of your account and password.
          </p>
          <p className="text-sm text-gray-500 mt-8">
            Last updated: {new Date().toLocaleDateString()}
          </p>
        </div>
      </div>
    </div>
  );
};

export default TermsPage;