import React from 'react';

const PrivacyPage = () => {
  return (
    <div className="min-h-screen bg-gray-50 py-12 px-4">
      <div className="max-w-3xl mx-auto bg-white rounded-xl shadow p-8">
        <h1 className="text-3xl font-bold mb-6">Privacy Policy</h1>
        <div className="prose max-w-none">
          <p className="mb-4">
            Your privacy is important to us. This privacy policy explains what personal data 
            we collect and how we use it.
          </p>
          <h2 className="text-xl font-semibold mt-6 mb-3">1. Information We Collect</h2>
          <p className="mb-4">
            We collect information you provide directly to us, such as when you create an account, 
            upload content, or contact us for support.
          </p>
          <h2 className="text-xl font-semibold mt-6 mb-3">2. How We Use Your Information</h2>
          <p className="mb-4">
            We use the information we collect to provide, maintain, and improve our services, 
            to communicate with you, and to protect PDAnalyzer.AI and our users.
          </p>
          <h2 className="text-xl font-semibold mt-6 mb-3">3. Data Security</h2>
          <p className="mb-4">
            We implement appropriate technical and organizational measures to protect your 
            personal data against unauthorized access, alteration, or destruction.
          </p>
          <p className="text-sm text-gray-500 mt-8">
            Last updated: {new Date().toLocaleDateString()}
          </p>
        </div>
      </div>
    </div>
  );
};

export default PrivacyPage;