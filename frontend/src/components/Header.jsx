import React from 'react';

export default function Header({ status }) {
  const isOnline = status?.status === 'online';

  return (
    <header className="navbar">
      <div className="brand">
        <div className="brand-logo">
          <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/>
          </svg>
        </div>
        <div className="brand-text">
          <h1>BLACKBOX <span className="accent-text">MD</span></h1>
          <p className="tagline">Explainable Pulmonary Oncology Diagnostic System • DenseNet-121</p>
        </div>
      </div>

      <div className="header-status">
        <div className="status-badge">
          <span className={`pulse-dot ${isOnline ? 'green' : ''}`}></span>
          <span>{isOnline ? 'DenseNet-121 Ready' : 'Connecting to Backend...'}</span>
        </div>

        <div className="device-badge">
          <span>Device:</span>
          <strong>{status?.device ? status.device.toUpperCase() : 'CPU'}</strong>
        </div>

        <div className="model-badge">
          <span>Model:</span>
          <strong className="accent-text">DenseNet-121 (Multi-Class)</strong>
        </div>
      </div>
    </header>
  );
}
