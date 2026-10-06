import React, { useRef } from 'react';

export default function ImageUploader({
  selectedFile,
  selectedSample,
  onFileSelect,
  onSampleSelect,
  onAnalyze,
  chkGradcam,
  setChkGradcam,
  chkIg,
  setChkIg,
  chkShap,
  setChkShap,
  isAnalyzing
}) {
  const fileInputRef = useRef(null);

  const samples = [
    { label: 'Adenocarcinoma Sample', cls: 'adenocarcinoma', file: '000114.png' },
    { label: 'Large Cell Sample', cls: 'large.cell.carcinoma', file: '000108.png' },
    { label: 'Normal Sample', cls: 'normal', file: '10.png' },
    { label: 'Squamous Cell Sample', cls: 'squamous.cell.carcinoma', file: '000119.png' }
  ];

  const handleDrop = (e) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      onFileSelect(e.dataTransfer.files[0]);
    }
  };

  return (
    <div className="panel glass-card">
      <div className="panel-header">
        <h3>1. Select or Upload CT Scan</h3>
      </div>

      {/* Upload Dropzone */}
      <div
        className="dropzone"
        onClick={() => fileInputRef.current?.click()}
        onDragOver={(e) => e.preventDefault()}
        onDrop={handleDrop}
      >
        <input
          type="file"
          ref={fileInputRef}
          accept="image/*"
          className="file-hidden"
          onChange={(e) => {
            if (e.target.files && e.target.files[0]) {
              onFileSelect(e.target.files[0]);
            }
          }}
        />
        <div className="dropzone-content">
          <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
            <polyline points="17 8 12 3 7 8"/>
            <line x1="12" y1="3" x2="12" y2="15"/>
          </svg>
          <p>{selectedFile ? `Selected: ${selectedFile.name}` : 'Drag & drop chest CT scan image here'}</p>
          <span className="subtext">or click to browse from your computer (256x256)</span>
        </div>
      </div>

      {/* Sample Selector */}
      <div className="sample-selector-section mt-3">
        <span className="subtext">Or select a test sample from dataset:</span>
        <div className="sample-chips">
          {samples.map((s, idx) => {
            const isSel = selectedSample?.class_name === s.cls && selectedSample?.filename === s.file;
            return (
              <button
                key={idx}
                className={`chip ${isSel ? 'selected' : ''}`}
                onClick={() => onSampleSelect({ class_name: s.cls, filename: s.file })}
              >
                {s.label}
              </button>
            );
          })}
        </div>
      </div>

      {/* XAI Explainability Options */}
      <div className="panel-header mt-4">
        <h3>2. XAI Explainability Options</h3>
      </div>
      <div className="xai-options-grid">
        <label className="xai-checkbox-card">
          <input
            type="checkbox"
            checked={chkGradcam}
            onChange={(e) => setChkGradcam(e.target.checked)}
          />
          <div className="opt-info">
            <strong>Grad-CAM</strong>
            <span>DenseNet Activation</span>
          </div>
        </label>

        <label className="xai-checkbox-card">
          <input
            type="checkbox"
            checked={chkIg}
            onChange={(e) => setChkIg(e.target.checked)}
          />
          <div className="opt-info">
            <strong>Integrated Gradients</strong>
            <span>Path Attributions</span>
          </div>
        </label>

        <label className="xai-checkbox-card">
          <input
            type="checkbox"
            checked={chkShap}
            onChange={(e) => setChkShap(e.target.checked)}
          />
          <div className="opt-info">
            <strong>SHAP</strong>
            <span>Shapley Values</span>
          </div>
        </label>
      </div>

      {/* Action Button */}
      <button className="primary-btn mt-4" onClick={onAnalyze} disabled={isAnalyzing}>
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <polygon points="5 3 19 12 5 21 5 3"/>
        </svg>
        {isAnalyzing ? 'Running Analysis & Generating Heatmaps...' : 'Run Diagnosis & Generate XAI Maps'}
      </button>
    </div>
  );
}
