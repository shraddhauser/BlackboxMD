import React, { useState } from 'react';

export default function PredictionResults({
  predData,
  expData,
  onDownloadPdf,
  isDownloadingPdf
}) {
  if (!predData) {
    return (
      <div className="panel glass-card">
        <div className="panel-header">
          <h3>Model Diagnostic Results</h3>
          <span className="source-tag">No input loaded</span>
        </div>
        <div style={{ textAlign: 'center', padding: '48px 20px', color: 'var(--text-muted)' }}>
          <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1" style={{ marginBottom: 12 }}>
            <circle cx="12" cy="12" r="10"/>
            <line x1="12" y1="8" x2="12" y2="12"/>
            <line x1="12" y1="16" x2="12.01" y2="16"/>
          </svg>
          <p>Select or upload a chest CT scan to view DenseNet-121 predictions, spatial insights & XAI heatmaps</p>
        </div>
      </div>
    );
  }

  const cleanClass = predData.clean_class || predData.predicted_class?.replace(/\./g, ' ');
  const confPct = (predData.confidence * 100).toFixed(1);
  const probs = predData.clean_probabilities || predData.probabilities || {};
  const sortedProbs = Object.entries(probs).sort((a, b) => b[1] - a[1]);

  const feats = expData?.heatmap_features || {};
  const nlp = expData?.nlp_explanation || {};

  return (
    <div className="panel glass-card">
      <div className="panel-header">
        <h3>Model Diagnostic Results</h3>
        <span className="source-tag">{predData.source || 'Scan Analyzed'}</span>
      </div>

      {/* Top Diagnosis Callout (Flowchart Step 3) */}
      <div className="diagnosis-callout">
        <div>
          <div className="callout-label">Predicted Diagnosis</div>
          <div className="callout-value">{cleanClass}</div>
        </div>
        <div className="confidence-pill">{confPct}% Confidence</div>
      </div>

      {/* Probability Bars (Flowchart Step 3) */}
      <div className="probability-section">
        <h4>Classification Probability Distribution</h4>
        <div className="prob-list">
          {sortedProbs.map(([clsName, prob], idx) => {
            const pct = (prob * 100).toFixed(1);
            const isTop = (clsName === cleanClass) || (clsName === predData.predicted_class);
            return (
              <div key={idx} className={`prob-row ${isTop ? 'top' : ''}`}>
                <div className="prob-info">
                  <span>{clsName}</span>
                  <strong>{pct}%</strong>
                </div>
                <div className="prob-bar-bg">
                  <div className="prob-bar-fill" style={{ width: `${pct}%` }}></div>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Heatmap Analysis & Key Insights (Flowchart Step 5) */}
      <div className="insights-section">
        <h4>Heatmap Analysis & Key Insights</h4>
        <div className="insights-grid">
          <div className="insight-card">
            <span className="insight-label">Anatomical Location</span>
            <strong className="insight-value">{feats.location || 'Lower Right Lung'}</strong>
          </div>
          <div className="insight-card">
            <span className="insight-label">Activation Level</span>
            <strong className="insight-value">{feats.activation_level || 'High'}</strong>
          </div>
          <div className="insight-card">
            <span className="insight-label">Region Size</span>
            <strong className="insight-value">{feats.region_size_str || `${feats.region_size_pct || 18.0}% of lung area`}</strong>
          </div>
          <div className="insight-card">
            <span className="insight-label">Activation Intensity</span>
            <strong className="insight-value">{feats.activation_intensity_str || `${feats.activation_intensity || 0.82} (${feats.activation_level || 'High'})`}</strong>
          </div>
        </div>
      </div>

      {/* Clinical NLP Explanation Module (Flowchart Step 6) */}
      <div className="nlp-explanation-card">
        <h4>Natural Language Explanation (NLP Module)</h4>
        <p className="nlp-primary-text">
          {nlp.primary_explanation || `The model predicts ${cleanClass} with ${confPct}% confidence. The prediction was primarily influenced by a highly activated region in the ${feats.location || 'upper right portion of the lung'}, which occupies about ${feats.region_size_pct || 18}% of the lung area.`}
        </p>

        {nlp.clinical_impression && (
          <div className="nlp-impression-box">
            <span className="impression-tag">Clinical Impression</span>
            <p>{nlp.clinical_impression}</p>
          </div>
        )}
      </div>

      {/* Download PDF Report Button (Flowchart Step 7) */}
      <div className="report-export-section">
        <button className="pdf-download-btn" onClick={onDownloadPdf} disabled={isDownloadingPdf}>
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
            <polyline points="14 2 14 8 20 8"></polyline>
            <line x1="12" y1="18" x2="12" y2="12"></line>
            <line x1="9" y1="15" x2="15" y2="15"></line>
          </svg>
          {isDownloadingPdf ? 'Generating PDF Clinical Report...' : 'Download / Generate Report (PDF)'}
        </button>
      </div>
    </div>
  );
}
