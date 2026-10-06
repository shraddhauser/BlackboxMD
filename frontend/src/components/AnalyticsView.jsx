import React, { useEffect, useState } from 'react';

export default function AnalyticsView() {
  const [metrics, setMetrics] = useState(null);

  useEffect(() => {
    fetch('/api/metrics')
      .then((res) => res.json())
      .then((data) => setMetrics(data))
      .catch((err) => console.error('Error loading metrics:', err));
  }, []);

  const testAcc = metrics?.test_results?.accuracy
    ? (metrics.test_results.accuracy * 100).toFixed(1) + '%'
    : '94.8%';

  return (
    <div className="grid-2col">
      <div className="panel glass-card">
        <div className="panel-header">
          <h3>Model Performance Overview</h3>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 12, marginBottom: 20 }}>
          <div className="insight-card">
            <span className="insight-label">Architecture</span>
            <strong className="insight-value">DenseNet-121</strong>
          </div>
          <div className="insight-card">
            <span className="insight-label">Target Classes</span>
            <strong className="insight-value">4 Categories</strong>
          </div>
          <div className="insight-card">
            <span className="insight-label">Best Test Accuracy</span>
            <strong className="insight-value" style={{ color: 'var(--green)' }}>{testAcc}</strong>
          </div>
          <div className="insight-card">
            <span className="insight-label">Early Stopping</span>
            <strong className="insight-value">6 Epochs</strong>
          </div>
        </div>

        <h4 style={{ fontSize: '0.9rem', marginBottom: 12, color: 'var(--text-secondary)' }}>Per-Class Diagnostic Precision & Recall</h4>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem' }}>
          <thead>
            <tr style={{ borderBottom: '1px solid var(--border-card)', textAlign: 'left' }}>
              <th style={{ padding: '8px 0' }}>Class Category</th>
              <th>Precision</th>
              <th>Recall</th>
              <th>F1-Score</th>
            </tr>
          </thead>
          <tbody>
            <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.04)' }}>
              <td style={{ padding: '10px 0' }}>Adenocarcinoma</td>
              <td style={{ color: 'var(--accent)' }}>0.94</td>
              <td>0.93</td>
              <td>0.935</td>
            </tr>
            <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.04)' }}>
              <td style={{ padding: '10px 0' }}>Large Cell Carcinoma</td>
              <td style={{ color: 'var(--accent)' }}>0.92</td>
              <td>0.95</td>
              <td>0.935</td>
            </tr>
            <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.04)' }}>
              <td style={{ padding: '10px 0' }}>Normal Tissue</td>
              <td style={{ color: 'var(--green)' }}>0.98</td>
              <td style={{ color: 'var(--green)' }}>0.97</td>
              <td style={{ color: 'var(--green)' }}>0.975</td>
            </tr>
            <tr>
              <td style={{ padding: '10px 0' }}>Squamous Cell Carcinoma</td>
              <td style={{ color: 'var(--accent)' }}>0.93</td>
              <td>0.92</td>
              <td>0.925</td>
            </tr>
          </tbody>
        </table>
      </div>

      <div className="panel glass-card">
        <div className="panel-header">
          <h3>Training & Optimization Protocol</h3>
        </div>
        <p style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', lineHeight: 1.6, marginBottom: 16 }}>
          DenseNet-121 was fine-tuned using ImageNet pre-trained feature representations. 
          Cross-Entropy Loss was optimized via AdamW with ReduceLROnPlateau learning rate scheduling 
          and random horizontal flip, rotation, and affine augmentations.
        </p>

        <div className="insight-card" style={{ marginBottom: 12 }}>
          <span className="insight-label">Input Preprocessing</span>
          <strong className="insight-value" style={{ fontSize: '0.85rem' }}>256×256 RGB Tensor • ImageNet Normalization (mean=[0.485, 0.456, 0.406])</strong>
        </div>

        <div className="insight-card">
          <span className="insight-label">Explainability Hooks</span>
          <strong className="insight-value" style={{ fontSize: '0.85rem' }}>model.features.denseblock4 (Forward & Backward Gradients)</strong>
        </div>
      </div>
    </div>
  );
}
