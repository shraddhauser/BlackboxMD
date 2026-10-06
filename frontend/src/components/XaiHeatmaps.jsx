import React, { useState } from 'react';

export default function XaiHeatmaps({ origImage, expData }) {
  const [opacity, setOpacity] = useState(0.45);

  const explanations = expData?.explanations || {};
  const placeholder = '/static/placeholder.png';

  const origSrc = origImage || expData?.original_image_base64 || placeholder;
  const gradcamSrc = explanations.gradcam?.overlay_base64 || placeholder;
  const igSrc = explanations.integrated_gradients?.overlay_base64 || placeholder;
  const shapSrc = explanations.shap?.overlay_base64 || placeholder;

  return (
    <div className="panel glass-card mt-4">
      <div className="panel-header flex-between">
        <div>
          <h3>Comparative Explainable AI (XAI) Heatmaps</h3>
          <p className="subtext">Side-by-side visualization of DenseNet-121 feature attribution maps</p>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <label style={{ fontSize: '0.82rem', color: 'var(--text-secondary)' }}>Overlay Opacity:</label>
          <input
            type="range"
            min="0.1"
            max="1.0"
            step="0.05"
            value={opacity}
            onChange={(e) => setOpacity(parseFloat(e.target.value))}
            style={{ width: 100, accentColor: 'var(--accent)' }}
          />
          <span style={{ fontSize: '0.82rem', fontWeight: 'bold', color: 'var(--accent)', minWidth: 36 }}>
            {Math.round(opacity * 100)}%
          </span>
        </div>
      </div>

      <div className="xai-grid">
        {/* Card 1: Original CT Scan */}
        <div className="xai-card">
          <div className="card-title">Original CT Scan</div>
          <div className="img-wrapper">
            <img src={origSrc} alt="Original CT Scan" />
          </div>
          <div className="card-meta">Raw input scan (256x256)</div>
        </div>

        {/* Card 2: Grad-CAM */}
        <div className="xai-card">
          <div className="card-title">Grad-CAM</div>
          <div className="img-wrapper">
            <img src={gradcamSrc} alt="Grad-CAM Result" style={{ opacity }} />
          </div>
          <div className="card-meta">
            {explanations.gradcam?.description || 'denseblock4 feature map activation'}
          </div>
        </div>

        {/* Card 3: Integrated Gradients */}
        <div className="xai-card">
          <div className="card-title">Integrated Gradients</div>
          <div className="img-wrapper">
            <img src={igSrc} alt="Integrated Gradients Result" style={{ opacity }} />
          </div>
          <div className="card-meta">
            {explanations.integrated_gradients?.description || 'Axiomatic path gradient attribution'}
          </div>
        </div>

        {/* Card 4: SHAP */}
        <div className="xai-card">
          <div className="card-title">SHAP</div>
          <div className="img-wrapper">
            <img src={shapSrc} alt="SHAP Result" style={{ opacity }} />
          </div>
          <div className="card-meta">
            {explanations.shap?.description || 'Shapley value pixel importance'}
          </div>
        </div>
      </div>
    </div>
  );
}
