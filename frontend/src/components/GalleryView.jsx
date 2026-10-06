import React, { useEffect, useState } from 'react';

export default function GalleryView({ onSelectSample }) {
  const [samples, setSamples] = useState({});

  useEffect(() => {
    fetch('/api/samples')
      .then((res) => res.json())
      .then((data) => setSamples(data.samples || {}))
      .catch((err) => console.error('Error loading gallery samples:', err));
  }, []);

  return (
    <div className="panel glass-card">
      <div className="panel-header">
        <div>
          <h3>Chest CT Dataset Test Gallery</h3>
          <p className="subtext">Browse sample scans from the 4 pulmonary carcinoma categories</p>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: 16, marginTop: 16 }}>
        {Object.entries(samples).flatMap(([cls, files]) =>
          files.slice(0, 4).map((file, idx) => {
            const cleanName = cls.replace(/\./g, ' ');
            const imgUrl = `/api/sample-image/${cls}/${file}`;
            return (
              <div
                key={`${cls}-${idx}`}
                className="xai-card"
                style={{ cursor: 'pointer' }}
                onClick={() => onSelectSample({ class_name: cls, filename: file })}
              >
                <div className="img-wrapper">
                  <img src={imgUrl} alt={`${cls} ${file}`} />
                </div>
                <div className="card-title" style={{ marginTop: 8, textTransform: 'capitalize' }}>
                  {cleanName}
                </div>
                <div className="card-meta">{file}</div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
