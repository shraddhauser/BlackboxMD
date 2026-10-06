import React, { useState, useEffect } from 'react';
import Header from './components/Header';
import ImageUploader from './components/ImageUploader';
import PredictionResults from './components/PredictionResults';
import XaiHeatmaps from './components/XaiHeatmaps';
import AnalyticsView from './components/AnalyticsView';
import GalleryView from './components/GalleryView';
import './index.css';

export default function App() {
  const [activeTab, setActiveTab] = useState('diagnostic');
  const [status, setStatus] = useState(null);

  const [selectedFile, setSelectedFile] = useState(null);
  const [selectedSample, setSelectedSample] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);

  const [chkGradcam, setChkGradcam] = useState(true);
  const [chkIg, setChkIg] = useState(true);
  const [chkShap, setChkShap] = useState(true);

  const [predData, setPredData] = useState(null);
  const [expData, setExpData] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isDownloadingPdf, setIsDownloadingPdf] = useState(false);

  useEffect(() => {
    fetch('/api/status')
      .then((res) => res.json())
      .then((data) => setStatus(data))
      .catch((err) => console.error('Status check failed:', err));
  }, []);

  const handleFileSelect = (file) => {
    setSelectedFile(file);
    setSelectedSample(null);
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
  };

  const handleSampleSelect = (sample) => {
    setSelectedSample(sample);
    setSelectedFile(null);
    setPreviewUrl(`/api/sample-image/${sample.class_name}/${sample.filename}`);
  };

  const handleAnalyze = async () => {
    if (!selectedFile && !selectedSample) {
      alert('Please select a test sample or upload a CT scan image first.');
      return;
    }

    setIsAnalyzing(true);
    try {
      // 1. Predict
      const predFormData = new FormData();
      if (selectedFile) {
        predFormData.append('file', selectedFile);
      } else if (selectedSample) {
        predFormData.append('sample_class', selectedSample.class_name);
        predFormData.append('sample_filename', selectedSample.filename);
      }

      const pRes = await fetch('/api/predict', { method: 'POST', body: predFormData });
      if (!pRes.ok) throw new Error('Prediction API call failed');
      const pData = await pRes.json();
      setPredData(pData);

      // 2. Explainability + Feature Extraction (Step 5) + NLP Explanation (Step 6)
      const selectedMethods = [];
      if (chkGradcam) selectedMethods.push('gradcam');
      if (chkIg) selectedMethods.push('integrated_gradients');
      if (chkShap) selectedMethods.push('shap');
      const methodArg = selectedMethods.length > 0 ? selectedMethods.join(',') : 'none';

      const expFormData = new FormData();
      if (selectedFile) {
        expFormData.append('file', selectedFile);
      } else if (selectedSample) {
        expFormData.append('sample_class', selectedSample.class_name);
        expFormData.append('sample_filename', selectedSample.filename);
      }
      expFormData.append('method', methodArg);

      const eRes = await fetch('/api/explain', { method: 'POST', body: expFormData });
      if (eRes.ok) {
        const eData = await eRes.json();
        setExpData(eData);
      }
    } catch (err) {
      alert('Error running analysis: ' + err.message);
      console.error(err);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleDownloadPdf = async () => {
    if (!selectedFile && !selectedSample) {
      alert('Please select or upload a CT scan first.');
      return;
    }

    setIsDownloadingPdf(true);
    try {
      const formData = new FormData();
      if (selectedFile) {
        formData.append('file', selectedFile);
      } else if (selectedSample) {
        formData.append('sample_class', selectedSample.class_name);
        formData.append('sample_filename', selectedSample.filename);
      }
      formData.append('scan_id', `CT-${Date.now().toString().slice(-6)}`);

      const res = await fetch('/api/report/pdf', { method: 'POST', body: formData });
      if (!res.ok) throw new Error('Failed to generate PDF report from server');

      const blob = await res.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `BlackBoxMD_Report_${Date.now()}.pdf`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
    } catch (err) {
      alert('Error downloading PDF report: ' + err.message);
      console.error(err);
    } finally {
      setIsDownloadingPdf(false);
    }
  };

  return (
    <div className="app-container">
      <Header status={status} />

      {/* Navigation Tabs */}
      <nav className="tabs-nav">
        <button
          className={`tab-btn ${activeTab === 'diagnostic' ? 'active' : ''}`}
          onClick={() => setActiveTab('diagnostic')}
        >
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M22 12h-4l-3 9L9 3l-3 9H2"/>
          </svg>
          Live Diagnostics & XAI
        </button>

        <button
          className={`tab-btn ${activeTab === 'analytics' ? 'active' : ''}`}
          onClick={() => setActiveTab('analytics')}
        >
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <line x1="18" y1="20" x2="18" y2="10"/>
            <line x1="12" y1="20" x2="12" y2="4"/>
            <line x1="6" y1="20" x2="6" y2="14"/>
          </svg>
          Model Analytics & Metrics
        </button>

        <button
          className={`tab-btn ${activeTab === 'explorer' ? 'active' : ''}`}
          onClick={() => setActiveTab('explorer')}
        >
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <rect x="3" y="3" width="7" height="7"/>
            <rect x="14" y="3" width="7" height="7"/>
            <rect x="14" y="14" width="7" height="7"/>
            <rect x="3" y="14" width="7" height="7"/>
          </svg>
          Dataset Sample Browser
        </button>
      </nav>

      {/* Main Content Pane */}
      <main className="main-content">
        {activeTab === 'diagnostic' && (
          <div>
            <div className="grid-2col">
              <ImageUploader
                selectedFile={selectedFile}
                selectedSample={selectedSample}
                onFileSelect={handleFileSelect}
                onSampleSelect={handleSampleSelect}
                onAnalyze={handleAnalyze}
                chkGradcam={chkGradcam}
                setChkGradcam={setChkGradcam}
                chkIg={chkIg}
                setChkIg={setChkIg}
                chkShap={chkShap}
                setChkShap={setChkShap}
                isAnalyzing={isAnalyzing}
              />

              <PredictionResults
                predData={predData}
                expData={expData}
                onDownloadPdf={handleDownloadPdf}
                isDownloadingPdf={isDownloadingPdf}
              />
            </div>

            <XaiHeatmaps origImage={previewUrl} expData={expData} />
          </div>
        )}

        {activeTab === 'analytics' && <AnalyticsView />}

        {activeTab === 'explorer' && (
          <GalleryView
            onSelectSample={(sample) => {
              handleSampleSelect(sample);
              setActiveTab('diagnostic');
            }}
          />
        )}
      </main>

      {/* Loading Overlay Modal */}
      {isAnalyzing && (
        <div className="modal-backdrop">
          <div className="modal-card">
            <div className="spinner"></div>
            <h3>Computing Diagnosis & XAI Maps</h3>
            <p>Running DenseNet-121 forward pass, Grad-CAM, spatial feature extraction & clinical NLP synthesis...</p>
          </div>
        </div>
      )}

      <footer style={{ marginTop: 40, textAlign: 'center', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
        BLACKBOX MD — Comparative Explainable AI for Pulmonary Oncology • DenseNet-121 • PyTorch, FastAPI & React
      </footer>
    </div>
  );
}
