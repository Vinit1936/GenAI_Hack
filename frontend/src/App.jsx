import React, { useState, useEffect } from 'react';
import ImageUploader from './components/ImageUploader';
import PredictionResult from './components/PredictionResult';
import { getHealthStatus, predictCropImage, fetchSampleImage } from './api';
import './App.css';

const SAMPLE_CLASSES = [
  { id: 'healthy', label: 'Healthy' },
  { id: 'low_stress', label: 'Low Stress' },
  { id: 'high_stress', label: 'High Stress' },
  { id: 'rust', label: 'Rust' },
  { id: 'bare_soil', label: 'Bare Soil' },
];

export default function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [apiError, setApiError] = useState(null);
  const [healthStatus, setHealthStatus] = useState({ online: false, modelLoaded: false });

  // Check backend health on mount
  useEffect(() => {
    let ignore = false;
    getHealthStatus()
      .then((data) => {
        if (!ignore) {
          setHealthStatus({
            online: true,
            modelLoaded: data.model_loaded,
            architecture: data.model_architecture,
          });
          setApiError(null);
        }
      })
      .catch(() => {
        if (!ignore) {
          setHealthStatus({ online: false, modelLoaded: false });
        }
      });

    return () => {
      ignore = true;
    };
  }, []);

  const handleImageSelected = (file) => {
    setSelectedFile(file);
    setPrediction(null);
    setApiError(null);
  };

  const handleClear = () => {
    setSelectedFile(null);
    setPrediction(null);
    setApiError(null);
  };

  const handleAnalyze = async () => {
    if (!selectedFile || loading) return;

    setLoading(true);
    setApiError(null);

    try {
      const result = await predictCropImage(selectedFile);
      setPrediction(result);
    } catch (err) {
      setApiError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleLoadSample = async (className) => {
    setLoading(true);
    setApiError(null);
    try {
      const file = await fetchSampleImage(className);
      setSelectedFile(file);
      const result = await predictCropImage(file);
      setPrediction(result);
    } catch (err) {
      setApiError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-layout">
      {/* Top Banner / Navbar */}
      <header className="app-header">
        <div className="header-inner">
          <div className="brand">
            <div className="brand-icon">
              <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M12 2a10 10 0 0 1 10 10c0 5.523-4.477 10-10 10S2 17.523 2 12A10 10 0 0 1 12 2z" />
                <path d="M7 12c2.5-4 7.5-4 10 0" />
                <path d="M12 8v8" />
              </svg>
            </div>
            <div>
              <div className="brand-title-row">
                <h1 className="brand-title">FarmWatch</h1>
                <span className="badge-ai">AI ESTIMATION</span>
              </div>
              <p className="brand-subtitle">Field Crop Water Stress & Condition Classifier</p>
            </div>
          </div>

          <div className="system-status">
            <span
              className={`status-pill ${
                healthStatus.online && healthStatus.modelLoaded
                  ? 'status-online'
                  : 'status-offline'
              }`}
            >
              <span className="status-dot"></span>
              {healthStatus.online
                ? healthStatus.modelLoaded
                  ? 'Model Ready (ResNet-18)'
                  : 'API Online (Model Not Ready)'
                : 'Backend Disconnected'}
            </span>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="main-content">
        <div className="intro-hero">
          <h2 className="hero-heading">Detect Crop Water Stress & Foliar Conditions</h2>
          <p className="hero-desc">
            Upload aerial or close-canopy maize photography to estimate water stress severity,
            foliar rust pathology, or ground cover using PyTorch transfer learning.
          </p>
        </div>

        <div className="content-grid">
          {/* Left Column: Image Input & Controls */}
          <section className="input-section" aria-labelledby="uploader-heading">
            <div className="section-header-row">
              <h3 id="uploader-heading" className="section-title">
                1. Select Crop Image
              </h3>
            </div>

            {/* Quick Test Samples */}
            <div className="sample-bar">
              <span className="sample-label">Or load a held-out test sample:</span>
              <div className="sample-buttons">
                {SAMPLE_CLASSES.map((cls) => (
                  <button
                    key={cls.id}
                    type="button"
                    className="sample-btn"
                    onClick={() => handleLoadSample(cls.id)}
                    disabled={loading || !healthStatus.online}
                    title={`Test with held-out ${cls.label} sample`}
                  >
                    {cls.label}
                  </button>
                ))}
              </div>
            </div>

            <ImageUploader
              onImageSelected={handleImageSelected}
              selectedFile={selectedFile}
              onClear={handleClear}
              disabled={loading}
            />

            {/* Action Bar */}
            <div className="action-bar">
              <button
                type="button"
                className={`btn-primary ${loading ? 'loading' : ''}`}
                onClick={handleAnalyze}
                disabled={!selectedFile || loading}
                aria-busy={loading}
              >
                {loading ? (
                  <>
                    <span className="spinner" aria-hidden="true"></span>
                    Analyzing Image...
                  </>
                ) : (
                  <>
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <circle cx="11" cy="11" r="8"></circle>
                      <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
                    </svg>
                    Analyze Crop Condition
                  </>
                )}
              </button>
            </div>

            {apiError && (
              <div className="alert-error" role="alert">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <polygon points="7.86 2 16.14 2 22 7.86 22 16.14 16.14 22 7.86 22 2 16.14 2 7.86 7.86 2"></polygon>
                  <line x1="12" y1="8" x2="12" y2="12"></line>
                  <line x1="12" y1="16" x2="12.01" y2="16"></line>
                </svg>
                <div className="alert-content">
                  <strong>Analysis Error</strong>
                  <p>{apiError}</p>
                </div>
              </div>
            )}
          </section>

          {/* Right Column: Prediction Results & Insights */}
          <section className="results-section" aria-labelledby="results-heading">
            <h3 id="results-heading" className="section-title">
              2. Analysis & Diagnostics
            </h3>

            {prediction ? (
              <PredictionResult result={prediction} onReset={handleClear} />
            ) : (
              <div className="empty-results-state">
                <div className="empty-icon">
                  <svg width="44" height="44" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                    <rect x="2" y="3" width="20" height="14" rx="2" ry="2" />
                    <line x1="8" y1="21" x2="16" y2="21" />
                    <line x1="12" y1="17" x2="12" y2="21" />
                  </svg>
                </div>
                <h4>Awaiting Crop Image</h4>
                <p>
                  Choose or drop a maize canopy photograph on the left, or click one of the quick test samples above to generate the deep-learning diagnostic report.
                </p>
                <div className="taxonomy-preview">
                  <span className="taxonomy-pill">Healthy</span>
                  <span className="taxonomy-pill">Low Stress</span>
                  <span className="taxonomy-pill">High Stress</span>
                  <span className="taxonomy-pill">Rust Disease</span>
                  <span className="taxonomy-pill">Bare Soil</span>
                </div>
              </div>
            )}
          </section>
        </div>
      </main>

      {/* Footer & Provenance */}
      <footer className="app-footer">
        <div className="footer-inner">
          <p className="footer-notice">
            <strong>FarmWatch Research MVP</strong> • Built with PyTorch (ResNet-18 Transfer Learning), FastAPI, and React.
            Trained on Zenodo Record 19108517 (Field UAV Maize Condition Subset).
          </p>
          <p className="footer-sub">
            AI predictions are statistical estimates. Not certified for prescription of fertilizers or agrochemicals.
          </p>
        </div>
      </footer>
    </div>
  );
}
