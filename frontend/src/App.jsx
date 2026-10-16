import React, { useState, useEffect } from 'react';
import ImageUploader from './components/ImageUploader';
import PredictionResult from './components/PredictionResult';
import CropCareHero from './components/CropCareHero';
import CropCareChatWelcome from './components/CropCareChatWelcome';
import { getHealthStatus, predictCropImage, fetchSampleImage } from './api';
import './App.css';

const SUPPORTED_DEMO_SAMPLES = [
  { id: 'rust', label: 'Corn Rust', icon: '🌽', crop: 'maize' },
  { id: 'high_stress', label: 'Corn Drought', icon: '🌽', crop: 'maize' },
  { id: 'potato_late_blight', label: 'Potato Late Blight', icon: '🥔', crop: 'solanaceae' },
  { id: 'potato_early_blight', label: 'Potato Early Blight', icon: '🥔', crop: 'solanaceae' },
  { id: 'tomato_bacterial_spot', label: 'Tomato Bacterial Spot', icon: '🍅', crop: 'solanaceae' },
  { id: 'tomato_early_blight', label: 'Tomato Early Blight', icon: '🍅', crop: 'solanaceae' },
  { id: 'healthy', label: 'Healthy Leaf', icon: '🍃', crop: 'maize' },
  { id: 'bare_soil', label: 'Bare Soil Control', icon: '🪨', crop: 'maize' },
];

export default function App() {
  const [currentPath, setCurrentPath] = useState(window.location.pathname);
  const [selectedFile, setSelectedFile] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [apiError, setApiError] = useState(null);
  const [healthStatus, setHealthStatus] = useState({ online: false, modelLoaded: false });

  // Handle URL history changes
  useEffect(() => {
    const handlePopState = () => {
      setCurrentPath(window.location.pathname);
    };
    window.addEventListener('popstate', handlePopState);
    return () => window.removeEventListener('popstate', handlePopState);
  }, []);

  const navigateTo = (path) => {
    window.history.pushState({}, '', path);
    setCurrentPath(path);
    window.scrollTo(0, 0);
  };

  // Check backend health on mount for the functional app
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

  // Dynamic page title based on active route
  useEffect(() => {
    if (currentPath === '/homepage' || currentPath === '/homepage/') {
      document.title = 'CropCare — Healthier Crops, Brighter Tomorrows';
    } else if (currentPath === '/chat' || currentPath === '/chat/') {
      document.title = 'CropCare — Chat with CropCare';
    } else {
      document.title = 'FarmWatch — AI Crop Condition & Water Stress Classifier';
    }
  }, [currentPath]);

  // If user visits /chat, render the static CropCare chat welcome page
  if (currentPath === '/chat' || currentPath === '/chat/') {
    return <CropCareChatWelcome />;
  }

  // If user visits /homepage, render the static CropCare hero section
  if (currentPath === '/homepage' || currentPath === '/homepage/') {
    return <CropCareHero onNavigateToApp={() => navigateTo('/')} />;
  }

  // Handlers for the fully functional crop disease detection app at /
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
      const result = await predictCropImage(selectedFile, 'auto');
      setPrediction(result);
    } catch (err) {
      setApiError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleLoadSample = async (sample) => {
    setLoading(true);
    setApiError(null);
    try {
      const file = await fetchSampleImage(sample.id, sample.crop || 'auto');
      setSelectedFile(file);
      const result = await predictCropImage(file, 'auto');
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
                <h1 className="brand-title">FarmWatch AI</h1>
                <span className="badge-ai">UNIVERSAL DIAGNOSTICS</span>
              </div>
              <p className="brand-subtitle">Automated Crop Health, Water Stress & Foliar Disease Engine</p>
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
                  ? 'AI Engine Online (ResNet-18 Multi-Crop)'
                  : 'API Online (Model Not Ready)'
                : 'Backend Disconnected'}
            </span>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="main-content">
        <div className="intro-hero">
          <h2 className="hero-heading">Instant Crop & Disease Diagnosis</h2>
          <p className="hero-desc">
            Upload any crop photograph (Corn, Potato, or Tomato). FarmWatch automatically detects the crop variety and diagnoses foliar disease pathology, drought stress severity, or soil coverage with Grad-CAM visual attention.
          </p>
        </div>

        {/* Curated Supported Demo Samples Strip */}
        <div className="demo-samples-card">
          <div className="demo-samples-header">
            <div className="demo-samples-title">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                <polyline points="14 2 14 8 20 8"></polyline>
                <line x1="16" y1="13" x2="8" y2="13"></line>
                <line x1="16" y1="17" x2="8" y2="17"></line>
                <polyline points="10 9 9 9 8 9"></polyline>
              </svg>
              Supported Demo Images (Click to Test Live):
            </div>
            <span className="demo-samples-sub">Corn • Potato • Tomato • Bare Soil</span>
          </div>
          <div className="sample-buttons">
            {SUPPORTED_DEMO_SAMPLES.map((s) => (
              <button
                key={s.id}
                type="button"
                className="sample-btn"
                onClick={() => handleLoadSample(s)}
                disabled={loading || !healthStatus.online}
                title={`Test with verified ${s.label} image`}
              >
                <span className="sample-icon">{s.icon}</span>
                <span>{s.label}</span>
              </button>
            ))}
          </div>
        </div>

        <div className="content-grid">
          {/* Left Column: Image Input & Controls */}
          <section className="input-section" aria-labelledby="uploader-heading">
            <div className="section-header-row">
              <h3 id="uploader-heading" className="section-title">
                1. Upload Crop Image
              </h3>
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
              <PredictionResult result={prediction} originalFile={selectedFile} onReset={handleClear} />
            ) : (
              <div className="empty-results-state">
                <div className="empty-icon">
                  <svg width="44" height="44" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                    <rect x="2" y="3" width="20" height="14" rx="2" ry="2" />
                    <line x1="8" y1="21" x2="16" y2="21" />
                    <line x1="12" y1="17" x2="12" y2="21" />
                  </svg>
                </div>
                <h4>Awaiting Crop Photograph</h4>
                <p>
                  Upload any crop leaf or field photograph on the left, or click any supported demo sample above to generate the diagnosis and attention heatmap.
                </p>

                <button
                  type="button"
                  className="btn-empty-demo"
                  onClick={() => handleLoadSample(SUPPORTED_DEMO_SAMPLES[2])}
                  disabled={loading || !healthStatus.online}
                >
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <circle cx="12" cy="12" r="3" />
                    <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z" />
                  </svg>
                  Try Demo: Potato Late Blight with Attention Heatmap
                </button>

                <div className="taxonomy-preview">
                  {SUPPORTED_DEMO_SAMPLES.map((s) => (
                    <span key={s.id} className="taxonomy-pill">
                      {s.icon} {s.label}
                    </span>
                  ))}
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
            <strong>FarmWatch Research MVP</strong> • Built with PyTorch (Dual ResNet-18 Models), FastAPI, and React.
            Supports Corn (Zenodo 19108517) and Potato & Tomato Foliar Diseases (PlantVillage).
          </p>
          <p className="footer-sub">
            AI predictions are statistical estimates. Not certified for prescription of fertilizers or agrochemicals.
          </p>
        </div>
      </footer>
    </div>
  );
}
