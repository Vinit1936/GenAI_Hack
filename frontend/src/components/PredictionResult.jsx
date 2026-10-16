import React, { useState, useMemo, useEffect } from 'react';
import ProbabilityList from './ProbabilityList';

const CATEGORY_TAGS = {
  plant_condition: { label: 'Plant Foliage', class: 'tag-plant' },
  water_stress: { label: 'Water Stress', class: 'tag-stress' },
  disease: { label: 'Pathology / Disease', class: 'tag-disease' },
  background: { label: 'Field Background', class: 'tag-bg' },
};

export default function PredictionResult({ result, originalFile, onReset }) {
  // Enabled by default so users immediately see explainability visual upon prediction
  const [showGradCam, setShowGradCam] = useState(true);

  const originalUrl = useMemo(() => {
    if (!originalFile) return null;
    return URL.createObjectURL(originalFile);
  }, [originalFile]);

  useEffect(() => {
    return () => {
      if (originalUrl) {
        URL.revokeObjectURL(originalUrl);
      }
    };
  }, [originalUrl]);

  if (!result) return null;

  const {
    predicted_class,
    display_name,
    category_type,
    disease_name,
    scientific_name,
    pathogen_type,
    host_crop,
    symptoms,
    confidence,
    confidence_warning,
    probabilities,
    interpretation,
    model_version,
    gradcam_image,
  } = result;

  const tag = CATEGORY_TAGS[category_type] || { label: category_type, class: 'tag-generic' };
  const confidencePercent = (confidence * 100).toFixed(1);

  return (
    <div className="result-card" role="region" aria-label="Model prediction results">
      <div className="result-header">
        <div className="result-badge-group">
          <span className={`category-tag ${tag.class}`}>{tag.label}</span>
          <span className="version-tag">Model v{model_version}</span>
        </div>
        <h2 className="predicted-title">{display_name}</h2>
        <div className="confidence-meter-row">
          <div className="confidence-stat">
            <span className="confidence-label">Model Confidence</span>
            <span className="confidence-number">{confidencePercent}%</span>
          </div>
          <div className="confidence-indicator-bar">
            <div
              className={`confidence-indicator-fill ${confidence_warning ? 'warning' : 'healthy'}`}
              style={{ width: `${confidencePercent}%` }}
            />
          </div>
        </div>
      </div>

      {confidence_warning && (
        <div className="warning-banner" role="alert">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
            <line x1="12" y1="9" x2="12" y2="13" />
            <line x1="12" y1="17" x2="12.01" y2="17" />
          </svg>
          <div>
            <strong>Low Confidence Estimate ({confidencePercent}%)</strong>
            <p>
              The model confidence falls below the standard reliability threshold (60%).
              Results may be ambiguous or outside the dataset distribution. Re-take photo with clearer lighting and focus.
            </p>
          </div>
        </div>
      )}

      {disease_name && (
        <div className={`disease-diagnostic-card ${category_type === 'disease' ? 'is-pathology' : ''}`}>
          <div className="disease-diagnostic-header">
            <span className="disease-badge-pill">
              {category_type === 'disease' ? 'Foliar Disease Identified' : 'Condition Identity'}
            </span>
            <span className="crop-pill">Host Crop: {host_crop || 'Maize / Corn (Zea mays)'}</span>
          </div>

          <div className="disease-name-block">
            <div className="disease-label-caption">Specific Disease / Condition:</div>
            <div className="disease-primary-name">{disease_name}</div>
            {scientific_name && scientific_name !== 'N/A' && (
              <div className="disease-scientific-name">
                <strong>Pathogen:</strong> <em>{scientific_name}</em> &bull; <span className="pathogen-type">{pathogen_type}</span>
              </div>
            )}
          </div>

          {symptoms && (
            <div className="disease-symptoms-block">
              <span className="disease-symptoms-label">Key Diagnostic Symptoms:</span>
              <p className="disease-symptoms-text">{symptoms}</p>
            </div>
          )}
        </div>
      )}

      <div className="interpretation-box">
        <h4 className="interpretation-heading">Agronomic Interpretation</h4>
        <p className="interpretation-text">{interpretation}</p>
      </div>

      {gradcam_image && (
        <div className="gradcam-card">
          <div
            className="gradcam-header"
            onClick={() => setShowGradCam(!showGradCam)}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => {
              if (e.key === 'Enter' || e.key === ' ') {
                setShowGradCam(!showGradCam);
              }
            }}
          >
            <div className="gradcam-title-group">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="12" cy="12" r="3" />
                <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z" />
              </svg>
              <span>Explainable AI: Attention Heatmap (Grad-CAM)</span>
            </div>
            <button
              type="button"
              className="gradcam-toggle-btn"
              onClick={(e) => {
                e.stopPropagation();
                setShowGradCam(!showGradCam);
              }}
            >
              {showGradCam ? 'Hide Heatmap' : 'View Heatmap'}
            </button>
          </div>

          {showGradCam && (
            <div className="gradcam-body">
              <div className="gradcam-visual-comparison">
                {originalUrl && (
                  <div className="visual-pane">
                    <span className="pane-label">Original Input Photo</span>
                    <div className="pane-img-wrap">
                      <img src={originalUrl} alt="Original crop foliage" className="pane-img" />
                    </div>
                  </div>
                )}
                <div className="visual-pane">
                  <span className="pane-label">AI Attention Heatmap</span>
                  <div className="pane-img-wrap heatmap-highlight">
                    <img src={gradcam_image} alt="Grad-CAM visual attribution overlay" className="pane-img" />
                  </div>
                </div>
              </div>
              <p className="gradcam-caption">
                <strong>Attribution Heatmap (ResNet-18 layer4):</strong> Warmer colors (red/orange) highlight the specific foliar regions that drove the model to predict <strong>{display_name}</strong>.
              </p>
            </div>
          )}
        </div>
      )}

      <ProbabilityList probabilities={probabilities} predictedClass={predicted_class} />

      <div className="disclaimer-footer">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <circle cx="12" cy="12" r="10" />
          <line x1="12" y1="16" x2="12" y2="12" />
          <line x1="12" y1="8" x2="12.01" y2="8" />
        </svg>
        <p>
          <strong>Scientific Notice:</strong> This prediction is an automated computer vision estimate trained on
          the Zenodo UAV maize canopy dataset. It does not constitute certified agronomic advice, lab pathology,
          or pesticide guidance. Consult certified agricultural professionals.
        </p>
      </div>

      <div className="action-row">
        <button type="button" className="btn-secondary" onClick={onReset} aria-label="Analyze another crop image">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <polyline points="1 4 1 10 7 10" />
            <polyline points="23 20 23 14 17 14" />
            <path d="M20.49 9A9 9 0 0 0 5.64 5.64L1 10m22 4l-4.64 4.36A9 9 0 0 1 3.51 15" />
          </svg>
          Analyze Another Image
        </button>
      </div>
    </div>
  );
}
