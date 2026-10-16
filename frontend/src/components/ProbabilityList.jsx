import React from 'react';

const CLASS_CONFIG = {
  healthy: {
    label: 'Healthy Canopy',
    color: '#2d6a4f',
    bg: 'rgba(45, 106, 79, 0.15)',
  },
  low_stress: {
    label: 'Low Water-Stress',
    color: '#e76f51',
    bg: 'rgba(231, 111, 81, 0.15)',
  },
  high_stress: {
    label: 'High Water-Stress',
    color: '#d90429',
    bg: 'rgba(217, 4, 41, 0.15)',
  },
  rust: {
    label: 'Rust Disease',
    color: '#9d4edd',
    bg: 'rgba(157, 78, 221, 0.15)',
  },
  bare_soil: {
    label: 'Bare Soil / Ground',
    color: '#936639',
    bg: 'rgba(147, 102, 57, 0.15)',
  },
};

export default function ProbabilityList({ probabilities, predictedClass }) {
  if (!probabilities) return null;

  // Convert to array and sort descending by probability
  const sortedClasses = Object.entries(probabilities)
    .map(([cls, prob]) => ({
      key: cls,
      prob: Math.min(1.0, Math.max(0.0, prob)),
      percentage: (prob * 100).toFixed(1),
      config: CLASS_CONFIG[cls] || { label: cls, color: '#40916c', bg: 'rgba(64,145,108,0.15)' },
    }))
    .sort((a, b) => b.prob - a.prob);

  return (
    <div className="prob-container">
      <h4 className="prob-heading">Condition Probability Distribution</h4>
      <div className="prob-list">
        {sortedClasses.map(({ key, percentage, config }) => {
          const isTop = key === predictedClass;
          return (
            <div key={key} className={`prob-item ${isTop ? 'prob-item-top' : ''}`}>
              <div className="prob-item-header">
                <span className="prob-item-name">
                  {config.label}
                  {isTop && <span className="top-indicator">Top Match</span>}
                </span>
                <span className="prob-item-val">{percentage}%</span>
              </div>
              <div className="prob-bar-track">
                <div
                  className="prob-bar-fill"
                  style={{
                    width: `${percentage}%`,
                    backgroundColor: config.color,
                  }}
                  role="progressbar"
                  aria-valuenow={percentage}
                  aria-valuemin="0"
                  aria-valuemax="100"
                  aria-label={`${config.label} probability: ${percentage}%`}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
