import React, { useState, useEffect } from 'react';
import './CropCareHero.css';

const SUGGESTIONS = [
  "Upload an image of your crop or ask a question...",
  "Why are my maize leaves turning yellow and brittle?",
  "How to identify early symptoms of common corn rust?",
  "Upload a canopy photo to diagnose water stress severity...",
  "What organic treatments work best for fungal leaf blight?",
];

export default function CropCareHero({ onNavigateToApp }) {
  const [suggestionIdx, setSuggestionIdx] = useState(0);
  const [displayedText, setDisplayedText] = useState(SUGGESTIONS[0]);
  const [animPhase, setAnimPhase] = useState('idle'); // 'typing' | 'idle' | 'fading'
  const [queryText, setQueryText] = useState('');
  const [isFocused, setIsFocused] = useState(false);

  // Typewriter typing in + backwards fade animation
  useEffect(() => {
    if (queryText || isFocused) return;

    let timeoutId;
    const currentTarget = SUGGESTIONS[suggestionIdx];

    if (animPhase === 'typing') {
      if (displayedText.length < currentTarget.length) {
        timeoutId = setTimeout(() => {
          setDisplayedText(currentTarget.slice(0, displayedText.length + 1));
        }, 28);
      } else {
        // Finished typing out, rest in readable idle phase
        timeoutId = setTimeout(() => {
          setAnimPhase('fading');
        }, 3600);
      }
    } else if (animPhase === 'idle') {
      // Initial state displays full suggestion, then fades backwards
      timeoutId = setTimeout(() => {
        setAnimPhase('fading');
      }, 3600);
    } else if (animPhase === 'fading') {
      // Wait for backwards-fade animation to complete before advancing to next suggestion
      timeoutId = setTimeout(() => {
        setSuggestionIdx((prev) => (prev + 1) % SUGGESTIONS.length);
        setDisplayedText('');
        setAnimPhase('typing');
      }, 420);
    }

    return () => clearTimeout(timeoutId);
  }, [animPhase, displayedText, suggestionIdx, queryText, isFocused]);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (onNavigateToApp) {
      onNavigateToApp();
    }
  };

  return (
    <section className="cropcare-hero" aria-label="CropCare Hero Section">
      {/* Background Video with Poster Fallback */}
      <div className="hero-video-container" aria-hidden="true">
        <video
          className="hero-video"
          autoPlay
          muted
          loop
          playsInline
          poster="/hero-poster.jpg"
          key="clean-hero-video"
        >
          <source src="/clean-hero-background.mp4" type="video/mp4" />
          <source src="/clean_hero background video.mp4" type="video/mp4" />
        </video>
        {/* Subtle atmospheric overlay protecting natural artwork vibrancy */}
        <div className="hero-video-overlay" />
      </div>

      {/* Top Navigation Row */}
      <header className="hero-navbar">
        {/* Brand Logo (Left) */}
        <div className="navbar-brand">
          <svg
            className="brand-leaf-icon"
            width="26"
            height="26"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            aria-hidden="true"
          >
            <path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.5 1.5 8.5A8 8 0 0 1 11 20z" />
            <path d="M11 13c1.5-1.5 3.5-2.5 5-3" />
          </svg>
          <span className="brand-name">CropCare</span>
        </div>

        {/* Center Navigation Links */}
        <nav className="navbar-nav" aria-label="Primary navigation">
          <a href="/homepage" className="nav-item nav-item-active">
            Home
          </a>
          <a href="#features" className="nav-item" onClick={(e) => { e.preventDefault(); if (onNavigateToApp) onNavigateToApp(); }}>
            Features
          </a>
          <a href="#how-it-works" className="nav-item" onClick={(e) => { e.preventDefault(); if (onNavigateToApp) onNavigateToApp(); }}>
            How It Works
          </a>
          <a href="#about" className="nav-item">
            About
          </a>
        </nav>

        {/* Right Actions */}
        <div className="navbar-actions">
          <button type="button" className="btn-login" aria-label="Login to CropCare">
            Login
          </button>
          <button
            type="button"
            className="btn-get-started"
            onClick={onNavigateToApp}
            aria-label="Get Started with CropCare"
          >
            Get Started
          </button>
        </div>
      </header>

      {/* Central Hero Content */}
      <div className="hero-content-wrapper">
        <div className="hero-headline-container">
          <h1 className="hero-heading">
            <span className="heading-line-1">Healthier Crops,</span>
            <span className="heading-line-2">Brighter Tomorrows</span>
          </h1>
          <p className="hero-supporting-text">
            Upload a crop image, get instant disease detection and expert answers — all in one place.
          </p>
        </div>

        {/* Glassmorphic Crop Upload and AI Chat Bar */}
        <form
          className="composer-glass-card"
          role="search"
          aria-label="Crop disease detection composer"
          onSubmit={handleSubmit}
        >
          {/* Top Prompt / Input Area */}
          <div className="composer-input-area">
            <div className="composer-input-wrapper">
              {!queryText && (
                <span
                  className={`animated-placeholder ${
                    animPhase === 'fading' ? 'fading-backwards' : 'typing-active'
                  }`}
                  aria-hidden="true"
                >
                  {displayedText}
                  {animPhase === 'typing' && <span className="typewriter-cursor">|</span>}
                </span>
              )}
              <input
                type="text"
                className="composer-input"
                value={queryText}
                onChange={(e) => setQueryText(e.target.value)}
                onFocus={() => setIsFocused(true)}
                onBlur={() => setIsFocused(false)}
                aria-label="Crop query input"
              />
            </div>
          </div>

          {/* Bottom Controls Row */}
          <div className="composer-controls-row">
            {/* Upload Button — Bottom Left */}
            <div className="upload-control-group">
              <button
                type="button"
                className="upload-circle-btn"
                onClick={onNavigateToApp}
                title="Upload crop image (Launch detector)"
                aria-label="Upload crop image"
              >
                {/* Upload tray with upward arrow icon matching the design reference */}
                <svg
                  width="18"
                  height="18"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2.2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  aria-hidden="true"
                >
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                  <polyline points="17 8 12 3 7 8" />
                  <line x1="12" y1="3" x2="12" y2="15" />
                </svg>
              </button>
            </div>

            {/* Primary Action — Bottom Right */}
            <div className="action-control-group">
              <button
                type="submit"
                className="ask-ai-pill-btn"
                aria-label="Ask AI for crop disease detection"
              >
                Ask AI
              </button>
            </div>
          </div>
        </form>
      </div>
    </section>
  );
}
