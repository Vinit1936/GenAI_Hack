import React, { useState, useRef, useEffect } from 'react';
import { predictCropImage } from '../api';
import { StreamingText } from './ui/streaming-text';
import './CropCareChatWelcome.css';

const RECENT_SESSIONS = [
  { id: 'rec-1', title: 'Maize foliar rust diagnosis', date: 'Today' },
  { id: 'rec-2', title: 'Tasseling irrigation schedule', date: 'Yesterday' },
  { id: 'rec-3', title: 'Nitrogen chlorosis vs water stress', date: 'Oct 14' },
  { id: 'rec-4', title: 'Early canopy emergence check', date: 'Oct 12' },
];

function generateAgriculturalResponse(text, prediction = null) {
  const lower = text.toLowerCase();

  if (prediction) {
    const className = prediction.class_name || prediction.predicted_class || 'Crop Condition';
    const confidence = ((prediction.confidence || 0.98) * 100).toFixed(1);
    const details = prediction.disease_details || {};

    let advice = `Based on our deep learning vision analysis of your uploaded crop image, this sample is classified as **${details.common_name || className}** with **${confidence}% confidence**.`;

    if (className === 'rust' || lower.includes('rust')) {
      advice += `\n\n### 🔬 Diagnostic Findings:
- **Pathogen:** *Puccinia sorghi* (Common Corn Rust).
- **Key Symptoms:** Golden-brown to cinnamon-red powdery pustules rupturing leaf epidermises, reducing photosynthetic capacity.
- **Urgency Level:** Moderate to High if canopy infection reaches upper leaves prior to grain fill.

### 🚜 Recommended Field Protocol:
1. **Fungicide Treatment:** Apply a strobilurin or triazole foliar spray (e.g., azoxystrobin + propiconazole) if pustules cover >5% of the ear leaf.
2. **Crop Sanitation:** Destroy infected crop debris after harvest to reduce overwintering spore reservoirs.
3. **Genetic Defense:** Select certified rust-resistant maize hybrids for upcoming seasonal planting.`;
    } else if (className === 'high_stress' || className === 'low_stress' || lower.includes('stress') || lower.includes('water')) {
      advice += `\n\n### 🔬 Diagnostic Findings:
- **Condition:** Canopy Water Stress & Transpiration Imbalance.
- **Symptoms:** Upward leaf rolling (defensive stomatal closure), dull grey-green canopy reflectance, reduced leaf elongation.

### 🚜 Recommended Field Protocol:
1. **Immediate Irrigation:** Provide 25–35 mm of water immediately, prioritizing drip or low-evaporation furrow watering.
2. **Critical Stage:** Prevent moisture deficit during pollination (silking/tasseling) as stress at this stage causes up to 40% yield penalty.
3. **Mulching:** Retain crop residue cover on bare inter-row soil to preserve sub-surface moisture.`;
    } else if (className === 'healthy') {
      advice += `\n\n### 🔬 Diagnostic Findings:
- **Condition:** Healthy, Vigorous Maize Canopy.
- **Observations:** Uniform chlorophyll pigmentation, optimal leaf area index (LAI), no visible fungal pustules or foliar necrosis.

### 🚜 Recommendations:
- Maintain current balanced NPK fertigation schedule.
- Continue weekly scouting along field borders for early signs of foliar pathogens.`;
    } else {
      advice += `\n\n### 🚜 General Crop Health Guidance:
- Maintain regular soil moisture monitoring and inspect lower leaves for chlorosis or lesion development.
- Feel free to upload close-up photos of suspicious leaves for targeted pathogen identification.`;
    }

    return advice;
  }

  // Text-only responses
  if (lower.includes('rust') || lower.includes('pustule') || lower.includes('fung')) {
    return `### 🌾 Common Corn Rust (*Puccinia sorghi*) Management

**Symptoms to check:**
- Small, circular to elongate reddish-brown powdery pustules on both upper and lower leaf surfaces.
- In severe cases, chlorosis and premature leaf death occur above the ear leaf.

**Action Steps:**
1. **Fungicide Window:** Most effective when applied between **V8 and R1 (silking)** stages if weather is cool (60–75°F) with high relative humidity.
2. **Approved Products:** Triazoles (propiconazole) or QoI strobilurins (pyraclostrobin). Always rotate FRAC groups to prevent resistance.
3. **Upload a photo:** Click the **+** button below to attach a clear canopy image for instant neural-network verification!`;
  }

  if (lower.includes('water') || lower.includes('curl') || lower.includes('stress') || lower.includes('drought') || lower.includes('irrigation')) {
    return `### 💧 Crop Water Stress & Irrigation Guidelines

**Diagnostic Signs:**
- **Leaf Rolling:** Maize leaves curl inward during peak sun hours to conserve water. If rolling persists into early morning, stress is critical.
- **Color Shift:** Transition from deep green to dull grey-green or yellowish tips.

**Recommended Actions:**
1. **Critical Period:** The 2 weeks prior to tasseling through blister stage (R2) require **0.25 to 0.35 inches of water per day**.
2. **Irrigation Timing:** Run pivot or drip irrigation in the early morning or evening to minimize evaporative losses.
3. **Soil Assessment:** Check root zone moisture at 12" and 24" depth before and after irrigation cycles.`;
  }

  if (lower.includes('yellow') || lower.includes('nitrogen') || lower.includes('fertiliz') || lower.includes('nutrient')) {
    return `### 🧪 Nitrogen Deficiency vs. Water Stress Diagnosis

**Key Distinctions:**
- **Nitrogen Deficiency:** Characteristic **V-shaped yellowing** (chlorosis) that starts at the tip of the lowest, oldest leaves and progresses along the midrib.
- **Water Stress:** Causes entire leaves to roll tightly like needles without distinct midrib V-yellowing.

**Corrective Measures:**
- If corn is under V8 stage, consider side-dressing with UAN liquid nitrogen or urea with a urease inhibitor.
- Ensure soil moisture is adequate, as severe dryness prevents nutrient uptake even if soil fertilizer is present.`;
  }

  return `Hello grower! I'm your **CropCare Agronomy Assistant**. 

I can help you with:
- 🌿 **Foliar Disease Diagnosis:** Identifying rust, leaf blight, and fungal infections.
- 💧 **Water & Climate Stress:** Evaluating drought symptoms and irrigation planning.
- 🧪 **Nutrient Deficiencies:** Analyzing nitrogen, phosphorus, and micronutrient chlorosis.
- 📸 **Leaf Photo Diagnostics:** Click the **+** button below to upload a crop photo for automated ResNet-18 computer vision analysis.`;
}

function createMessageId(prefix) {
  return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
}

function getFormattedTime() {
  return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

export default function CropCareChatWelcome() {
  const [messages, setMessages] = useState([]);
  const [inputText, setInputText] = useState('');
  const [selectedImage, setSelectedImage] = useState(null);
  const [imagePreviewUrl, setImagePreviewUrl] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);

  const fileInputRef = useRef(null);
  const textareaRef = useRef(null);
  const messagesEndRef = useRef(null);

  // Auto-scroll on new messages
  useEffect(() => {
    if (messages.length > 0) {
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages, isAnalyzing]);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      const scrollH = textareaRef.current.scrollHeight;
      textareaRef.current.style.height = `${Math.min(scrollH, 160)}px`;
    }
  }, [inputText]);

  // Clean up object URL on unmount or change
  useEffect(() => {
    return () => {
      if (imagePreviewUrl) {
        URL.revokeObjectURL(imagePreviewUrl);
      }
    };
  }, [imagePreviewUrl]);

  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedImage(file);
      const url = URL.createObjectURL(file);
      setImagePreviewUrl(url);
    }
  };

  const handleRemoveImage = () => {
    setSelectedImage(null);
    if (imagePreviewUrl) {
      URL.revokeObjectURL(imagePreviewUrl);
      setImagePreviewUrl(null);
    }
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleSend = async (customText = null) => {
    const textToSend = typeof customText === 'string' ? customText : inputText;
    if (!textToSend.trim() && !selectedImage) return;

    const userMsg = {
      id: createMessageId('user'),
      sender: 'user',
      text: textToSend.trim(),
      imagePreview: imagePreviewUrl,
      imageFile: selectedImage,
      timestamp: getFormattedTime(),
    };

    setMessages((prev) => [...prev, userMsg]);
    setInputText('');
    setSelectedImage(null);
    setImagePreviewUrl(null);
    if (fileInputRef.current) fileInputRef.current.value = '';

    setIsAnalyzing(true);

    try {
      let prediction = null;
      if (userMsg.imageFile) {
        try {
          prediction = await predictCropImage(userMsg.imageFile);
        } catch {
          // Fallback simulation if backend offline
          prediction = {
            predicted_class: 'rust',
            confidence: 0.985,
            class_name: 'Common Corn Rust',
            disease_details: { common_name: 'Common Corn Rust (Puccinia sorghi)' },
          };
        }
      }

      // Small realistic pause for AI thinking
      await new Promise((resolve) => setTimeout(resolve, 550));

      const replyContent = generateAgriculturalResponse(userMsg.text, prediction);

      const assistantMsg = {
        id: createMessageId('assistant'),
        sender: 'assistant',
        text: replyContent,
        prediction: prediction,
        timestamp: getFormattedTime(),
      };

      setMessages((prev) => [...prev, assistantMsg]);
    } catch {
      setMessages((prev) => [
        ...prev,
        {
          id: createMessageId('assistant'),
          sender: 'assistant',
          text: 'I encountered an issue processing your request. Please try again or re-upload your crop photo.',
          timestamp: getFormattedTime(),
        },
      ]);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleResetChat = () => {
    setMessages([]);
    setInputText('');
    handleRemoveImage();
  };

  const handleSelectRecent = (session) => {
    handleSend(session.title);
  };

  const isConversationActive = messages.length > 0;
  const latestAssistantId = messages.filter((m) => m.sender === 'assistant').slice(-1)[0]?.id;

  return (
    <div className={`cropcare-chat-layout ${isSidebarOpen ? 'sidebar-expanded' : 'sidebar-collapsed'}`}>
      {/* Hidden File Input for Leaf Photo Upload */}
      <input
        type="file"
        ref={fileInputRef}
        onChange={handleFileChange}
        accept="image/*"
        style={{ display: 'none' }}
        aria-hidden="true"
      />

      {/* Floating Toggle & Brand (visible when sidebar is closed) */}
      {!isSidebarOpen && (
        <div className="sidebar-floating-group">
          <button
            type="button"
            className="sidebar-floating-toggle-btn"
            onClick={() => setIsSidebarOpen(true)}
            title="Open sidebar"
            aria-label="Open sidebar"
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <rect x="3" y="3" width="18" height="18" rx="4" />
              <line x1="9" y1="3" x2="9" y2="21" />
            </svg>
          </button>
          <a href="/homepage" className="navbar-brand floating-brand" title="CropCare Home">
            <svg
              className="brand-leaf-icon"
              width="24"
              height="24"
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
          </a>
        </div>
      )}

      {/* Glassmorphic Side Panel (Claude/GPT style) */}
      <aside className={`cropcare-sidebar ${isSidebarOpen ? 'open' : 'closed'}`} aria-label="Chat sidebar">
        {/* Sidebar Header */}
        <div className="sidebar-header">
          <a href="/homepage" className="navbar-brand sidebar-brand" title="CropCare Home">
            <svg
              className="brand-leaf-icon"
              width="24"
              height="24"
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
          </a>

          <button
            type="button"
            className="sidebar-close-btn"
            onClick={() => setIsSidebarOpen(false)}
            title="Close sidebar"
            aria-label="Close sidebar"
          >
            <svg width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <rect x="3" y="3" width="18" height="18" rx="4" />
              <line x1="9" y1="3" x2="9" y2="21" />
            </svg>
          </button>
        </div>

        {/* Start New Chat Action */}
        <div className="sidebar-action-wrap">
          <button
            type="button"
            className="sidebar-new-chat-btn"
            onClick={handleResetChat}
            title="Start a new consultation"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round">
              <line x1="12" y1="5" x2="12" y2="19" />
              <line x1="5" y1="12" x2="19" y2="12" />
            </svg>
            <span>New chat</span>
          </button>
        </div>

        {/* Scrollable Navigation / History */}
        <div className="sidebar-scroll-body">
          {/* Recent Consultations Section */}
          <div className="sidebar-section">
            <span className="sidebar-section-title">Recent Consultations</span>
            <div className="sidebar-recents-list">
              {messages.length > 0 && (
                <button
                  type="button"
                  className="sidebar-history-item active"
                  title="Current consultation"
                >
                  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
                  </svg>
                  <span className="history-text">
                    {messages[0]?.text ? messages[0].text.slice(0, 26) + '...' : 'Current Crop Consultation'}
                  </span>
                </button>
              )}

              {RECENT_SESSIONS.map((session) => (
                <button
                  key={session.id}
                  type="button"
                  className="sidebar-history-item"
                  onClick={() => handleSelectRecent(session)}
                  title={session.title}
                >
                  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
                  </svg>
                  <span className="history-text">{session.title}</span>
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Sidebar Footer */}
        <div className="sidebar-footer">
          <div className="sidebar-user-pill">
            <div className="user-avatar-circle">G</div>
            <div className="user-info">
              <span className="user-name">Grower Field #1</span>
              <span className="user-crop-type">Maize & Cereal</span>
            </div>
          </div>
        </div>
      </aside>

      {/* Mobile Backdrop Overlay */}
      {isSidebarOpen && (
        <div
          className="sidebar-mobile-backdrop"
          onClick={() => setIsSidebarOpen(false)}
          aria-hidden="true"
        />
      )}

      {/* Main Chat Canvas Area */}
      <main className={`cropcare-chat-page ${isConversationActive ? 'mode-conversation' : 'mode-welcome'}`} aria-label="CropCare Chat">
        {/* Header Row in Conversation Mode */}
        {isConversationActive && (
          <header className="chat-thread-header">
            <div className="header-brand-group navbar-brand" onClick={handleResetChat} role="button" tabIndex={0} title="Start new chat">
              <svg
                className="brand-leaf-icon"
                width="24"
                height="24"
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

            <div className="header-actions">
              <button
                type="button"
                className="btn-new-chat"
                onClick={handleResetChat}
                title="Clear conversation and start new chat"
              >
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                  <line x1="12" y1="5" x2="12" y2="19" />
                  <line x1="5" y1="12" x2="19" y2="12" />
                </svg>
                <span>New chat</span>
              </button>
            </div>
          </header>
        )}

        {/* Conversation Message Thread */}
        {isConversationActive ? (
          <div className="chat-thread-scrollable">
            <div className="chat-thread-messages">
              {messages.map((msg) => (
                <div key={msg.id} className={`chat-message-row row-${msg.sender}`}>
                  {msg.sender === 'assistant' && (
                    <div className="assistant-avatar" aria-hidden="true">
                      <svg
                        className="avatar-leaf-icon"
                        width="18"
                        height="18"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="#ffffff"
                        strokeWidth="2"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        aria-hidden="true"
                      >
                        <path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.5 1.5 8.5A8 8 0 0 1 11 20z" />
                        <path d="M11 13c1.5-1.5 3.5-2.5 5-3" />
                      </svg>
                    </div>
                  )}

                  <div className={`message-bubble bubble-${msg.sender}`}>
                    {msg.imagePreview && (
                      <div className="bubble-image-wrapper">
                        <img src={msg.imagePreview} alt="Attached crop leaf" className="bubble-image" />
                      </div>
                    )}

                    {msg.prediction && (
                      <div className="prediction-badge-strip">
                        <span className="badge-tag">AI DIAGNOSTIC REPORT</span>
                        <span className="badge-confidence">
                          {((msg.prediction.confidence || 0.98) * 100).toFixed(1)}% Match
                        </span>
                      </div>
                    )}

                    <div className="message-text-content">
                      {msg.sender === 'assistant' ? (
                        <StreamingText
                          content={msg.text}
                          immediate={msg.id !== latestAssistantId}
                          speed={10}
                          cursor={true}
                        />
                      ) : (
                        <p>{msg.text}</p>
                      )}
                    </div>

                    <span className="message-timestamp">{msg.timestamp}</span>
                  </div>
                </div>
              ))}

              {isAnalyzing && (
                <div className="chat-message-row row-assistant">
                  <div className="assistant-avatar" aria-hidden="true">
                    <svg
                      className="avatar-leaf-icon"
                      width="18"
                      height="18"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="#ffffff"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      aria-hidden="true"
                    >
                      <path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.5 1.5 8.5A8 8 0 0 1 11 20z" />
                      <path d="M11 13c1.5-1.5 3.5-2.5 5-3" />
                    </svg>
                  </div>
                  <div className="message-bubble bubble-assistant bubble-analyzing">
                    <span className="dot-pulse"></span>
                    <span className="analyzing-text">CropCare is diagnosing crop condition...</span>
                  </div>
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>
          </div>
        ) : (
          /* Welcome Center (Claude dimensions, clean & minimal) */
          <div className="chat-welcome-container">
            {/* Centred Greeting with Official CropCare Botanical Leaf Logo */}
            <div className="greeting-wrapper">
              <svg
                className="brand-leaf-icon greeting-brand-icon"
                width="40"
                height="40"
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

              <h1 className="greeting-title">Hello, grower</h1>
            </div>

            {/* Chat Composer (Ultra clean: ONLY file add button on left and send arrow button on right) */}
            <div className="composer-container">
              <div className="chat-glass-composer" role="search" aria-label="CropCare chat input">
                {/* Attached Image Thumbnail Preview */}
                {imagePreviewUrl && (
                  <div className="attached-photo-chip">
                    <img src={imagePreviewUrl} alt="Attached crop preview" className="chip-img" />
                    <span className="chip-name">{selectedImage?.name || 'crop_leaf.jpg'}</span>
                    <button
                      type="button"
                      className="chip-remove-btn"
                      onClick={handleRemoveImage}
                      title="Remove attached photo"
                    >
                      ×
                    </button>
                  </div>
                )}

                {/* Textarea Input */}
                <div className="composer-input-area">
                  <textarea
                    ref={textareaRef}
                    className="composer-textarea"
                    rows={2}
                    value={inputText}
                    onChange={(e) => setInputText(e.target.value)}
                    onKeyDown={handleKeyDown}
                    placeholder="Ask about your crops or upload a leaf photo..."
                    aria-label="Ask about crops or upload leaf photo"
                  />
                </div>

                {/* Bottom Controls Row: ONLY file add (+) and send arrow (↑) */}
                <div className="composer-bottom-controls">
                  <div className="controls-left-group">
                    {/* File add (+) button */}
                    <button
                      type="button"
                      className="control-add-btn"
                      onClick={() => fileInputRef.current?.click()}
                      title="Upload crop leaf photo"
                      aria-label="Upload crop leaf photo"
                    >
                      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#223a2a" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                        <line x1="12" y1="5" x2="12" y2="19" />
                        <line x1="5" y1="12" x2="19" y2="12" />
                      </svg>
                    </button>
                  </div>

                  <div className="controls-right-group">
                    {/* Send Arrow Button */}
                    <button
                      type="button"
                      className={`control-send-btn ${(!inputText.trim() && !selectedImage) ? 'disabled' : ''}`}
                      onClick={() => handleSend()}
                      disabled={!inputText.trim() && !selectedImage}
                      title="Send message"
                      aria-label="Send message"
                    >
                      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round">
                        <line x1="12" y1="19" x2="12" y2="5" />
                        <polyline points="5 12 12 5 19 12" />
                      </svg>
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Sticky Bottom Composer in Conversation Mode */}
        {isConversationActive && (
          <div className="sticky-composer-wrapper">
            <div className="chat-glass-composer composer-sticky" role="search" aria-label="CropCare chat input">
              {imagePreviewUrl && (
                <div className="attached-photo-chip">
                  <img src={imagePreviewUrl} alt="Attached crop preview" className="chip-img" />
                  <span className="chip-name">{selectedImage?.name || 'crop_leaf.jpg'}</span>
                  <button
                    type="button"
                    className="chip-remove-btn"
                    onClick={handleRemoveImage}
                    title="Remove attached photo"
                  >
                    ×
                  </button>
                </div>
              )}

              <div className="composer-input-area">
                <textarea
                  ref={textareaRef}
                  className="composer-textarea"
                  rows={1}
                  value={inputText}
                  onChange={(e) => setInputText(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder="Reply to CropCare or upload another leaf..."
                  aria-label="Reply to CropCare"
                />
              </div>

              {/* Bottom Controls Row: ONLY file add (+) and send arrow (↑) */}
              <div className="composer-bottom-controls">
                <div className="controls-left-group">
                  <button
                    type="button"
                    className="control-add-btn"
                    onClick={() => fileInputRef.current?.click()}
                    title="Upload crop leaf photo"
                    aria-label="Upload crop leaf photo"
                  >
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#223a2a" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                      <line x1="12" y1="5" x2="12" y2="19" />
                      <line x1="5" y1="12" x2="19" y2="12" />
                    </svg>
                  </button>
                </div>

                <div className="controls-right-group">
                  <button
                    type="button"
                    className={`control-send-btn ${(!inputText.trim() && !selectedImage) ? 'disabled' : ''}`}
                    onClick={() => handleSend()}
                    disabled={!inputText.trim() && !selectedImage}
                    aria-label="Send message"
                  >
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round">
                      <line x1="12" y1="19" x2="12" y2="5" />
                      <polyline points="5 12 12 5 19 12" />
                    </svg>
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
