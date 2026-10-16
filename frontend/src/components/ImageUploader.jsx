import React, { useState, useRef, useEffect } from 'react';

const MAX_BYTES = 10 * 1024 * 1024; // 10MB
const ALLOWED_TYPES = ['image/jpeg', 'image/png', 'image/tiff', 'image/webp'];

export default function ImageUploader({ onImageSelected, selectedFile, onClear, disabled }) {
  const [dragOver, setDragOver] = useState(false);
  const [errorMessage, setErrorMessage] = useState(null);
  const fileInputRef = useRef(null);

  // Manage object URL with useMemo and cleanup effect
  const previewUrl = React.useMemo(() => {
    if (!selectedFile) return null;
    return URL.createObjectURL(selectedFile);
  }, [selectedFile]);

  useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

  const handleValidateAndSelect = (file) => {
    setErrorMessage(null);

    if (!file) return;

    if (file.size > MAX_BYTES) {
      setErrorMessage(`File is too large (${(file.size / (1024 * 1024)).toFixed(1)} MB). Maximum allowed size is 10 MB.`);
      return;
    }

    // Check extension or MIME type
    const isAllowedMime = ALLOWED_TYPES.includes(file.type);
    const hasAllowedExt = /\.(jpe?g|png|tif|tiff|webp)$/i.test(file.name);

    if (!isAllowedMime && !hasAllowedExt) {
      setErrorMessage('Unsupported file format. Please upload a JPEG, PNG, or TIFF image.');
      return;
    }

    onImageSelected(file);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    if (disabled) return;

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleValidateAndSelect(e.dataTransfer.files[0]);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    if (!disabled) setDragOver(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setDragOver(false);
  };

  const handleFileInputChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleValidateAndSelect(e.target.files[0]);
    }
  };

  const handleReset = (e) => {
    e.stopPropagation();
    if (fileInputRef.current) fileInputRef.current.value = '';
    setErrorMessage(null);
    onClear();
  };

  return (
    <div className="uploader-container">
      <input
        type="file"
        ref={fileInputRef}
        onChange={handleFileInputChange}
        accept="image/jpeg,image/png,image/tiff,image/webp,.jpg,.jpeg,.png,.tif,.tiff"
        style={{ display: 'none' }}
        id="farmwatch-file-input"
        disabled={disabled}
      />

      {!previewUrl ? (
        <div
          className={`dropzone ${dragOver ? 'dropzone-active' : ''} ${disabled ? 'dropzone-disabled' : ''}`}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onClick={() => !disabled && fileInputRef.current?.click()}
          role="button"
          tabIndex={0}
          onKeyDown={(e) => {
            if (e.key === 'Enter' || e.key === ' ') {
              fileInputRef.current?.click();
            }
          }}
          aria-label="Upload crop image"
        >
          <div className="dropzone-icon">
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
              <polyline points="17 8 12 3 7 8" />
              <line x1="12" y1="3" x2="12" y2="15" />
            </svg>
          </div>
          <h3 className="dropzone-title">Upload Crop Photo</h3>
          <p className="dropzone-hint">
            Drag and drop your maize canopy image here, or <span>browse files</span>
          </p>
          <div className="dropzone-meta">
            <span>Supports JPEG, PNG, TIFF</span>
            <span>Max size: 10 MB</span>
          </div>
        </div>
      ) : (
        <div className="preview-card">
          <div className="preview-media-wrapper">
            <img src={previewUrl} alt="Crop inspection preview" className="preview-img" />
          </div>
          <div className="preview-details">
            <div className="preview-file-info">
              <span className="preview-filename">{selectedFile.name}</span>
              <span className="preview-filesize">({(selectedFile.size / 1024).toFixed(0)} KB)</span>
            </div>
            {!disabled && (
              <button
                type="button"
                className="btn-remove"
                onClick={handleReset}
                title="Remove this image and choose another"
                aria-label="Remove selected image"
              >
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <line x1="18" y1="6" x2="6" y2="18"></line>
                  <line x1="6" y1="6" x2="18" y2="18"></line>
                </svg>
                Replace Photo
              </button>
            )}
          </div>
        </div>
      )}

      {errorMessage && (
        <div className="alert-error" role="alert">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="12" cy="12" r="10"></circle>
            <line x1="12" y1="8" x2="12" y2="12"></line>
            <line x1="12" y1="16" x2="12.01" y2="16"></line>
          </svg>
          <span>{errorMessage}</span>
        </div>
      )}
    </div>
  );
}
