/**
 * FarmWatch API Client Service
 */

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';

/**
 * Checks API server health and model status
 */
export async function getHealthStatus() {
  try {
    const res = await fetch(`${API_BASE_URL}/health`);
    if (!res.ok) {
      throw new Error(`Server returned HTTP ${res.status}`);
    }
    return await res.json();
  } catch (err) {
    throw new Error(`Backend offline or unreachable (${err.message})`);
  }
}

/**
 * Fetches class metadata definitions
 */
export async function getClassDefinitions() {
  try {
    const res = await fetch(`${API_BASE_URL}/api/classes`);
    if (!res.ok) {
      throw new Error(`Failed to load classes: HTTP ${res.status}`);
    }
    return await res.json();
  } catch (err) {
    throw new Error(`Could not fetch class metadata: ${err.message}`);
  }
}

/**
 * Fetches a representative sample image from the test set
 * @param {string} className
 */
export async function fetchSampleImage(className) {
  const res = await fetch(`${API_BASE_URL}/api/sample/${className}`);
  if (!res.ok) {
    throw new Error(`Could not fetch sample image for ${className}`);
  }
  const blob = await res.blob();
  return new File([blob], `sample_${className}.jpg`, { type: 'image/jpeg' });
}

/**
 * Submits an image file for AI classification
 * @param {File|Blob} file 
 */
export async function predictCropImage(file) {
  const formData = new FormData();
  formData.append('file', file);

  try {
    // Note: Do NOT manually set Content-Type header so browser generates multipart boundary
    const res = await fetch(`${API_BASE_URL}/api/predict`, {
      method: 'POST',
      body: formData,
    });

    const data = await res.json().catch(() => null);

    if (!res.ok) {
      const errorMsg = data?.detail || `Prediction failed with HTTP ${res.status}`;
      throw new Error(errorMsg);
    }

    return data;
  } catch (err) {
    if (err.message.includes('Failed to fetch')) {
      throw new Error('Cannot connect to FarmWatch backend server (is it running on port 8000?)');
    }
    throw err;
  }
}
