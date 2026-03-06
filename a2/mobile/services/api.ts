/**
 * API client for communicating with the Eye Hear U backend.
 *
 * In development, the backend runs at localhost:8000.
 * In production, this will point to a deployed server.
 */

// TODO: Use environment variable or config file
const API_BASE_URL = __DEV__
  ? "http://localhost:8000"
  : "https://api.eyehearu.app";

export interface PredictionResult {
  sign: string;
  confidence: number;
  top_k: { sign: string; confidence: number }[];
  message?: string;
}

/**
 * Send an image to the backend for ASL sign prediction.
 */
export async function predictSign(imageUri: string): Promise<PredictionResult> {
  const formData = new FormData();
  formData.append("file", {
    uri: imageUri,
    name: "sign.jpg",
    type: "image/jpeg",
  } as any);

  const response = await fetch(`${API_BASE_URL}/api/v1/predict`, {
    method: "POST",
    body: formData,
    headers: {
      "Content-Type": "multipart/form-data",
    },
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`Prediction failed (${response.status}): ${text}`);
  }

  return response.json();
}

/**
 * Check if the backend is reachable and ready.
 */
export async function checkHealth(): Promise<boolean> {
  try {
    const response = await fetch(`${API_BASE_URL}/health`);
    const data = await response.json();
    return data.status === "ok";
  } catch {
    return false;
  }
}
