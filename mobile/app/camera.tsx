import { useState, useRef } from "react";
import { View, Text, StyleSheet, TouchableOpacity, Image } from "react-native";
import { CameraView, useCameraPermissions } from "expo-camera";
import * as Speech from "expo-speech";
import { predictSign, type PredictionResult } from "../services/api";

/**
 * Camera screen — captures images and sends them to the backend for prediction.
 *
 * Flow:
 *   1. User sees live camera preview
 *   2. User taps "Capture" to take a photo
 *   3. Image is sent to POST /api/v1/predict
 *   4. Prediction is displayed on screen
 *   5. Optional: text-to-speech reads the prediction aloud
 */
export default function CameraScreen() {
  const [permission, requestPermission] = useCameraPermissions();
  const [prediction, setPrediction] = useState<string | null>(null);
  const [confidence, setConfidence] = useState<number>(0);
  const [isLoading, setIsLoading] = useState(false);
  const [capturedUri, setCapturedUri] = useState<string | null>(null);
  const cameraRef = useRef<CameraView>(null);

  // Handle permissions
  if (!permission) {
    return <View style={styles.container} />;
  }

  if (!permission.granted) {
    return (
      <View style={styles.permissionContainer}>
        <Text style={styles.permissionText}>
          Camera access is required to recognize ASL signs.
        </Text>
        <TouchableOpacity style={styles.button} onPress={requestPermission}>
          <Text style={styles.buttonText}>Grant Permission</Text>
        </TouchableOpacity>
      </View>
    );
  }

  const captureAndPredict = async () => {
    if (!cameraRef.current || isLoading) return;

    setIsLoading(true);
    try {
      // 1) Capture photo from the live camera preview
      const photo = await cameraRef.current.takePictureAsync({
        quality: 0.8,
        base64: false,
      });

      if (!photo) return;
      setCapturedUri(photo.uri);

      // 2) Send image to backend for prediction via shared API client
      const result: PredictionResult = await predictSign(photo.uri);
      setPrediction(result.sign);
      setConfidence(result.confidence);

      // Text-to-speech
      if (result.sign && result.confidence > 0.5) {
        Speech.speak(result.sign, { language: "en-US" });
      }
    } catch (error) {
      console.error("Prediction failed:", error);
      setPrediction("Error — check connection");
      setConfidence(0);
    } finally {
      setIsLoading(false);
    }
  };

  const speakPrediction = () => {
    if (prediction) {
      Speech.speak(prediction, { language: "en-US" });
    }
  };

  return (
    <View style={styles.container}>
      {/* Camera preview */}
      <CameraView
        ref={cameraRef}
        style={styles.camera}
        facing="front"
      >
        {/* Overlay for captured image */}
        {capturedUri && (
          <Image source={{ uri: capturedUri }} style={styles.capturedOverlay} />
        )}
      </CameraView>

      {/* Prediction result */}
      <View style={styles.resultContainer}>
        {prediction ? (
          <>
            <Text style={styles.predictionLabel}>Detected Sign:</Text>
            <Text style={styles.predictionText}>{prediction}</Text>
            <Text style={styles.confidenceText}>
              Confidence: {(confidence * 100).toFixed(1)}%
            </Text>
            <TouchableOpacity
              style={styles.speakButton}
              onPress={speakPrediction}
            >
              <Text style={styles.speakButtonText}>Speak Again</Text>
            </TouchableOpacity>
          </>
        ) : (
          <Text style={styles.instructionText}>
            Sign in front of the camera and tap Capture
          </Text>
        )}
      </View>

      {/* Capture button */}
      <TouchableOpacity
        style={[styles.captureButton, isLoading && styles.captureButtonDisabled]}
        onPress={captureAndPredict}
        disabled={isLoading}
      >
        <Text style={styles.captureButtonText}>
          {isLoading ? "Processing..." : "Capture"}
        </Text>
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#000",
  },
  camera: {
    flex: 1,
  },
  capturedOverlay: {
    position: "absolute",
    top: 8,
    right: 8,
    width: 80,
    height: 80,
    borderRadius: 8,
    borderWidth: 2,
    borderColor: "#fff",
  },
  resultContainer: {
    backgroundColor: "#fff",
    paddingVertical: 20,
    paddingHorizontal: 24,
    alignItems: "center",
    minHeight: 120,
    justifyContent: "center",
  },
  predictionLabel: {
    fontSize: 14,
    color: "#999",
    marginBottom: 4,
  },
  predictionText: {
    fontSize: 32,
    fontWeight: "bold",
    color: "#4A90D9",
    marginBottom: 4,
  },
  confidenceText: {
    fontSize: 14,
    color: "#666",
    marginBottom: 8,
  },
  instructionText: {
    fontSize: 16,
    color: "#999",
    textAlign: "center",
  },
  captureButton: {
    backgroundColor: "#4A90D9",
    paddingVertical: 18,
    alignItems: "center",
  },
  captureButtonDisabled: {
    backgroundColor: "#A0C4E8",
  },
  captureButtonText: {
    color: "#fff",
    fontSize: 20,
    fontWeight: "bold",
  },
  speakButton: {
    backgroundColor: "#E8EDF2",
    paddingVertical: 8,
    paddingHorizontal: 16,
    borderRadius: 8,
  },
  speakButtonText: {
    color: "#4A90D9",
    fontWeight: "600",
  },
  permissionContainer: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    padding: 24,
    backgroundColor: "#F8F9FA",
  },
  permissionText: {
    fontSize: 16,
    color: "#666",
    textAlign: "center",
    marginBottom: 16,
  },
  button: {
    backgroundColor: "#4A90D9",
    paddingVertical: 12,
    paddingHorizontal: 24,
    borderRadius: 8,
  },
  buttonText: {
    color: "#fff",
    fontSize: 16,
    fontWeight: "600",
  },
});
