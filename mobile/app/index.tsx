import { View, Text, StyleSheet, TouchableOpacity } from "react-native";
import { router } from "expo-router";

/**
 * Home screen — entry point of the app.
 * Provides navigation to the camera (translation) screen.
 */
export default function HomeScreen() {
  return (
    <View style={styles.container}>
      <View style={styles.hero}>
        <Text style={styles.title}>Eye Hear U</Text>
        <Text style={styles.subtitle}>
          ASL to English, one sign at a time
        </Text>
      </View>

      <View style={styles.actions}>
        <TouchableOpacity
          style={styles.primaryButton}
          onPress={() => router.push("/camera")}
        >
          <Text style={styles.primaryButtonText}>Start Translating</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={styles.secondaryButton}
          onPress={() => router.push("/history")}
        >
          <Text style={styles.secondaryButtonText}>View History</Text>
        </TouchableOpacity>
      </View>

      <Text style={styles.footer}>
        Point your camera at an ASL sign to see the English translation.
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#F8F9FA",
    alignItems: "center",
    justifyContent: "center",
    padding: 24,
  },
  hero: {
    alignItems: "center",
    marginBottom: 48,
  },
  title: {
    fontSize: 36,
    fontWeight: "bold",
    color: "#4A90D9",
    marginBottom: 8,
  },
  subtitle: {
    fontSize: 18,
    color: "#666",
    textAlign: "center",
  },
  actions: {
    width: "100%",
    gap: 16,
    marginBottom: 32,
  },
  primaryButton: {
    backgroundColor: "#4A90D9",
    paddingVertical: 16,
    borderRadius: 12,
    alignItems: "center",
  },
  primaryButtonText: {
    color: "#fff",
    fontSize: 18,
    fontWeight: "600",
  },
  secondaryButton: {
    backgroundColor: "#E8EDF2",
    paddingVertical: 16,
    borderRadius: 12,
    alignItems: "center",
  },
  secondaryButtonText: {
    color: "#4A90D9",
    fontSize: 18,
    fontWeight: "600",
  },
  footer: {
    color: "#999",
    fontSize: 14,
    textAlign: "center",
  },
});
