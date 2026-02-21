import { View, Text, StyleSheet, FlatList } from "react-native";

/**
 * History screen — shows recent translation results.
 *
 * TODO: Fetch from Firebase or local storage.
 * For now, this is a placeholder UI.
 */

// Placeholder data
const PLACEHOLDER_HISTORY = [
  { id: "1", sign: "hello", confidence: 0.95, timestamp: "2 min ago" },
  { id: "2", sign: "thank you", confidence: 0.88, timestamp: "5 min ago" },
  { id: "3", sign: "water", confidence: 0.72, timestamp: "8 min ago" },
];

export default function HistoryScreen() {
  return (
    <View style={styles.container}>
      {PLACEHOLDER_HISTORY.length === 0 ? (
        <View style={styles.empty}>
          <Text style={styles.emptyText}>No translations yet.</Text>
          <Text style={styles.emptySubtext}>
            Go to the camera to start translating ASL signs.
          </Text>
        </View>
      ) : (
        <FlatList
          data={PLACEHOLDER_HISTORY}
          keyExtractor={(item) => item.id}
          contentContainerStyle={styles.list}
          renderItem={({ item }) => (
            <View style={styles.card}>
              <Text style={styles.sign}>{item.sign}</Text>
              <View style={styles.meta}>
                <Text style={styles.confidence}>
                  {(item.confidence * 100).toFixed(0)}% confidence
                </Text>
                <Text style={styles.timestamp}>{item.timestamp}</Text>
              </View>
            </View>
          )}
        />
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#F8F9FA",
  },
  list: {
    padding: 16,
    gap: 12,
  },
  card: {
    backgroundColor: "#fff",
    padding: 16,
    borderRadius: 12,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 3,
    elevation: 2,
  },
  sign: {
    fontSize: 24,
    fontWeight: "bold",
    color: "#4A90D9",
    marginBottom: 8,
  },
  meta: {
    flexDirection: "row",
    justifyContent: "space-between",
  },
  confidence: {
    color: "#666",
    fontSize: 14,
  },
  timestamp: {
    color: "#999",
    fontSize: 14,
  },
  empty: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    padding: 24,
  },
  emptyText: {
    fontSize: 18,
    color: "#666",
    marginBottom: 8,
  },
  emptySubtext: {
    fontSize: 14,
    color: "#999",
    textAlign: "center",
  },
});
