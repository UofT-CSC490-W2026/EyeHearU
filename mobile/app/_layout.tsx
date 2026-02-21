import { Stack } from "expo-router";
import { StatusBar } from "expo-status-bar";

/**
 * Root layout for the app.
 * Uses expo-router for file-based routing.
 */
export default function RootLayout() {
  return (
    <>
      <StatusBar style="dark" />
      <Stack
        screenOptions={{
          headerStyle: { backgroundColor: "#4A90D9" },
          headerTintColor: "#fff",
          headerTitleStyle: { fontWeight: "bold" },
        }}
      >
        <Stack.Screen name="index" options={{ title: "Eye Hear U" }} />
        <Stack.Screen name="camera" options={{ title: "Translate" }} />
        <Stack.Screen name="history" options={{ title: "History" }} />
      </Stack>
    </>
  );
}
