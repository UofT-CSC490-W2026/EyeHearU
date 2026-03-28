/**
 * Tests for the History screen (app/history.tsx).
 */

import React from "react";
import { render, screen, act } from "@testing-library/react-native";
import { Alert } from "react-native";

/* ------------------------------------------------------------------ */
/*  Mocks                                                              */
/* ------------------------------------------------------------------ */

// Mock expo-router's useFocusEffect to just call the callback immediately
jest.mock("expo-router", () => ({
  useFocusEffect: (cb: () => void) => {
    // eslint-disable-next-line react-hooks/rules-of-hooks
    const React = require("react");
    React.useEffect(() => {
      cb();
    }, []);
  },
}));

// Mock AsyncStorage
const mockAsyncStorage: Record<string, string> = {};
jest.mock("@react-native-async-storage/async-storage", () => ({
  __esModule: true,
  default: {
    getItem: jest.fn(async (key: string) => mockAsyncStorage[key] ?? null),
    setItem: jest.fn(async (key: string, value: string) => {
      mockAsyncStorage[key] = value;
    }),
    removeItem: jest.fn(async (key: string) => {
      delete mockAsyncStorage[key];
    }),
  },
}));

import AsyncStorage from "@react-native-async-storage/async-storage";
import HistoryScreen from "../app/history";

/* ------------------------------------------------------------------ */
/*  Helper: timeAgo is not exported, so we test it indirectly via      */
/*  rendered output. We also create a standalone copy for unit tests.  */
/* ------------------------------------------------------------------ */
function timeAgo(iso: string): string {
  const diff = Date.now() - new Date(iso).getTime();
  const mins = Math.floor(diff / 60_000);
  if (mins < 1) return "Just now";
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  const days = Math.floor(hrs / 24);
  return `${days}d ago`;
}

/* ------------------------------------------------------------------ */
/*  timeAgo unit tests                                                 */
/* ------------------------------------------------------------------ */
describe("timeAgo", () => {
  it('returns "Just now" for timestamps less than 1 minute ago', () => {
    const now = new Date().toISOString();
    expect(timeAgo(now)).toBe("Just now");
  });

  it('returns "Xm ago" for timestamps minutes ago', () => {
    const fiveMinAgo = new Date(Date.now() - 5 * 60_000).toISOString();
    expect(timeAgo(fiveMinAgo)).toBe("5m ago");
  });

  it('returns "Xh ago" for timestamps hours ago', () => {
    const threeHoursAgo = new Date(Date.now() - 3 * 60 * 60_000).toISOString();
    expect(timeAgo(threeHoursAgo)).toBe("3h ago");
  });

  it('returns "Xd ago" for timestamps days ago', () => {
    const twoDaysAgo = new Date(
      Date.now() - 2 * 24 * 60 * 60_000
    ).toISOString();
    expect(timeAgo(twoDaysAgo)).toBe("2d ago");
  });

  it("handles edge case at exactly 1 minute", () => {
    const oneMinAgo = new Date(Date.now() - 60_000).toISOString();
    expect(timeAgo(oneMinAgo)).toBe("1m ago");
  });

  it("handles edge case at exactly 1 hour", () => {
    const oneHourAgo = new Date(Date.now() - 60 * 60_000).toISOString();
    expect(timeAgo(oneHourAgo)).toBe("1h ago");
  });

  it("handles edge case at exactly 24 hours", () => {
    const oneDayAgo = new Date(Date.now() - 24 * 60 * 60_000).toISOString();
    expect(timeAgo(oneDayAgo)).toBe("1d ago");
  });
});

/* ------------------------------------------------------------------ */
/*  HistoryScreen component tests                                      */
/* ------------------------------------------------------------------ */
describe("HistoryScreen", () => {
  beforeEach(() => {
    jest.clearAllMocks();
    // Clear our mock storage
    Object.keys(mockAsyncStorage).forEach(
      (key) => delete mockAsyncStorage[key]
    );
  });

  it('renders "No translations yet" when history is empty', async () => {
    render(<HistoryScreen />);

    expect(
      await screen.findByText("No translations yet")
    ).toBeTruthy();
    expect(
      screen.getByText("Go to the camera to start translating ASL signs.")
    ).toBeTruthy();
  });

  it("renders history items when present", async () => {
    const historyData = [
      {
        id: "1",
        sign: "hello",
        confidence: 0.95,
        timestamp: new Date().toISOString(),
      },
      {
        id: "2",
        sign: "thank you",
        confidence: 0.88,
        timestamp: new Date(Date.now() - 5 * 60_000).toISOString(),
      },
    ];
    mockAsyncStorage["eyehearu_history"] = JSON.stringify(historyData);

    render(<HistoryScreen />);

    expect(await screen.findByText("hello")).toBeTruthy();
    expect(screen.getByText("thank you")).toBeTruthy();
    expect(screen.getByText("95%")).toBeTruthy();
    expect(screen.getByText("88%")).toBeTruthy();
  });

  it("renders the Clear History button when history is present", async () => {
    const historyData = [
      {
        id: "1",
        sign: "hello",
        confidence: 0.9,
        timestamp: new Date().toISOString(),
      },
    ];
    mockAsyncStorage["eyehearu_history"] = JSON.stringify(historyData);

    render(<HistoryScreen />);

    expect(await screen.findByText("Clear History")).toBeTruthy();
  });

  it("does not render Clear History button when history is empty", async () => {
    render(<HistoryScreen />);

    await screen.findByText("No translations yet");
    expect(screen.queryByText("Clear History")).toBeNull();
  });

  it("shows Alert when Clear History is pressed", async () => {
    const alertSpy = jest.spyOn(Alert, "alert");
    const historyData = [
      {
        id: "1",
        sign: "hello",
        confidence: 0.9,
        timestamp: new Date().toISOString(),
      },
    ];
    mockAsyncStorage["eyehearu_history"] = JSON.stringify(historyData);

    const { getByText } = render(<HistoryScreen />);

    const clearButton = await screen.findByText("Clear History");

    // Simulate press via the fireEvent import
    const { fireEvent } = require("@testing-library/react-native");
    fireEvent.press(clearButton);

    expect(alertSpy).toHaveBeenCalledWith(
      "Clear History",
      "Remove all saved translations?",
      expect.arrayContaining([
        expect.objectContaining({ text: "Cancel" }),
        expect.objectContaining({ text: "Clear", style: "destructive" }),
      ])
    );

    alertSpy.mockRestore();
  });

  it("handles AsyncStorage errors gracefully (sets empty history)", async () => {
    (AsyncStorage.getItem as jest.Mock).mockRejectedValueOnce(
      new Error("storage error")
    );

    render(<HistoryScreen />);

    // Should still render empty state
    expect(
      await screen.findByText("No translations yet")
    ).toBeTruthy();
  });

  it("renders hours-ago timestamps for items a few hours old", async () => {
    const historyData = [
      {
        id: "h1",
        sign: "water",
        confidence: 0.8,
        timestamp: new Date(Date.now() - 3 * 60 * 60_000).toISOString(),
      },
    ];
    mockAsyncStorage["eyehearu_history"] = JSON.stringify(historyData);

    render(<HistoryScreen />);

    expect(await screen.findByText("water")).toBeTruthy();
    expect(screen.getByText("3h ago")).toBeTruthy();
  });

  it("renders days-ago timestamps for items days old", async () => {
    const historyData = [
      {
        id: "d1",
        sign: "book",
        confidence: 0.75,
        timestamp: new Date(Date.now() - 2 * 24 * 60 * 60_000).toISOString(),
      },
    ];
    mockAsyncStorage["eyehearu_history"] = JSON.stringify(historyData);

    render(<HistoryScreen />);

    expect(await screen.findByText("book")).toBeTruthy();
    expect(screen.getByText("2d ago")).toBeTruthy();
  });

  it("clears history when Clear button is pressed in the Alert", async () => {
    const alertSpy = jest.spyOn(Alert, "alert");
    const historyData = [
      {
        id: "c1",
        sign: "yes",
        confidence: 0.9,
        timestamp: new Date().toISOString(),
      },
    ];
    mockAsyncStorage["eyehearu_history"] = JSON.stringify(historyData);

    render(<HistoryScreen />);

    const clearButton = await screen.findByText("Clear History");
    const { fireEvent } = require("@testing-library/react-native");
    fireEvent.press(clearButton);

    // Extract the "Clear" button's onPress callback from the Alert call
    const alertButtons = alertSpy.mock.calls[0][2] as Array<{
      text: string;
      onPress?: () => void | Promise<void>;
    }>;
    const clearAction = alertButtons.find((b) => b.text === "Clear");
    expect(clearAction).toBeDefined();

    // Invoke the clear handler
    await act(async () => {
      await clearAction!.onPress!();
    });

    // Verify AsyncStorage.removeItem was called
    expect(AsyncStorage.removeItem).toHaveBeenCalledWith("eyehearu_history");

    // After clearing, should show empty state
    expect(await screen.findByText("No translations yet")).toBeTruthy();

    alertSpy.mockRestore();
  });
});
