## Mobile App (`mobile/`) – Eye Hear U (React Native / Expo)

This folder contains the **React Native (Expo)** mobile app for Eye Hear U.  
The app:
- Captures ASL signs via the device camera,
- Sends images to the backend for prediction,
- Displays the predicted English label + confidence, and
- Uses text-to-speech (TTS) to read the prediction aloud.

---

### Entry Points & Routing

- `app/_layout.tsx`  
  - Configures navigation with `expo-router` (`Stack` navigator).  
  - Screens:
    - `index`  → `app/index.tsx` (Home),
    - `camera` → `app/camera.tsx` (Translate),
    - `history` → `app/history.tsx` (Translation history UI).

- `app/index.tsx`  
  - Home screen with:
    - **Start Translating** → navigates to `/camera`.
    - **View History** → navigates to `/history`.

---

### Camera & Prediction Flow

- `app/camera.tsx` – **core screen**.

High-level flow:
1. Requests **camera permission** via `useCameraPermissions()` from `expo-camera`.
2. Renders a `CameraView` with a live preview (front-facing camera).
3. When the user taps **Capture**:
   - Takes a picture with `takePictureAsync`.
   - Calls the shared API client `predictSign` from `services/api.ts`.
4. Updates state with the predicted `sign` and `confidence`.
5. If `confidence > 0.5`, calls `Speech.speak(sign)` from `expo-speech`.
6. Renders:
   - Big predicted sign text,
   - Confidence percentage,
   - “Speak Again” button to re-trigger TTS.

**Important:**  
The screen **does not** hard-code the backend URL. All networking goes through `services/api.ts`.

---

### Backend API Client

- `services/api.ts`

Exports:
- `predictSign(imageUri: string)`  
  - Wraps the image in `FormData` under `file`.  
  - POSTs to `/api/v1/predict`.  
  - Returns `{ sign, confidence, top_k, message? }`.

- `checkHealth()`  
  - GETs `/health` to see if the backend is reachable.

**TODO (Mobile):**  
Update `API_BASE_URL` in `services/api.ts` for your environment:
- Simulator on same machine: `http://localhost:8000`.  
- Physical device on WiFi: `http://<your-laptop-LAN-IP>:8000`.

---

### History Screen

- `app/history.tsx`  
  - Currently shows **placeholder** history data.  
  - Intended to later:
    - Pull real history from local storage / Firestore,
    - Show sign, confidence, timestamp,
    - Allow “correct/wrong” feedback on each entry.

---

### How to Run the App

1. Install dependencies (after Node 20.x is installed):

   ```bash
   cd mobile
   npm install
   ```

2. Start the Expo dev server:

   ```bash
   npx expo start
   ```

3. Run the app:
   - Scan the QR code with the **Expo Go** app on your iPhone, or  
   - Press `i` to open the iOS simulator.

4. Make sure the backend is running (see `backend/README.md`).

---

### Who Should Work Here & Typical Tasks

- **Maria (iOS / Frontend):**
  - Modify screen layouts & styling.
  - Improve UX (loading states, error messages).
  - Implement history UI and feedback buttons.

- **Everyone:**
  - Can use this app to test changes to the model/backend end-to-end.

