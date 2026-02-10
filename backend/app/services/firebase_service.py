"""
Firebase integration service.

Uses Firebase/Firestore for:
  - Translation history (per-session or per-user)
  - Usage analytics
  - (Future) User accounts if login is added

Setup:
  1. Create a Firebase project at https://console.firebase.google.com
  2. Generate a service account key (JSON)
  3. Save it as backend/firebase-credentials.json
  4. Set FIREBASE_PROJECT_ID in .env
"""

import firebase_admin
from firebase_admin import credentials, firestore
from app.config import get_settings

_db = None


def init_firebase():
    """Initialize Firebase Admin SDK (call once at app startup)."""
    global _db
    settings = get_settings()

    if not firebase_admin._apps:
        cred = credentials.Certificate(settings.firebase_credentials_path)
        firebase_admin.initialize_app(cred, {
            "projectId": settings.firebase_project_id,
        })

    _db = firestore.client()
    return _db


def get_db():
    """Return the Firestore client, initializing if needed."""
    global _db
    if _db is None:
        return init_firebase()
    return _db


# ---------------------------------------------------------------
# Translation History
# ---------------------------------------------------------------

def save_translation(session_id: str, data: dict):
    """
    Save a translation record to Firestore.

    Collection: translations
    Document fields:
      - session_id: str
      - image_url: str (optional, if we store images)
      - predicted_sign: str
      - confidence: float
      - timestamp: datetime
    """
    db = get_db()
    db.collection("translations").add({
        "session_id": session_id,
        **data,
    })


def get_translation_history(session_id: str, limit: int = 50):
    """Retrieve recent translations for a session."""
    db = get_db()
    docs = (
        db.collection("translations")
        .where("session_id", "==", session_id)
        .order_by("timestamp", direction=firestore.Query.DESCENDING)
        .limit(limit)
        .stream()
    )
    return [doc.to_dict() for doc in docs]
