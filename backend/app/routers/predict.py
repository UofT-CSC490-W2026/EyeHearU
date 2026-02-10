"""
Prediction endpoints.
Accepts an image (upload or base64) and returns the predicted ASL sign label.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from app.schemas.prediction import PredictionResponse

router = APIRouter()


@router.post("/predict", response_model=PredictionResponse)
async def predict_sign(file: UploadFile = File(...)):
    """
    Accept an image file and return the predicted ASL sign.

    Flow:
      1. Receive image from mobile client
      2. Preprocess (resize, normalize)
      3. Run through the loaded CNN-Transformer model
      4. Return top-k predictions with confidence scores

    Currently returns a placeholder while the model is being trained.
    """
    # Validate file type
    if file.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Use JPEG, PNG, or WebP.",
        )

    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    # ---------------------------------------------------------------
    # TODO: Replace placeholder with actual model inference
    # ---------------------------------------------------------------
    # from app.services.model_service import predict
    # from app.services.preprocessing import preprocess_image
    #
    # image = preprocess_image(contents)
    # result = predict(app.state.model, image)
    # ---------------------------------------------------------------

    return PredictionResponse(
        sign="hello",
        confidence=0.0,
        top_k=[
            {"sign": "hello", "confidence": 0.0},
        ],
        message="Placeholder response — model not yet loaded.",
    )
