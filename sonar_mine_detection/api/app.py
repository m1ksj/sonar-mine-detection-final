from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


CLASS_NAMES = {
    0: "MILCO",
    1: "NOMBO",
}

MODEL_PATH = Path("models/final/yolo26n_selected.pt")


class Detection(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    box_xyxy: list[float]


class PredictionResponse(BaseModel):
    image_name: str
    detections: list[Detection]


app = FastAPI(
    title="Sonar Mine Detection API",
    description=(
        "Object detection API for MILCO and NOMBO targets "
        "in side-scan sonar images."
    ),
    version="0.1.0",
)

model = None


def get_model():
    global model

    if YOLO is None:
        raise HTTPException(
            status_code=500,
            detail="Ultralytics is not installed.",
        )

    if not MODEL_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail=(
                "Deployment model not found. Run scripts/setup_model.py "
                "after the final model artifact is available."
            ),
        )

    if model is None:
        model = YOLO(str(MODEL_PATH))

    return model


def validate_image(file_path):
    try:
        with Image.open(file_path) as image:
            image.verify()
    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is not a valid image.",
        ) from error


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if file.content_type not in {"image/jpeg", "image/png"}:
        raise HTTPException(
            status_code=400,
            detail="Only JPEG and PNG images are supported.",
        )

    suffix = Path(file.filename or "image.jpg").suffix

    with NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_path = Path(temp_file.name)
        temp_file.write(await file.read())

    validate_image(temp_path)

    detector = get_model()
    results = detector.predict(str(temp_path), verbose=False)[0]

    detections = []

    for box in results.boxes:
        class_id = int(box.cls.item())
        detections.append(
            Detection(
                class_id=class_id,
                class_name=CLASS_NAMES.get(class_id, "unknown"),
                confidence=float(box.conf.item()),
                box_xyxy=[
                    float(value)
                    for value in box.xyxy[0].tolist()
                ],
            )
        )

    temp_path.unlink(missing_ok=True)

    return PredictionResponse(
        image_name=file.filename or "uploaded_image",
        detections=detections,
    )
