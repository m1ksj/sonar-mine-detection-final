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


class BoxXYXY(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class Detection(BaseModel):
    class_name: str
    confidence: float
    box: BoxXYXY


class PredictionResponse(BaseModel):
    image_name: str
    detections: list[Detection]


app = FastAPI(
    title="Sonar Mine Detection API",
    description=(
        "Upload one JPEG or PNG side-scan sonar image. The API returns "
        "MILCO/NOMBO detections as class names, confidence scores and "
        "pixel-space xyxy bounding boxes. Image validation and model "
        "preprocessing are handled server-side."
    ),
    version="1.0.0",
    redoc_url=None,
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
                "Deployment model missing at "
                "models/final/yolo26n_selected.pt."
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


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Detect MILCO and NOMBO objects",
    description=(
        "Request: multipart/form-data with one field named `file` containing "
        "a JPEG or PNG image. Response: JSON with the image name and a list "
        "of detections. Each detection contains a class name, confidence and "
        "pixel-space bounding box coordinates `x1`, `y1`, `x2`, `y2`."
    ),
)
async def predict(file: UploadFile = File(...)):
    if file.content_type not in {"image/jpeg", "image/png"}:
        raise HTTPException(
            status_code=400,
            detail="Only JPEG and PNG images are supported.",
        )

    suffix = Path(file.filename or "image.jpg").suffix or ".jpg"

    with NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_path = Path(temp_file.name)
        temp_file.write(await file.read())

    try:
        validate_image(temp_path)

        detector = get_model()
        result = detector.predict(str(temp_path), verbose=False)[0]

        detections = []
        for box in result.boxes:
            class_id = int(box.cls.item())
            x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
            detections.append(
                Detection(
                    class_name=CLASS_NAMES.get(class_id, "unknown"),
                    confidence=float(box.conf.item()),
                    box=BoxXYXY(x1=x1, y1=y1, x2=x2, y2=y2),
                )
            )

        return PredictionResponse(
            image_name=file.filename or "uploaded_image",
            detections=detections,
        )
    finally:
        temp_path.unlink(missing_ok=True)
