from pathlib import Path
import io

from PIL import Image, ImageDraw
import requests
import streamlit as st


CLASS_NAMES = {
    0: "MILCO",
    1: "NOMBO",
}

API_URL = "http://127.0.0.1:8000/predict"
IMAGE_DIR = Path("data/processed/yolo26n/images/test")
LABEL_DIR = Path("data/processed/yolo26n/labels/test")

PREDICTION_COLOR = "red"
GROUND_TRUTH_COLOR = "lime"


def yolo_to_xyxy(values, image_width, image_height):
    class_id, x_center, y_center, width, height = values

    x_center *= image_width
    y_center *= image_height
    width *= image_width
    height *= image_height

    return {
        "class_id": int(class_id),
        "class_name": CLASS_NAMES[int(class_id)],
        "x1": x_center - width / 2,
        "y1": y_center - height / 2,
        "x2": x_center + width / 2,
        "y2": y_center + height / 2,
    }


def read_yolo_labels(label_text, image_width, image_height):
    labels = []

    for line in label_text.splitlines():
        if line.strip():
            values = [float(value) for value in line.split()]
            labels.append(yolo_to_xyxy(values, image_width, image_height))

    return labels


def read_label_file(path, image_width, image_height):
    if not path.exists():
        return []

    return read_yolo_labels(
        path.read_text(encoding="utf-8"),
        image_width,
        image_height,
    )


def draw_box(draw, box, color, label):
    x1 = float(box["x1"])
    y1 = float(box["y1"])
    x2 = float(box["x2"])
    y2 = float(box["y2"])

    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
    draw.text((x1 + 3, max(0, y1 - 16)), label, fill=color)


def prediction_boxes(detections):
    boxes = []

    for detection in detections:
        box = detection["box"]
        confidence = float(detection["confidence"])

        boxes.append(
            {
                "class_name": detection["class_name"],
                "confidence": confidence,
                "x1": float(box["x1"]),
                "y1": float(box["y1"]),
                "x2": float(box["x2"]),
                "y2": float(box["y2"]),
            }
        )

    return boxes


def render_overlay(image, predictions, ground_truth):
    output = image.copy().convert("RGB")
    draw = ImageDraw.Draw(output)

    for box in ground_truth:
        draw_box(
            draw,
            box,
            GROUND_TRUTH_COLOR,
            f"GT: {box['class_name']}",
        )

    for box in predictions:
        draw_box(
            draw,
            box,
            PREDICTION_COLOR,
            f"{box['class_name']} {box['confidence']:.2f}",
        )

    return output


def call_api(image_name, image):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)

    response = requests.post(
        API_URL,
        files={"file": (image_name, buffer, "image/jpeg")},
        timeout=60,
    )
    response.raise_for_status()

    return response.json()


def local_test_images():
    if not IMAGE_DIR.exists():
        return []

    return sorted(IMAGE_DIR.glob("*.jpg"))


def main():
    st.set_page_config(
        page_title="Sonar Mine Detection Demo",
        layout="wide",
    )

    st.title("Side-Scan Sonar Object Detection Demo")
    st.write(
        "This demo calls the local FastAPI `/predict` endpoint and "
        "visualizes predicted MILCO/NOMBO boxes together with "
        "ground-truth annotations."
    )

    image_paths = local_test_images()

    if not image_paths:
        st.warning(
            "No local test images found in "
            "`data/processed/yolo26n/images/test`."
        )
        st.stop()

    selected = st.sidebar.selectbox(
        "Test image",
        image_paths,
        format_func=lambda path: path.name,
    )

    image = Image.open(selected).convert("RGB")
    image_name = selected.name

    label_path = LABEL_DIR / f"{selected.stem}.txt"
    ground_truth = read_label_file(
        label_path,
        image.width,
        image.height,
    )

    col_original, col_result = st.columns(2)

    with col_original:
        st.subheader("Original image")
        st.image(image, use_container_width=True)

    if st.button("Predict", type="primary"):
        result = call_api(image_name, image)
        predictions = prediction_boxes(result["detections"])
        overlay = render_overlay(image, predictions, ground_truth)

        with col_result:
            st.subheader("Prediction vs. ground truth")
            st.image(overlay, use_container_width=True)
            st.caption("Red = prediction, green = ground truth")

        st.subheader("Predictions")
        st.dataframe(predictions, use_container_width=True, hide_index=True)

        st.subheader("Ground truth")
        st.dataframe(ground_truth, use_container_width=True, hide_index=True)
    else:
        with col_result:
            st.subheader("Prediction vs. ground truth")
            st.info("Click Predict to run the local FastAPI model.")


if __name__ == "__main__":
    main()
