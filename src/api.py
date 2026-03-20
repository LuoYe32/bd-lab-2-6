from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
from PIL import Image
import io

MODEL_PATH = Path("artifacts/model.joblib")

CLASS_NAMES = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot",
}


class PredictRequest(BaseModel):
    pixels: Optional[List[float]] = Field(
        default=None,
        description="Length 784 array"
    )
    fill: Optional[float] = Field(
        default=None,
        description="Fill all pixels with one value"
    )
    random_seed: Optional[int] = Field(
        default=None,
        description="Generate deterministic random pixels"
    )


class PredictResponse(BaseModel):
    class_id: int
    class_name: str
    proba: List[float]


app = FastAPI(title="Fashion-MNIST Classic ML API")
_model = None


def _load_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. Run training or dvc pull artifacts."
            )
        _model = joblib.load(MODEL_PATH)
    return _model


def _predict_array(x: np.ndarray):

    model = _load_model()

    if x.max() > 1.5:
        x = x / 255.0

    X = x.reshape(1, -1)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        class_id = int(np.argmax(proba))
    else:
        class_id = int(model.predict(X)[0])
        proba = np.zeros(10)
        proba[class_id] = 1.0

    return {
        "class_id": class_id,
        "class_name": CLASS_NAMES.get(class_id, str(class_id)),
        "proba": [float(p) for p in proba],
    }


@app.get("/health")
def health():
    ok = MODEL_PATH.exists()
    return {"status": "ok", "model_present": ok}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):

    provided = [
        req.pixels is not None,
        req.fill is not None,
        req.random_seed is not None,
    ]

    if sum(provided) == 0:
        raise HTTPException(
            status_code=400,
            detail="One of pixels, fill or random_seed must be provided"
        )

    if sum(provided) > 1:
        raise HTTPException(
            status_code=400,
            detail="Only one of pixels, fill or random_seed can be provided"
        )

    if req.pixels is not None:

        if not isinstance(req.pixels, list):
            raise HTTPException(status_code=400, detail="pixels must be a list")

        if len(req.pixels) != 784:
            raise HTTPException(
                status_code=400,
                detail="pixels must contain exactly 784 values"
            )

        try:
            x = np.array(req.pixels, dtype=np.float32)
        except Exception:
            raise HTTPException(
                status_code=400,
                detail="pixels must contain numeric values"
            )

        if not np.isfinite(x).all():
            raise HTTPException(
                status_code=400,
                detail="pixels must not contain NaN or infinite values"
            )

        if x.min() < 0:
            raise HTTPException(
                status_code=400,
                detail="pixels must be non-negative"
            )

    elif req.fill is not None:

        try:
            fill_value = float(req.fill)
        except Exception:
            raise HTTPException(
                status_code=400,
                detail="fill must be numeric"
            )

        if not np.isfinite(fill_value):
            raise HTTPException(
                status_code=400,
                detail="fill must be a finite number"
            )

        if fill_value < 0:
            raise HTTPException(
                status_code=400,
                detail="fill must be non-negative"
            )

        x = np.full((784,), fill_value, dtype=np.float32)

    elif req.random_seed is not None:

        try:
            seed = int(req.random_seed)
        except Exception:
            raise HTTPException(
                status_code=400,
                detail="random_seed must be an integer"
            )

        if seed < 0:
            raise HTTPException(
                status_code=400,
                detail="random_seed must be non-negative"
            )

        rng = np.random.default_rng(seed)
        x = rng.random(784)

    return _predict_array(x)


@app.post("/predict/image", response_model=PredictResponse)
async def predict_image(file: UploadFile = File(...)):

    if file.content_type not in {"image/png", "image/jpeg", "image/jpg", "image/bmp"}:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Allowed: png, jpg, jpeg, bmp"
        )

    contents = await file.read()

    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Empty file")

    if len(contents) > 5 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large")

    try:
        image = Image.open(io.BytesIO(contents))
        image = image.convert("L")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    if image.width > 4096 or image.height > 4096:
        raise HTTPException(status_code=400, detail="Image resolution too large")

    image = image.resize((28, 28))

    arr = np.array(image, dtype=np.float32).flatten()

    if not np.isfinite(arr).all():
        raise HTTPException(status_code=400, detail="Invalid pixel values")

    return _predict_array(arr)


@app.get("/predict/random", response_model=PredictResponse)
def predict_random(seed: int | None = None):

    if seed is not None:
        if seed < 0:
            raise HTTPException(status_code=400, detail="seed must be non-negative")
        rng = np.random.default_rng(seed)
        x = rng.random(784)
    else:
        x = np.random.random(784)

    return _predict_array(x)