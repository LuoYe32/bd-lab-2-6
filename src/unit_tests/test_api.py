from pathlib import Path
import io

import numpy as np
import joblib
from PIL import Image
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

import src.api as api


def create_dummy_model(tmp_path):
    Path("artifacts").mkdir(parents=True, exist_ok=True)

    X = np.random.random((50, 784)).astype("float32")
    y = np.array([i % 10 for i in range(50)], dtype="int64")

    model = LogisticRegression(max_iter=200, solver="lbfgs")
    model.fit(X, y)

    joblib.dump(model, "artifacts/model.joblib")


def reset_model_cache():
    api._model = None


def test_health_endpoint(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    create_dummy_model(tmp_path)
    reset_model_cache()

    client = TestClient(api.app)

    r = client.get("/health")

    assert r.status_code == 200

    data = r.json()

    assert data["status"] == "ok"
    assert data["model_present"] is True


def test_predict_fill(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    create_dummy_model(tmp_path)
    reset_model_cache()

    client = TestClient(api.app)

    r = client.post("/predict", json={"fill": 0})

    assert r.status_code == 200

    data = r.json()

    assert "class_id" in data
    assert "class_name" in data
    assert "proba" in data
    assert len(data["proba"]) == 10


def test_predict_random(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    create_dummy_model(tmp_path)
    reset_model_cache()

    client = TestClient(api.app)

    r = client.get("/predict/random")

    assert r.status_code == 200

    data = r.json()

    assert "class_id" in data
    assert len(data["proba"]) == 10


def test_predict_image(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    create_dummy_model(tmp_path)
    reset_model_cache()

    client = TestClient(api.app)

    img = Image.fromarray((np.random.rand(28, 28) * 255).astype("uint8"))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    response = client.post(
        "/predict/image",
        files={"file": ("test.png", buf, "image/png")},
    )

    assert response.status_code == 200

    data = response.json()

    assert "class_id" in data
    assert "class_name" in data
    assert len(data["proba"]) == 10