import io

import pytest
from PIL import Image

from app import create_app


class StubClassifier:
    def __init__(self, preds):
        self.preds = preds

    def predict(self, image, top_k):
        return self.preds[:top_k]


NEW = StubClassifier([("Treecko", 91.23456), ("Grovyle", 5.111), ("Sceptile", 1.0), ("Mudkip", 0.5)])
GEN1_NEW = StubClassifier([("Pikachu", 70.0), ("Raichu", 20.0), ("Pichu", 5.0)])
LEGACY = StubClassifier([("Pikachu", 80.0), ("Raichu", 10.0), ("MrMime", 5.0)])


def make_client(new_clf):
    app = create_app(new_clf=new_clf, legacy_clf=LEGACY, gen1_names={"Pikachu", "Raichu"},
                     api_key="test-key", legacy_trust_pct=55.0)
    return app.test_client()


@pytest.fixture
def client():
    return make_client(NEW)


def png_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), "yellow").save(buf, "PNG")
    return buf.getvalue()


def post(client, data, key="test-key", query=""):
    headers = {"X-Api-Key": key} if key else {}
    return client.post(f"/predict{query}", headers=headers,
                       data={"file": (io.BytesIO(data), "x.png")},
                       content_type="multipart/form-data")


def test_health_no_key(client):
    assert client.get("/health").status_code == 200


@pytest.mark.parametrize("key", [None, "wrong"])
def test_predict_requires_key(client, key):
    res = post(client, png_bytes(), key=key)
    assert res.status_code == 401


def test_predict_rejects_non_image(client):
    res = post(client, b"not an image")
    assert res.status_code == 400
    assert res.get_json()["error"] == "Could not read image file."


def test_predict_rejects_missing_file(client):
    res = client.post("/predict", headers={"X-Api-Key": "test-key"})
    assert res.status_code == 400


def test_predict_ok(client):
    res = post(client, png_bytes())
    assert res.status_code == 200
    body = res.get_json()
    assert body["model"] == "gen1-3"
    assert body["predictions"] == [
        {"name": "Treecko", "confidence": 91.23},
        {"name": "Grovyle", "confidence": 5.11},
        {"name": "Sceptile", "confidence": 1.0},
    ]


@pytest.mark.parametrize("query,expected", [("?top_k=1", 1), ("?top_k=99", 4), ("?top_k=0", 1)])
def test_predict_clamps_top_k(client, query, expected):
    assert len(post(client, png_bytes(), query=query).get_json()["predictions"]) == expected


def test_predict_routes_to_legacy():
    body = post(make_client(GEN1_NEW), png_bytes()).get_json()
    assert body["model"] == "legacy-gen1"
    assert body["predictions"][0] == {"name": "Pikachu", "confidence": 80.0}
    assert body["predictions"][2]["name"] == "Mr. Mime"
