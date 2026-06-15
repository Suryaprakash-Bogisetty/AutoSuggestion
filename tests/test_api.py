import pytest
import httpx
from unittest.mock import AsyncMock, MagicMock, patch
from httpx import AsyncClient, ASGITransport
from openai import APITimeoutError, APIStatusError

from main import app

BASE_PAYLOAD = {
    "prefix": "Patient complains of",
    "full_note": "Fever for three days and body aches.",
    "patient_id": "101",
    "stage_id": 1,
    "purpose": "chief_complaint",
    "debug": False,
    "context": {"chief_complaint": "headache"},
}


def _mock_openai(content: str = "fever and chills for days"):
    mock = MagicMock()
    mock.choices = [MagicMock()]
    mock.choices[0].message.content = content
    return mock


@pytest.fixture
async def http_client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


# ── health ────────────────────────────────────────────────────────────────────

async def test_health(http_client):
    r = await http_client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


# ── happy path ────────────────────────────────────────────────────────────────

async def test_suggest_success(http_client):
    with patch("main.client.chat.completions.create", new_callable=AsyncMock,
               return_value=_mock_openai("fever and chills for days")):
        r = await http_client.post("/suggest", json=BASE_PAYLOAD)

    assert r.status_code == 200
    data = r.json()
    assert data["suggestion"] == "fever and chills for days"
    assert data["full_text"] == "Patient complains of fever and chills for days"
    assert data["purpose"] == "chief_complaint"
    assert data["prompt"] is None


async def test_suggest_debug_returns_prompt(http_client):
    with patch("main.client.chat.completions.create", new_callable=AsyncMock,
               return_value=_mock_openai()):
        r = await http_client.post("/suggest", json={**BASE_PAYLOAD, "debug": True})

    assert r.status_code == 200
    assert r.json()["prompt"] is not None
    assert "chief_complaint" in r.json()["prompt"]


# ── all purposes ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("purpose", [
    "chief_complaint", "diagnosis", "investigations", "medications", "procedures",
])
async def test_suggest_all_purposes(http_client, purpose):
    with patch("main.client.chat.completions.create", new_callable=AsyncMock,
               return_value=_mock_openai()):
        r = await http_client.post("/suggest", json={**BASE_PAYLOAD, "purpose": purpose})

    assert r.status_code == 200, f"Unexpected status for purpose={purpose}"
    assert r.json()["purpose"] == purpose


# ── validation errors (422) ───────────────────────────────────────────────────

async def test_invalid_purpose_422(http_client):
    r = await http_client.post("/suggest", json={**BASE_PAYLOAD, "purpose": "xyz"})
    assert r.status_code == 422


async def test_empty_prefix_422(http_client):
    r = await http_client.post("/suggest", json={**BASE_PAYLOAD, "prefix": ""})
    assert r.status_code == 422


async def test_stage_id_zero_422(http_client):
    r = await http_client.post("/suggest", json={**BASE_PAYLOAD, "stage_id": 0})
    assert r.status_code == 422


async def test_missing_required_field_422(http_client):
    payload = {k: v for k, v in BASE_PAYLOAD.items() if k != "purpose"}
    r = await http_client.post("/suggest", json=payload)
    assert r.status_code == 422


# ── upstream error handling ───────────────────────────────────────────────────

async def test_missing_api_key_500(http_client):
    with patch("main.DEEPINFRA_API_KEY", ""):
        r = await http_client.post("/suggest", json=BASE_PAYLOAD)
    assert r.status_code == 500
    assert "DEEPINFRA_API_KEY" in r.json()["detail"]


async def test_timeout_returns_504(http_client):
    err = APITimeoutError(request=MagicMock(spec=httpx.Request))
    with patch("main.client.chat.completions.create", new_callable=AsyncMock, side_effect=err):
        r = await http_client.post("/suggest", json=BASE_PAYLOAD)
    assert r.status_code == 504
    assert "timed out" in r.json()["detail"].lower()


async def test_api_error_returns_502(http_client):
    mock_resp = MagicMock()
    mock_resp.status_code = 401
    mock_resp.request = MagicMock()
    mock_resp.headers.get.return_value = None
    err = APIStatusError("Unauthorized", response=mock_resp, body=None)
    with patch("main.client.chat.completions.create", new_callable=AsyncMock, side_effect=err):
        r = await http_client.post("/suggest", json=BASE_PAYLOAD)
    assert r.status_code == 502
