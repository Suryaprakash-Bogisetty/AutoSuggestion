import re
import time
from enum import Enum
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request
from openai import AsyncOpenAI, APITimeoutError, APIStatusError
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from config import DEEPINFRA_API_KEY, DEEPINFRA_BASE_URL, MODEL_ID, RATE_LIMIT, REQUEST_TIMEOUT
from logger import get_logger
from prompt import SYSTEM_PROMPT, build_user_message

log = get_logger("autosuggestion")


class Purpose(str, Enum):
    chief_complaint = "chief_complaint"
    diagnosis = "diagnosis"
    investigations = "investigations"
    medications = "medications"
    procedures = "procedures"
    vitals = "vitals"
    advice_followup = "advice_followup"
    doctors_notes = "doctors_notes"


class SuggestRequest(BaseModel):
    prefix: str = Field(..., min_length=1, max_length=500)
    full_note: str = Field(..., min_length=1, max_length=5000)
    patient_id: str = Field(..., min_length=1)
    stage_id: int = Field(..., ge=1)
    purpose: Purpose
    debug: bool = False
    context: dict[str, Any] = {}


class SuggestResponse(BaseModel):
    suggestion: str
    full_text: str
    purpose: str
    prompt: Optional[str] = None


limiter = Limiter(key_func=get_remote_address, default_limits=[RATE_LIMIT])

app = FastAPI(title="AutoSuggestion API", version="1.0.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

client = AsyncOpenAI(
    api_key=DEEPINFRA_API_KEY,
    base_url=DEEPINFRA_BASE_URL,
)


def _extract_completion(raw: str) -> str:
    """Strip <think>...</think> reasoning blocks then return first 5-6 words."""
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    if not cleaned:
        cleaned = raw.strip()
    return " ".join(cleaned.split()[:6])


@app.post("/suggest", response_model=SuggestResponse)
@limiter.limit(RATE_LIMIT)
async def suggest(request: Request, req: SuggestRequest):
    if not DEEPINFRA_API_KEY:
        raise HTTPException(status_code=500, detail="DEEPINFRA_API_KEY is not configured")

    user_message = build_user_message(
        prefix=req.prefix,
        full_note=req.full_note,
        context=req.context,
        purpose=req.purpose.value,
    )

    log.info(
        "suggest_request",
        extra={
            "patient_id": req.patient_id,
            "stage_id": req.stage_id,
            "purpose": req.purpose.value,
            "prefix_preview": req.prefix[:60],
        },
    )

    start = time.monotonic()
    try:
        response = await client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            max_tokens=4096,
            temperature=0.3,
            timeout=REQUEST_TIMEOUT,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
    except APITimeoutError:
        log.error("deepinfra_timeout", extra={"patient_id": req.patient_id, "purpose": req.purpose.value})
        raise HTTPException(status_code=504, detail="Model inference timed out. Please retry.")
    except APIStatusError as e:
        log.error(
            "deepinfra_error",
            extra={"patient_id": req.patient_id, "status_code": e.status_code, "error_message": str(e)},
        )
        raise HTTPException(status_code=502, detail=f"Model API error: {e.message}")

    duration_ms = round((time.monotonic() - start) * 1000)
    raw_content = response.choices[0].message.content or ""
    suggestion = _extract_completion(raw_content)

    log.info(
        "suggest_response",
        extra={
            "patient_id": req.patient_id,
            "stage_id": req.stage_id,
            "purpose": req.purpose.value,
            "suggestion": suggestion,
            "duration_ms": duration_ms,
        },
    )

    return SuggestResponse(
        suggestion=suggestion,
        full_text=f"{req.prefix} {suggestion}",
        purpose=req.purpose.value,
        prompt=user_message if req.debug else None,
    )


@app.get("/health")
async def health():
    return {"status": "ok"}
