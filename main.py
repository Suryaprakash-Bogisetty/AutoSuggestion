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
    text_before_cursor: str = Field(..., min_length=1, max_length=5000)
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
    """Strip <think>...</think> blocks, return up to first sentence end (max 20 words) or 15 words."""
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    if not cleaned:
        cleaned = raw.strip()
    words = cleaned.split()
    for i, word in enumerate(words[:20]):
        if word.endswith((".", "!", "?")):
            return " ".join(words[: i + 1])
    return " ".join(words[:15])


_VOWELS = set("aeiouAEIOU")
_CONSONANTS = set("bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ")


def _has_real_text(text: str) -> bool:
    """Return False if text looks like keyboard mash.

    Only applies vowel-ratio check to words with 4+ chars — medical
    abbreviations (BP, CBC, LFT) are short and vowel-free but still real.
    """
    all_words = re.findall(r"[a-zA-Z]{2,}", text)
    if not all_words:
        return False
    long_words = [w for w in all_words if len(w) >= 4]
    if not long_words:
        # Only abbreviations/short tokens — treat as real
        return True
    # A real word has ≥2 total vowels and at least one consonant.
    # "asdf"/"qwerty" have only 1 vowel; "Sputum"/"Started"/"reports" have 2+.
    return any(
        sum(c in _VOWELS for c in w) >= 2
        and any(c in _CONSONANTS for c in w)
        for w in long_words
    )


def _sanitize(text: str) -> str:
    """Strip non-ASCII characters, trailing ellipsis/artifacts, and normalize whitespace."""
    text = re.sub(r"[^\x00-\x7F]+", "", text)
    text = re.sub(r"\.{2,}$", "", text)   # trailing "..." or ".."
    return text.strip()


_STOP_WORDS = {
    "the", "a", "an", "and", "or", "of", "with", "in", "on", "is", "has",
    "he", "she", "it", "to", "for", "at", "by", "as", "was", "be", "are",
    "from", "his", "her", "have", "had", "this", "that", "are", "not", "no",
}


def _content_words(text: str) -> set:
    return {w.lower() for w in re.findall(r"[a-zA-Z]+", text) if w.lower() not in _STOP_WORDS and len(w) > 2}


def _is_near_echo(suggestion: str, text_before_cursor: str) -> bool:
    """Return True if 90%+ of the suggestion's content words are already in the text before cursor."""
    s_words = _content_words(suggestion)
    if not s_words:
        return False
    n_words = _content_words(text_before_cursor)
    overlap = s_words & n_words
    return len(overlap) / len(s_words) >= 0.75


@app.post("/suggest", response_model=SuggestResponse)
@limiter.limit(RATE_LIMIT)
async def suggest(request: Request, req: SuggestRequest):
    if not DEEPINFRA_API_KEY:
        raise HTTPException(status_code=500, detail="DEEPINFRA_API_KEY is not configured")

    if not _has_real_text(req.text_before_cursor):
        log.info("suggest_skipped_garbage", extra={"patient_id": req.patient_id})
        return SuggestResponse(suggestion="", full_text=req.text_before_cursor, purpose=req.purpose.value)

    user_message = build_user_message(
        text_before_cursor=req.text_before_cursor,
        context=req.context,
        purpose=req.purpose.value,
    )

    log.info(
        "suggest_request",
        extra={
            "patient_id": req.patient_id,
            "stage_id": req.stage_id,
            "purpose": req.purpose.value,
            "cursor_preview": req.text_before_cursor[-60:],
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
            max_tokens=50,
            temperature=0.7,
            stop=["\n"],
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
    suggestion = _sanitize(_extract_completion(raw_content))

    if _is_near_echo(suggestion, req.text_before_cursor):
        log.info("suggest_near_echo_detected", extra={"patient_id": req.patient_id, "suggestion": suggestion})
        suggestion = ""

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
        full_text=req.text_before_cursor + suggestion,
        purpose=req.purpose.value,
        prompt=user_message if req.debug else None,
    )


@app.get("/health")
async def health():
    return {"status": "ok"}
