import re
import time
from enum import Enum
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI, APITimeoutError, APIStatusError
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from config import (
    DEEPINFRA_API_KEY, DEEPINFRA_BASE_URL, MODEL_ID, RATE_LIMIT, REQUEST_TIMEOUT,
    REPHRASE_MODEL_ID, REPHRASE_TIMEOUT, REPHRASE_RATE_LIMIT,
    REPHRASE_TEMPERATURE, REPHRASE_MAX_TOKENS, REPHRASE_TOP_P, REPHRASE_FREQUENCY_PENALTY,
)
from logger import get_logger
from prompt import SYSTEM_PROMPT, build_user_message, REPHRASE_SYSTEM_PROMPT, build_rephrase_user_message

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


class RephraseRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    purpose: Optional[Purpose] = None
    context: dict[str, Any] = {}
    patient_id: Optional[str] = None
    debug: bool = False


class RephraseResponse(BaseModel):
    rephrased: str
    original: str
    purpose: Optional[str] = None
    fallback: bool = False
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
    """Strip non-ASCII except whitelisted clinical symbols, remove trailing ellipsis."""
    # °  degree (temperature), μ  micro (μg), ≥≤ comparators, ±  plus-minus
    text = re.sub(r"[^\x00-\x7F°μ≥≤±]", "", text)
    text = re.sub(r"\.{2,}$", "", text)
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


def _strip_think_blocks(text: str) -> str:
    """Remove <think>...</think> blocks from LLM output."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


# Word immediately before a dosage pattern is treated as a drug name.
_DOSAGE_RE = re.compile(
    r"\b([A-Za-z][a-z]{2,})\s+\d+\s*(?:mg|mcg|g\b|ml|mL|L\b|IU|units?|tabs?|caps?)",
    re.IGNORECASE,
)


def _extract_drug_names(text: str) -> set:
    """Return lowercase drug names identified by dosage-proximity heuristic."""
    return {m.lower() for m in _DOSAGE_RE.findall(text)}


_LONG_NOTE_THRESHOLD = 50  # words


def _pick_model(purpose: Optional[str], text: str) -> tuple[str, float]:
    """Return (model_id, timeout) for this rephrase request.

    72B is used only for long doctor's notes — where structure quality matters.
    7B is used for everything else — same accuracy, 3x faster.
    """
    if purpose == Purpose.doctors_notes.value and len(text.split()) > _LONG_NOTE_THRESHOLD:
        return REPHRASE_MODEL_ID, REPHRASE_TIMEOUT
    return MODEL_ID, REQUEST_TIMEOUT


def _validate_rephrased_output(original: str, rephrased: str) -> bool:
    """Return True if the rephrased text is safe to use.

    Checks: non-empty, reasonable length ratio, numeric values preserved, drug names preserved.
    """
    if not rephrased:
        return False
    orig_words = original.split()
    rep_words = rephrased.split()
    if orig_words:
        ratio = len(rep_words) / len(orig_words)
        if ratio < 0.4 or ratio > 3.0:
            return False
    # No \b — units like 325mg/4L/101.4F have no word boundary after the digit.
    # Single-digit integers excluded: may become Roman numerals (II/III) or words ("two days").
    orig_numbers = {n for n in re.findall(r"\d+(?:\.\d+)?", original)
                    if float(n) >= 10 or "." in n}
    if orig_numbers:
        rep_numbers = set(re.findall(r"\d+(?:\.\d+)?", rephrased))
        if not orig_numbers.issubset(rep_numbers):
            return False
    # Drug names must survive unchanged
    orig_drugs = _extract_drug_names(original)
    if orig_drugs:
        rep_lower = rephrased.lower()
        if any(drug not in rep_lower for drug in orig_drugs):
            return False
    return True


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


@app.post("/rephrase", response_model=RephraseResponse)
@limiter.limit(REPHRASE_RATE_LIMIT)
async def rephrase(request: Request, req: RephraseRequest):
    if not DEEPINFRA_API_KEY:
        raise HTTPException(status_code=500, detail="DEEPINFRA_API_KEY is not configured")

    if not _has_real_text(req.text):
        log.info("rephrase_skipped_garbage", extra={"patient_id": req.patient_id})
        return RephraseResponse(
            rephrased=req.text,
            original=req.text,
            purpose=req.purpose.value if req.purpose else None,
            fallback=True,
        )

    purpose_str = req.purpose.value if req.purpose else None
    model, timeout = _pick_model(purpose_str, req.text)

    user_message = build_rephrase_user_message(
        text=req.text,
        purpose=purpose_str,
        context=req.context,
    )

    log.info(
        "rephrase_request",
        extra={
            "patient_id": req.patient_id,
            "purpose": purpose_str,
            "model": model,
            "text_preview": req.text[:100],
        },
    )

    start = time.monotonic()
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": REPHRASE_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            max_tokens=REPHRASE_MAX_TOKENS,
            temperature=REPHRASE_TEMPERATURE,
            top_p=REPHRASE_TOP_P,
            frequency_penalty=REPHRASE_FREQUENCY_PENALTY,
            timeout=timeout,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
    except APITimeoutError:
        log.error("deepinfra_timeout_rephrase", extra={"patient_id": req.patient_id})
        raise HTTPException(status_code=504, detail="Model inference timed out. Please retry.")
    except APIStatusError as e:
        log.error(
            "deepinfra_error_rephrase",
            extra={"patient_id": req.patient_id, "status_code": e.status_code, "error_message": str(e)},
        )
        raise HTTPException(status_code=502, detail=f"Model API error: {e.message}")

    duration_ms = round((time.monotonic() - start) * 1000)
    raw_content = response.choices[0].message.content or ""
    rephrased = _sanitize(_strip_think_blocks(raw_content))

    fallback = not _validate_rephrased_output(req.text, rephrased)
    if fallback:
        log.info("rephrase_fallback", extra={"patient_id": req.patient_id, "raw": raw_content[:100]})
        rephrased = req.text

    log.info(
        "rephrase_response",
        extra={
            "patient_id": req.patient_id,
            "purpose": purpose_str,
            "model": model,
            "rephrased_preview": rephrased[:100],
            "fallback": fallback,
            "duration_ms": duration_ms,
        },
    )

    return RephraseResponse(
        rephrased=rephrased,
        original=req.text,
        purpose=purpose_str,
        fallback=fallback,
        prompt=user_message if req.debug else None,
    )


@app.post("/rephrase/stream")
@limiter.limit(REPHRASE_RATE_LIMIT)
async def rephrase_stream(request: Request, req: RephraseRequest):
    if not DEEPINFRA_API_KEY:
        raise HTTPException(status_code=500, detail="DEEPINFRA_API_KEY is not configured")

    if not _has_real_text(req.text):
        async def _fallback_gen():
            yield f"data: {req.text}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(_fallback_gen(), media_type="text/event-stream")

    purpose_str = req.purpose.value if req.purpose else None
    model, timeout = _pick_model(purpose_str, req.text)

    user_message = build_rephrase_user_message(
        text=req.text,
        purpose=purpose_str,
        context=req.context,
    )

    log.info(
        "rephrase_stream_request",
        extra={
            "patient_id": req.patient_id,
            "purpose": purpose_str,
            "model": model,
            "text_preview": req.text[:100],
        },
    )

    async def _stream_gen():
        buffer = ""
        in_think = False
        try:
            stream = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": REPHRASE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=REPHRASE_MAX_TOKENS,
                temperature=REPHRASE_TEMPERATURE,
                top_p=REPHRASE_TOP_P,
                frequency_penalty=REPHRASE_FREQUENCY_PENALTY,
                timeout=timeout,
                stream=True,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            async for chunk in stream:
                token = chunk.choices[0].delta.content or ""
                if not token:
                    continue
                buffer += token
                # Filter <think> blocks on the fly
                while True:
                    if in_think:
                        end = buffer.find("</think>")
                        if end != -1:
                            buffer = buffer[end + len("</think>"):]
                            in_think = False
                        else:
                            buffer = ""
                            break
                    else:
                        start_idx = buffer.find("<think>")
                        if start_idx != -1:
                            safe = buffer[:start_idx]
                            if safe:
                                yield f"data: {safe}\n\n"
                            buffer = buffer[start_idx + len("<think>"):]
                            in_think = True
                        else:
                            # Keep 8-char tail to catch "</think>" split across chunks
                            cutoff = max(0, len(buffer) - 8)
                            if cutoff > 0:
                                yield f"data: {buffer[:cutoff]}\n\n"
                                buffer = buffer[cutoff:]
                            break
            # Flush remainder
            if buffer and not in_think:
                clean = _sanitize(buffer)
                if clean:
                    yield f"data: {clean}\n\n"
        except (APITimeoutError, APIStatusError) as e:
            log.error("rephrase_stream_error", extra={"patient_id": req.patient_id, "error": str(e)})
            yield "event: error\ndata: inference failed\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(_stream_gen(), media_type="text/event-stream")


@app.get("/health")
async def health():
    return {"status": "ok"}
