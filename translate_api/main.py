import json
import re
import time
from enum import Enum
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from google import genai
from google.genai import types
from google.api_core import exceptions as google_exceptions
from openai import AsyncOpenAI, APITimeoutError, APIStatusError
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from config import (
    GEMINI_API_KEY, TRANSLATE_MODEL_ID,
    OPENAI_API_KEY, OPENAI_MODEL_ID,
    TRANSLATE_TIMEOUT, TRANSLATE_RATE_LIMIT,
    TRANSLATE_TEMPERATURE, TRANSLATE_MAX_TOKENS,
)
from logger import get_logger
from prompt import LANGUAGE_MAP, build_system_prompt, build_user_message

log = get_logger("translate_api")

gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


class LanguageCode(str, Enum):
    te = "te"
    hi = "hi"
    ta = "ta"
    kn = "kn"
    ml = "ml"
    mr = "mr"
    bn = "bn"
    pa = "pa"
    gu = "gu"
    or_ = "or"
    as_ = "as"


class Domain(str, Enum):
    medical = "medical"
    general = "general"


class TranslateRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    target_language: LanguageCode
    domain: Domain = Domain.general
    patient_id: Optional[str] = None
    debug: bool = False
    context: Optional[dict[str, Any]] = None


class TranslateResponse(BaseModel):
    translated: str
    original: str
    target_language: str
    language_name: str
    domain: str
    fallback: bool = False       # True when OpenAI fallback was used
    prompt: Optional[str] = None


limiter = Limiter(key_func=get_remote_address, default_limits=[TRANSLATE_RATE_LIMIT])
app = FastAPI(title="Translate API", version="1.0.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _gen_config(system_prompt: str) -> types.GenerateContentConfig:
    return types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=TRANSLATE_TEMPERATURE,
        max_output_tokens=TRANSLATE_MAX_TOKENS,
    )


_PREFIX_RE = re.compile(
    r"^(?:Translation|Translated\s+text|Here\s+is|Output|Result)\s*[:：]\s*",
    re.IGNORECASE,
)

_DOSAGE_RE = re.compile(
    r"\b([A-Za-z][a-z]{2,})\s+\d+\s*(?:mg|mcg|g\b|ml|mL|L\b|IU|units?|tabs?|caps?)",
    re.IGNORECASE,
)


def _strip_prefixes(text: str) -> str:
    return _PREFIX_RE.sub("", text).strip()


def _validate_numbers(original: str, translated: str) -> bool:
    orig_nums = {n for n in re.findall(r"\d+(?:\.\d+)?", original) if float(n) >= 10 or "." in n}
    if not orig_nums:
        return True
    return orig_nums.issubset(set(re.findall(r"\d+(?:\.\d+)?", translated)))


def _validate_drugs(original: str, translated: str) -> bool:
    drugs = {m.lower() for m in _DOSAGE_RE.findall(original)}
    if not drugs:
        return True
    t_lower = translated.lower()
    return all(drug in t_lower for drug in drugs)


# ── Gemini error classifier ──────────────────────────────────────
# Transient errors → safe to fallback to OpenAI
# Config/auth errors → raise immediately, don't burn OpenAI credits silently

_GEMINI_TRANSIENT = (
    google_exceptions.ServiceUnavailable,   # 503 — model overloaded
    google_exceptions.ResourceExhausted,    # 429 — quota exceeded
    google_exceptions.DeadlineExceeded,     # timeout
    google_exceptions.InternalServerError,  # 500 — Gemini server error
    google_exceptions.Aborted,              # request aborted
)

_GEMINI_CONFIG_ERRORS = (
    google_exceptions.PermissionDenied,     # 403 — bad/expired API key
    google_exceptions.Unauthenticated,      # 401 — missing credentials
    google_exceptions.NotFound,             # 404 — wrong model name
    google_exceptions.InvalidArgument,      # 400 — bad request (our bug)
)


def _should_fallback(exc: Exception) -> bool:
    """Return True only for transient Gemini errors where OpenAI may succeed."""
    if isinstance(exc, _GEMINI_TRANSIENT):
        return True
    # google-genai wraps errors inside google.api_core; also check message for 503/429
    msg = str(exc).lower()
    return any(code in msg for code in ("503", "unavailable", "resource exhausted", "429", "deadline"))


def _gemini_http_status(exc: Exception) -> int:
    """Map a non-transient Gemini error to an HTTP status for the caller."""
    if isinstance(exc, (google_exceptions.PermissionDenied, google_exceptions.Unauthenticated)):
        return 500  # config problem — surface as server error
    if isinstance(exc, google_exceptions.InvalidArgument):
        return 400
    return 502


# ── Gemini call ──────────────────────────────────────────────────

async def _gemini_translate(system_prompt: str, user_msg: str) -> str:
    response = await gemini_client.aio.models.generate_content(
        model=TRANSLATE_MODEL_ID,
        contents=user_msg,
        config=_gen_config(system_prompt),
    )
    return _strip_prefixes(response.text or "")


# ── OpenAI fallback call ─────────────────────────────────────────

async def _openai_translate(system_prompt: str, user_msg: str) -> str:
    response = await openai_client.chat.completions.create(
        model=OPENAI_MODEL_ID,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_msg},
        ],
        temperature=TRANSLATE_TEMPERATURE,
        max_tokens=TRANSLATE_MAX_TOKENS,
        timeout=TRANSLATE_TIMEOUT,
    )
    return _strip_prefixes(response.choices[0].message.content or "")


# ── OpenAI streaming fallback ────────────────────────────────────

async def _openai_stream(system_prompt: str, user_msg: str):
    stream = await openai_client.chat.completions.create(
        model=OPENAI_MODEL_ID,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_msg},
        ],
        temperature=TRANSLATE_TEMPERATURE,
        max_tokens=TRANSLATE_MAX_TOKENS,
        timeout=TRANSLATE_TIMEOUT,
        stream=True,
    )
    first = True
    async for chunk in stream:
        text = chunk.choices[0].delta.content or ""
        if not text:
            continue
        if first:
            text = _strip_prefixes(text)
            first = False
        if text:
            yield text


# ── Shared post-processing log ───────────────────────────────────

def _log_quality_warnings(req: TranslateRequest, translated: str) -> None:
    if req.domain.value == "medical":
        if not _validate_numbers(req.text, translated):
            log.warning("translate_number_mismatch", extra={"patient_id": req.patient_id})
        if not _validate_drugs(req.text, translated):
            log.warning("translate_drug_name_mismatch", extra={"patient_id": req.patient_id})


# ════════════════════════════════════════════════════════════════
# Endpoints
# ════════════════════════════════════════════════════════════════

@app.post("/translate", response_model=TranslateResponse)
@limiter.limit(TRANSLATE_RATE_LIMIT)
async def translate(request: Request, req: TranslateRequest):
    if not gemini_client and not openai_client:
        raise HTTPException(status_code=500, detail="No API keys configured")

    lang_code     = req.target_language.value
    language_name = LANGUAGE_MAP[lang_code]
    system_prompt = build_system_prompt(language_name, req.domain.value)
    user_msg      = build_user_message(req.text, req.domain.value, req.context)

    log.info("translate_request", extra={
        "patient_id": req.patient_id, "target_language": lang_code,
        "domain": req.domain.value, "text_preview": req.text[:100],
    })

    fallback = False
    start = time.monotonic()

    # ── Try Gemini first ────────────────────────────────────────
    translated = None
    if gemini_client:
        try:
            translated = await _gemini_translate(system_prompt, user_msg)
        except Exception as e:
            if _should_fallback(e):
                log.warning("gemini_transient_error_using_fallback", extra={
                    "patient_id": req.patient_id, "error": str(e)[:120],
                })
            else:
                # Config/auth/bad-request — raise immediately, don't silently use OpenAI
                log.error("gemini_config_error_no_fallback", extra={
                    "patient_id": req.patient_id, "error": str(e)[:120],
                })
                raise HTTPException(
                    status_code=_gemini_http_status(e),
                    detail=f"Translation service error: {str(e)[:200]}",
                )

    # ── Fall back to OpenAI ─────────────────────────────────────
    if translated is None:
        if not openai_client:
            raise HTTPException(status_code=502, detail="Primary model failed and no fallback configured")
        try:
            translated = await _openai_translate(system_prompt, user_msg)
            fallback = True
        except APITimeoutError:
            raise HTTPException(status_code=504, detail="Translation timed out on both providers")
        except APIStatusError as e:
            raise HTTPException(status_code=502, detail=f"Fallback API error: {e.message}")
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Translation failed: {str(e)}")

    duration_ms = round((time.monotonic() - start) * 1000)
    _log_quality_warnings(req, translated)

    log.info("translate_response", extra={
        "patient_id": req.patient_id, "target_language": lang_code,
        "domain": req.domain.value, "duration_ms": duration_ms,
        "fallback": fallback, "translated_preview": translated[:100],
    })

    return TranslateResponse(
        translated=translated,
        original=req.text,
        target_language=lang_code,
        language_name=language_name,
        domain=req.domain.value,
        fallback=fallback,
        prompt=f"[System]\n{system_prompt}\n\n[User]\n{user_msg}" if req.debug else None,
    )


@app.post("/translate/stream")
@limiter.limit(TRANSLATE_RATE_LIMIT)
async def translate_stream(request: Request, req: TranslateRequest):
    if not gemini_client and not openai_client:
        raise HTTPException(status_code=500, detail="No API keys configured")

    lang_code     = req.target_language.value
    language_name = LANGUAGE_MAP[lang_code]
    system_prompt = build_system_prompt(language_name, req.domain.value)
    user_msg      = build_user_message(req.text, req.domain.value, req.context)

    log.info("translate_stream_request", extra={
        "patient_id": req.patient_id, "target_language": lang_code,
        "domain": req.domain.value,
    })

    async def _stream_gen():
        first_chunk = True
        used_fallback = False
        try:
            # ── Gemini stream ───────────────────────────────────
            if gemini_client:
                try:
                    async for chunk in await gemini_client.aio.models.generate_content_stream(
                        model=TRANSLATE_MODEL_ID,
                        contents=user_msg,
                        config=_gen_config(system_prompt),
                    ):
                        text = getattr(chunk, "text", None) or ""
                        if not text:
                            continue
                        if first_chunk:
                            text = _strip_prefixes(text)
                            first_chunk = False
                        if text:
                            yield f"data: {json.dumps({'chunk': text, 'done': False, 'fallback': False})}\n\n"
                    return  # Gemini finished successfully — skip fallback
                except Exception as e:
                    if not _should_fallback(e):
                        # Config/auth error — raise immediately
                        log.error("gemini_stream_config_error", extra={
                            "patient_id": req.patient_id, "error": str(e)[:120],
                        })
                        yield "event: error\ndata: translation service configuration error\n\n"
                        return
                    log.warning("gemini_stream_transient_error_using_fallback", extra={
                        "patient_id": req.patient_id, "error": str(e)[:120],
                    })
                    # Only fall through to OpenAI if nothing was yielded yet
                    if not first_chunk:
                        yield "event: error\ndata: primary stream failed mid-response\n\n"
                        return
                    used_fallback = True

            # ── OpenAI fallback stream ──────────────────────────
            if openai_client:
                async for text in _openai_stream(system_prompt, user_msg):
                    yield f"data: {json.dumps({'chunk': text, 'done': False, 'fallback': True})}\n\n"
            else:
                yield "event: error\ndata: no fallback configured\n\n"

        except Exception as e:
            log.error("translate_stream_error", extra={"patient_id": req.patient_id, "error": str(e)})
            yield "event: error\ndata: translation failed\n\n"
        finally:
            yield f"data: {json.dumps({'done': True, 'fallback': used_fallback})}\n\n"

    return StreamingResponse(_stream_gen(), media_type="text/event-stream")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "primary": TRANSLATE_MODEL_ID,
        "fallback": OPENAI_MODEL_ID,
        "gemini_configured": bool(GEMINI_API_KEY),
        "openai_configured": bool(OPENAI_API_KEY),
    }
