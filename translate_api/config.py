from dotenv import load_dotenv
import os

load_dotenv()

# Primary: Gemini
GEMINI_API_KEY     = os.getenv("GEMINI_API_KEY", "")
TRANSLATE_MODEL_ID = "gemini-2.5-flash-lite"

# Fallback: OpenAI
OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL_ID    = os.getenv("OPENAI_MODEL_ID", "gpt-4o-mini")

TRANSLATE_TIMEOUT      = float(os.getenv("TRANSLATE_TIMEOUT",      "30"))
TRANSLATE_RATE_LIMIT   = os.getenv("TRANSLATE_RATE_LIMIT",          "20/minute")
TRANSLATE_TEMPERATURE  = float(os.getenv("TRANSLATE_TEMPERATURE",   "0.3"))
TRANSLATE_MAX_TOKENS   = int(os.getenv("TRANSLATE_MAX_TOKENS",       "2000"))
