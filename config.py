from dotenv import load_dotenv
import os

load_dotenv()

DEEPINFRA_API_KEY  = os.getenv("DEEPINFRA_API_KEY", "")
DEEPINFRA_BASE_URL = "https://api.deepinfra.com/v1/openai"

# Suggest endpoint — fast, latency-sensitive
MODEL_ID        = "Qwen/Qwen2-7B-Instruct"
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "30"))
RATE_LIMIT      = os.getenv("RATE_LIMIT", "30/minute")

# Rephrase endpoints — quality-focused, separate model and limits
REPHRASE_MODEL_ID           = os.getenv("REPHRASE_MODEL_ID",           "Qwen/Qwen2-72B-Instruct")
REPHRASE_TIMEOUT            = float(os.getenv("REPHRASE_TIMEOUT",            "60"))
REPHRASE_RATE_LIMIT         = os.getenv("REPHRASE_RATE_LIMIT",               "10/minute")
REPHRASE_TEMPERATURE        = float(os.getenv("REPHRASE_TEMPERATURE",        "0.05"))
REPHRASE_MAX_TOKENS         = int(os.getenv("REPHRASE_MAX_TOKENS",           "1000"))
REPHRASE_TOP_P              = float(os.getenv("REPHRASE_TOP_P",              "0.85"))
REPHRASE_FREQUENCY_PENALTY  = float(os.getenv("REPHRASE_FREQUENCY_PENALTY",  "0.1"))
