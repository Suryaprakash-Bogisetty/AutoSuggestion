from dotenv import load_dotenv
import os

load_dotenv()

DEEPINFRA_API_KEY = os.getenv("DEEPINFRA_API_KEY", "")
DEEPINFRA_BASE_URL = "https://api.deepinfra.com/v1/openai"
MODEL_ID = "Qwen/Qwen2-7B-Instruct"

# How long to wait for a response from DeepInfra (seconds)
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "30"))

# slowapi rate limit string — e.g. "30/minute", "100/hour"
RATE_LIMIT = os.getenv("RATE_LIMIT", "30/minute")
