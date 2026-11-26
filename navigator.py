import os
from pathlib import Path
from dotenv import load_dotenv

# Load Navigator Toolkit API key (overrides to ensure correct key)
load_dotenv()

from langchain_openai import ChatOpenAI  # noqa: E402 — import after env is loaded

# NAVIGATOR_KEY_INDEX (1/2/3) picks which of the NAVIGATOR_TOOLKIT_API_KEY[_2/_3]
# keys this process uses. Lets separate grading runs (e.g. one per Kripendorf's
# story set) use separate keys in parallel instead of sharing one key's
# ~10-requests-in-flight limit. Set via evaluator.py's --key flag; defaults to
# key 1, so nothing changes for anyone not passing it.
_key_index = os.getenv("NAVIGATOR_KEY_INDEX", "1").strip()
_key_var   = "NAVIGATOR_TOOLKIT_API_KEY" if _key_index == "1" else f"NAVIGATOR_TOOLKIT_API_KEY_{_key_index}"

API_KEY  = os.getenv(_key_var)
BASE_URL = "https://api.ai.it.ufl.edu/v1"

def make_llm(model: str, max_tokens: int = 4000, temperature: float = 0.0) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        openai_api_key=API_KEY,
        openai_api_base=BASE_URL,
        max_tokens=max_tokens,
        temperature=temperature,
    )
