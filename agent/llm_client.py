"""
Thin LLM wrapper so the rest of the codebase doesn't care which provider
is behind it. Every other file only ever calls call_llm() / call_llm_json() -
this is the ONLY file that needs to change to swap providers.

Set LLM_PROVIDER=anthropic (default) or LLM_PROVIDER=gemini in .env.
"""
import os
import json

from dotenv import load_dotenv
load_dotenv()  # loads .env into os.environ - this is the only place we need to do this

PROVIDER = os.environ.get("LLM_PROVIDER", "anthropic").lower()

# Anthropic: https://docs.anthropic.com/en/docs/about-claude/models
ANTHROPIC_MODEL = os.environ.get("BIZCOPILOT_MODEL", "claude-sonnet-4-6")

# Gemini: gemini-2.5-flash is the current stable GA model (scheduled to be
# retired ~Oct 2026 - if that's happened by the time you read this, switch
# this env var to whatever the current stable Gemini model is).
GEMINI_MODEL = os.environ.get("BIZCOPILOT_MODEL", "gemini-2.5-flash")


def _call_anthropic(system, user, max_tokens):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set. Copy .env.example to .env and add your key."
        )
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    parts = [block.text for block in response.content if block.type == "text"]
    return "".join(parts)


def _call_gemini(system, user, max_tokens, json_mode):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY not set. Copy .env.example to .env and add your key. "
            "Get one at https://aistudio.google.com/ - note this is separate from "
            "a Gemini Advanced/consumer subscription."
        )
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(
        system_instruction=system,
        max_output_tokens=max_tokens,
    )
    if json_mode:
        config.response_mime_type = "application/json"

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=user,
        config=config,
    )
    return response.text


def call_llm(system: str, user: str, max_tokens: int = 1000, json_mode: bool = False) -> str:
    """
    Calls the configured LLM (LLM_PROVIDER env var) and returns raw text.
    If json_mode=True, the system prompt should already instruct the model
    to return ONLY valid JSON - this function does not silently coerce it.
    """
    if PROVIDER == "gemini":
        return _call_gemini(system, user, max_tokens, json_mode)
    elif PROVIDER == "anthropic":
        return _call_anthropic(system, user, max_tokens)
    else:
        raise RuntimeError(f"Unknown LLM_PROVIDER '{PROVIDER}' - use 'anthropic' or 'gemini'.")


def call_llm_json(system: str, user: str, max_tokens: int = 1000) -> dict:
    """Calls the LLM expecting a JSON object back and parses it."""
    raw = call_llm(system, user, max_tokens=max_tokens, json_mode=True)
    cleaned = raw.strip()

    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]

    try:
        return json.loads(cleaned.strip())
    except json.JSONDecodeError as e:
        print("\n--- INVALID JSON FROM LLM ---")
        print(cleaned)
        print(f"\nJSON length: {len(cleaned)}")
        print(f"Error position: {e.pos}")
        print(f"Characters around error: {repr(cleaned[max(0, e.pos-80):e.pos+80])}")
        print("--- END INVALID JSON ---\n")
        raise
