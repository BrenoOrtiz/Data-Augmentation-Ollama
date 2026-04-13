import logging
import time

import ollama

from config import MAX_RETRIES, MAX_TOKENS, OLLAMA_HOST, TEMPERATURE

logger = logging.getLogger(__name__)


_SYSTEM_PROMPT = (
    "You are a financial journalist. Your task is to generate realistic, concise "
    "financial news sentences in English, similar to the Financial PhraseBank dataset. "
    "Write exactly one sentence per response. Do not include labels, explanations, "
    "quotes, or any additional text."
)

_USER_PROMPTS: dict[str, str] = {
    "positive": (
        "Generate one short financial news sentence with POSITIVE sentiment. "
        "Examples of positive topics: earnings beat, revenue growth, profit increase, "
        "successful acquisition, raised guidance, dividend increase, stock rally."
    ),
    "neutral": (
        "Generate one short financial news sentence with NEUTRAL sentiment. "
        "Examples of neutral topics: scheduled filings, management appointments, "
        "operational announcements, product launches with no clear financial impact, "
        "routine business updates."
    ),
    "negative": (
        "Generate one short financial news sentence with NEGATIVE sentiment. "
        "Examples of negative topics: profit warning, revenue decline, layoffs, "
        "failed deal, debt downgrade, stock drop, missed earnings, legal setback."
    ),
}


def _clean(text: str) -> str:
    """Strip surrounding quotes and extra whitespace."""
    return text.strip().strip("\"'").strip()


def generate_sentence(model: str, sentiment: str) -> str | None:
    """
    Generate one synthetic financial sentence for the given sentiment using
    a local Ollama model. Returns None after MAX_RETRIES failures.
    """
    client = ollama.Client(host=OLLAMA_HOST)
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": _USER_PROMPTS[sentiment]},
    ]

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat(
                model=model,
                messages=messages,
                options={
                    "temperature": TEMPERATURE,
                    "num_predict": MAX_TOKENS,
                },
            )
            text = _clean(response["message"]["content"])
            if text:
                return text
        except Exception as exc:
            wait = 2**attempt
            logger.warning(
                "Attempt %d/%d failed for model=%s sentiment=%s: %s — retrying in %ds",
                attempt + 1,
                MAX_RETRIES,
                model,
                sentiment,
                exc,
                wait,
            )
            if attempt < MAX_RETRIES - 1:
                time.sleep(wait)

    logger.error(
        "All %d attempts failed for model=%s sentiment=%s", MAX_RETRIES, model, sentiment
    )
    return None
