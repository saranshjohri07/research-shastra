import json
from functools import lru_cache

from groq import Groq

from ..config import get_settings


SEMANTIC_ABSTENTION_REASON = (
    "The supplied paper does not contain information needed to answer this question."
)

SYSTEM_PROMPT = """You answer research questions using only the supplied paper evidence.
Do not use outside knowledge or introduce unsupported facts. If the evidence does
not support an answer, return valid JSON with:
{"answerable": false, "answer": null, "reason": "The supplied paper does not contain information needed to answer this question."}
If the evidence supports an answer, return valid JSON with:
{"answerable": true, "answer": "your answer here", "reason": null}
Do not invent citations, page numbers, sections, or paper names. Cite supporting
evidence only by its provided identifier, such as [E1]. Be concise but complete,
and preserve uncertainty expressed in the evidence."""


@lru_cache(maxsize=1)
def _get_client(api_key: str) -> Groq:
    return Groq(api_key=api_key)


def generate_grounded_answer(question: str, evidence_context: str) -> dict[str, str | bool | None]:
    settings = get_settings()
    completion = _get_client(settings.groq_api_key).chat.completions.create(
        model=settings.groq_model,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Question:\n{question}\n\n"
                    f"Supplied evidence:\n{evidence_context}"
                ),
            },
        ],
    )
    raw_answer = completion.choices[0].message.content
    if not raw_answer or not raw_answer.strip():
        raise RuntimeError("The answer provider returned an empty response")

    try:
        payload = json.loads(raw_answer)
    except json.JSONDecodeError:
        payload = None

    if isinstance(payload, dict):
        answerable = payload.get("answerable")
        answer = payload.get("answer")
        reason = payload.get("reason") or payload.get("abstention_reason")

        if answerable is False:
            return {
                "answerable": False,
                "answer": None,
                "reason": reason or SEMANTIC_ABSTENTION_REASON,
            }

        if answerable is True:
            if not isinstance(answer, str) or not answer.strip():
                raise RuntimeError("The answer provider returned an invalid structured answer")
            return {
                "answerable": True,
                "answer": answer.strip(),
                "reason": None,
            }

    text = raw_answer.strip()
    lowered = text.lower()
    if any(
        marker in lowered
        for marker in (
            "does not contain",
            "not enough information",
            "cannot determine",
            "not provided",
            "does not provide enough information",
        )
    ):
        return {
            "answerable": False,
            "answer": None,
            "reason": SEMANTIC_ABSTENTION_REASON,
        }

    return {
        "answerable": True,
        "answer": text,
        "reason": None,
    }
