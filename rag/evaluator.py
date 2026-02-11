"""실시간 Faithfulness 평가기 — RAGAS와 동일한 2단계 로직을 OpenRouter로 구현.

1. 답변에서 사실적 주장(claims) 추출
2. 각 주장이 검색된 컨텍스트에 의해 뒷받침되는지 검증
3. 점수 = 뒷받침된 주장 수 / 전체 주장 수
"""

import json

from openai import OpenAI

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
EVAL_MODEL = "deepseek/deepseek-v3.2"


def evaluate_faithfulness(
    answer: str, contexts: list[str], api_key: str
) -> float:
    """Faithfulness 점수 계산 (0.0 ~ 1.0)."""
    client = OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)

    claims = _extract_claims(client, answer)
    if not claims:
        return 1.0  # 주장이 없으면 faithful로 간주

    context_text = "\n\n---\n\n".join(contexts)
    supported = _verify_claims(client, claims, context_text)

    return round(supported / len(claims), 2)


def _extract_claims(client: OpenAI, answer: str) -> list[str]:
    """답변에서 검증 가능한 사실적 주장을 추출."""
    response = client.chat.completions.create(
        model=EVAL_MODEL,
        messages=[
            {
                "role": "user",
                "content": (
                    "다음 답변에서 검증 가능한 사실적 주장(factual claims)을 추출하세요.\n"
                    "각 주장은 독립적이고 구체적이어야 합니다.\n"
                    "JSON 배열로만 반환하세요. 다른 텍스트 없이 배열만 출력하세요.\n\n"
                    f"답변:\n{answer}\n\n"
                    '출력: ["주장1", "주장2", ...]'
                ),
            }
        ],
        temperature=0.0,
        max_tokens=1024,
    )

    return _parse_json_array(response.choices[0].message.content)


def _verify_claims(
    client: OpenAI, claims: list[str], context: str
) -> int:
    """주장 목록을 컨텍스트와 대조하여 지지되는 수를 반환."""
    claims_text = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(claims))

    response = client.chat.completions.create(
        model=EVAL_MODEL,
        messages=[
            {
                "role": "user",
                "content": (
                    "각 주장이 아래 컨텍스트에 의해 뒷받침되는지 판단하세요.\n"
                    "컨텍스트에 명시적으로 있거나 직접 추론 가능한 경우에만 1(지지), "
                    "그렇지 않으면 0을 부여하세요.\n"
                    "JSON 배열로만 반환하세요.\n\n"
                    f"컨텍스트:\n{context}\n\n"
                    f"주장 목록:\n{claims_text}\n\n"
                    "출력: [1, 0, 1, ...]"
                ),
            }
        ],
        temperature=0.0,
        max_tokens=256,
    )

    verdicts = _parse_json_array(response.choices[0].message.content)
    return sum(1 for v in verdicts if v == 1)


def _parse_json_array(text: str) -> list:
    """LLM 응답에서 JSON 배열을 안전하게 추출."""
    text = text.strip()
    # 코드블록 제거
    if "```" in text:
        parts = text.split("```")
        text = parts[1] if len(parts) >= 3 else parts[-1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, IndexError):
        return []
