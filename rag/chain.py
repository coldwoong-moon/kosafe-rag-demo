"""RAG 체인 — 검색 + 스트리밍 생성 (OpenRouter + DeepSeek V3.2)."""

from openai import OpenAI

from .vectorstore import search

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = "deepseek/deepseek-v3.2"

SYSTEM_PROMPT = """당신은 건설 안전 전문가입니다.
사용자의 질문에 대해 검색된 정보를 바탕으로 정확하고 유용한 답변을 제공합니다.

## 답변 규칙
1. 검색 결과에 있는 정보만 사용하세요. 검색 결과에 없는 정보는 절대 만들어내지 마세요.
2. 각 정보를 인용할 때 [1], [2] 형식의 각주를 사용하세요.
3. 검색 결과가 질문과 관련 없으면 솔직히 알려주세요.
4. 한국어로 답변하세요.
"""


def retrieve(
    query: str, reg_col, case_col, k: int = 3
) -> tuple[list[dict], list[dict]]:
    """규정 + 사고사례 양쪽에서 검색."""
    regulations = search(reg_col, query, k=k)
    cases = search(case_col, query, k=k)
    return regulations, cases


def format_contexts(
    regulations: list[dict], cases: list[dict]
) -> tuple[str, list[str]]:
    """검색 결과를 LLM 프롬프트용 컨텍스트로 포맷.

    Returns:
        tuple: (formatted_context_string, list_of_raw_context_texts)
    """
    parts: list[str] = []
    context_texts: list[str] = []
    idx = 1

    for reg in regulations:
        meta = reg["metadata"]
        source = meta.get("source", "")
        article = meta.get("article_number", "")
        title = meta.get("article_title", "")
        content = reg["content"]

        header = f"[{idx}] {source} {article}"
        if title:
            header += f" ({title})"

        display = content[:500] + "..." if len(content) > 500 else content
        parts.append(f"{header}\n{display}")
        context_texts.append(content)
        idx += 1

    for case in cases:
        meta = case["metadata"]
        title = meta.get("title", "사고사례")
        industry = meta.get("industry", "")
        hazard = meta.get("hazard_type", "")
        content = case["content"]

        header = f"[{idx}] 사고사례: {title}"
        if industry or hazard:
            tags = "/".join(filter(None, [industry, hazard]))
            header += f" ({tags})"

        display = content[:500] + "..." if len(content) > 500 else content
        parts.append(f"{header}\n{display}")
        context_texts.append(content)
        idx += 1

    return "\n\n".join(parts), context_texts


def generate_stream(query: str, context_str: str, api_key: str):
    """LLM 스트리밍 답변 생성 (OpenRouter). Yields str chunks."""
    client = OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)

    user_message = f"""## 질문
{query}

## 검색된 정보
{context_str}

위 정보를 바탕으로 질문에 답변해 주세요."""

    stream = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        stream=True,
        temperature=0.1,
        max_tokens=1024,
    )

    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content
