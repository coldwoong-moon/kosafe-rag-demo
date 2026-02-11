"""ChromaDB 벡터 스토어 — 기본 임베딩 + 키워드 하이브리드 검색."""

# Streamlit Cloud의 SQLite가 ChromaDB 요구 버전보다 낮을 수 있음
try:
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import json
import re
from pathlib import Path

import chromadb
import streamlit as st

DATA_DIR = Path(__file__).parent.parent / "data"
PERSIST_DIR = str(Path(__file__).parent.parent / "chroma_db")
BATCH_SIZE = 500

# 한국어 조사/어미 — 키워드 추출 시 제거
_KO_PARTICLES = frozenset(
    "은 는 이 가 을 를 에 의 와 과 로 으로 에서 까지 부터 도 만 "
    "하는 하고 하며 하면 하여 해서 했을 에서의 으로의 인가 인지 때 중".split()
)


def _load_jsonl(path: Path) -> list[dict]:
    docs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs


@st.cache_resource(show_spinner=False)
def build_vectorstore():
    """벡터 스토어 빌드 또는 로드.

    Returns:
        tuple: (regulations_collection, cases_collection)
    """
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    reg_col = client.get_or_create_collection("regulations")
    case_col = client.get_or_create_collection("cases")

    # 이미 인덱싱 완료된 경우 즉시 반환
    if reg_col.count() > 0 and case_col.count() > 0:
        return reg_col, case_col

    # --- 규정 인덱싱 ---
    if reg_col.count() == 0:
        regs = _load_jsonl(DATA_DIR / "regulations.jsonl")
        progress = st.progress(0, text=f"규정 {len(regs):,}건 임베딩 생성 중...")
        for i in range(0, len(regs), BATCH_SIZE):
            batch = regs[i : i + BATCH_SIZE]
            reg_col.upsert(
                ids=[r["id"] for r in batch],
                documents=[r.get("content", "") for r in batch],
                metadatas=[
                    {
                        "source": r.get("source", ""),
                        "law_type": r.get("law_type", ""),
                        "article_number": r.get("article_number", ""),
                        "article_title": r.get("article_title", ""),
                        "chapter": r.get("chapter", ""),
                    }
                    for r in batch
                ],
            )
            progress.progress(min((i + BATCH_SIZE) / len(regs), 1.0))
        progress.empty()

    # --- 사고사례 인덱싱 ---
    if case_col.count() == 0:
        raw_cases = _load_jsonl(DATA_DIR / "cases.jsonl")
        seen_ids: set[str] = set()
        cases = []
        for c in raw_cases:
            if c["id"] not in seen_ids and (
                c.get("accident_summary") or c.get("accident_detail")
            ):
                seen_ids.add(c["id"])
                cases.append(c)
        progress = st.progress(0, text=f"사고사례 {len(cases):,}건 임베딩 생성 중...")
        for i in range(0, len(cases), BATCH_SIZE):
            batch = cases[i : i + BATCH_SIZE]
            texts = []
            for c in batch:
                summary = c.get("accident_summary", "")
                detail = c.get("accident_detail", "")
                texts.append(f"{summary}\n{detail}".strip() if detail else summary)

            case_col.upsert(
                ids=[c["id"] for c in batch],
                documents=texts,
                metadatas=[
                    {
                        "title": c.get("title", ""),
                        "industry": c.get("industry", ""),
                        "hazard_type": c.get("hazard_type", ""),
                        "accident_date": c.get("accident_date", ""),
                    }
                    for c in batch
                ],
            )
            progress.progress(min((i + BATCH_SIZE) / len(cases), 1.0))
        progress.empty()

    return reg_col, case_col


def _extract_keywords(query: str) -> list[str]:
    """쿼리에서 한국어 핵심 키워드 추출 (조사/어미 제거)."""
    tokens = re.split(r"\s+", query.strip())
    keywords = []
    for token in tokens:
        # 물음표 등 특수문자 제거
        clean = re.sub(r"[?!.,;:~\-()（）]", "", token)
        if len(clean) >= 2 and clean not in _KO_PARTICLES:
            keywords.append(clean)
    return keywords


def _query_with_filter(
    collection, query: str, keyword: str, k: int
) -> dict:
    """키워드 필터링된 벡터 검색."""
    try:
        return collection.query(
            query_texts=[query],
            n_results=k,
            where_document={"$contains": keyword},
        )
    except Exception:
        return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}


def search(collection, query: str, k: int = 5) -> list[dict]:
    """하이브리드 검색: 시맨틱 + 키워드 필터링 결과 병합."""
    seen_ids: set[str] = set()
    docs: list[dict] = []

    def _add_results(results: dict) -> None:
        if not results or not results.get("documents") or not results["documents"][0]:
            return
        for i, doc_text in enumerate(results["documents"][0]):
            doc_id = results["ids"][0][i]
            if doc_id in seen_ids:
                continue
            seen_ids.add(doc_id)
            meta = results["metadatas"][0][i] if results.get("metadatas") else {}
            distance = results["distances"][0][i] if results.get("distances") else 0.0
            docs.append(
                {
                    "content": doc_text,
                    "metadata": meta,
                    "score": round(1.0 - distance, 4),
                }
            )

    # 1) 시맨틱 검색
    semantic_results = collection.query(query_texts=[query], n_results=k)
    _add_results(semantic_results)

    # 2) 키워드별 필터링 검색 (상위 3개 키워드)
    keywords = _extract_keywords(query)
    for kw in keywords[:3]:
        if len(docs) >= k * 2:
            break
        kw_results = _query_with_filter(collection, query, kw, k)
        _add_results(kw_results)

    # 점수 내림차순 정렬 후 상위 k개 반환
    docs.sort(key=lambda d: d["score"], reverse=True)
    return docs[:k]
