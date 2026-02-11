"""ChromaDB 벡터 스토어 — 기본 임베딩 (all-MiniLM-L6-v2, 로컬, 무료)."""

# Streamlit Cloud의 SQLite가 ChromaDB 요구 버전보다 낮을 수 있음
try:
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import json
from pathlib import Path

import chromadb
import streamlit as st

DATA_DIR = Path(__file__).parent.parent / "data"
PERSIST_DIR = str(Path(__file__).parent.parent / "chroma_db")
BATCH_SIZE = 500


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

    ChromaDB 기본 임베딩(all-MiniLM-L6-v2)을 사용합니다.
    API 키 불필요, 로컬에서 무료로 동작합니다.

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
            reg_col.add(
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
        # 중복 ID 제거 (첫 번째 등장만 유지)
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

            case_col.add(
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


def search(collection, query: str, k: int = 5) -> list[dict]:
    """컬렉션에서 유사도 검색."""
    results = collection.query(query_texts=[query], n_results=k)

    docs = []
    if results and results.get("documents") and results["documents"][0]:
        for i, doc_text in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][i] if results.get("metadatas") else {}
            distance = (
                results["distances"][0][i] if results.get("distances") else 0.0
            )
            docs.append(
                {
                    "content": doc_text,
                    "metadata": meta,
                    "score": round(1.0 - distance, 4),
                }
            )
    return docs
