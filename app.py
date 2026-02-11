"""KoSafe-RAG Demo — 산업안전 AI 챗봇.

건설 안전 규정(1,248건)과 사고사례(4,114건)를 기반으로
질문에 답변하고, 답변마다 Faithfulness를 실시간 평가합니다.
"""

import os

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from rag.chain import LLM_MODEL, format_contexts, generate_stream, retrieve
from rag.evaluator import evaluate_faithfulness
from rag.vectorstore import build_vectorstore

# ── 페이지 설정 ──────────────────────────────────────────
st.set_page_config(
    page_title="KoSafe-RAG Demo",
    page_icon="\U0001f3d7\ufe0f",
    layout="wide",
)


# ── API 키 ───────────────────────────────────────────────
def _get_api_key() -> str:
    try:
        return st.secrets["OPENROUTER_API_KEY"]
    except (KeyError, FileNotFoundError):
        return os.getenv("OPENROUTER_API_KEY", "")


api_key = _get_api_key()
if not api_key:
    st.error("OPENROUTER_API_KEY가 설정되지 않았습니다.")
    st.info(
        "`.env` 파일에 `OPENROUTER_API_KEY=sk-or-...`를 추가하거나, "
        "Streamlit Cloud Secrets에 등록하세요."
    )
    st.stop()


# ── 벡터 스토어 초기화 ───────────────────────────────────
with st.spinner("벡터 인덱스 로딩 중..."):
    reg_col, case_col = build_vectorstore()


# ── 사이드바 ─────────────────────────────────────────────
with st.sidebar:
    st.title("\U0001f3d7\ufe0f KoSafe-RAG")
    st.caption("산업안전 AI 어시스턴트")

    st.divider()

    st.subheader("데이터")
    c1, c2 = st.columns(2)
    c1.metric("규정", f"{reg_col.count():,}건")
    c2.metric("사고사례", f"{case_col.count():,}건")

    st.subheader("모델")
    st.text(f"LLM: {LLM_MODEL}")
    st.text("Embedding: all-MiniLM-L6-v2")
    st.text("평가: Faithfulness (실시간)")
    st.caption("via OpenRouter")

    st.divider()

    st.subheader("평가 이력")
    if "eval_scores" not in st.session_state:
        st.session_state.eval_scores = []

    if st.session_state.eval_scores:
        st.bar_chart(
            {"Faithfulness": st.session_state.eval_scores},
            height=150,
        )
        avg = sum(st.session_state.eval_scores) / len(
            st.session_state.eval_scores
        )
        st.caption(
            f"평균: {avg:.2f}  ({len(st.session_state.eval_scores)}회)"
        )
    else:
        st.caption("질문을 입력하면 평가 점수가 기록됩니다.")

    st.divider()
    st.caption("KoSafe-RAG Research Project")
    st.caption("건설 안전 규정 + 사고사례 기반 RAG")


# ── 메인 채팅 ────────────────────────────────────────────
st.title("산업안전 AI 어시스턴트")
st.caption(
    "건설 안전 규정과 사고사례를 기반으로 답변합니다. "
    "답변마다 Faithfulness를 실시간 평가합니다."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

# 기존 메시지 렌더링
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("검색 출처"):
                st.markdown(msg["sources"])
        if msg.get("score") is not None:
            _s = msg["score"]
            _e = (
                "\U0001f7e2" if _s >= 0.7
                else "\U0001f7e1" if _s >= 0.4
                else "\U0001f534"
            )
            st.caption(f"Faithfulness: {_e} {_s:.2f}")

# 사용자 입력 처리
if prompt := st.chat_input("산업안전에 관해 질문하세요..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # 1) 검색
        with st.spinner("관련 정보 검색 중..."):
            regulations, cases = retrieve(prompt, reg_col, case_col, k=3)
            context_str, context_texts = format_contexts(regulations, cases)

        # 2) 출처 표시
        if context_str:
            with st.expander("검색 출처", expanded=False):
                st.markdown(context_str)

        # 3) 스트리밍 답변
        full_response = st.write_stream(
            generate_stream(prompt, context_str, api_key)
        )

        # 4) Faithfulness 평가
        score = None
        if context_texts and full_response:
            with st.spinner("답변 신뢰도 평가 중..."):
                try:
                    score = evaluate_faithfulness(
                        full_response, context_texts, api_key
                    )
                    emoji = (
                        "\U0001f7e2" if score >= 0.7
                        else "\U0001f7e1" if score >= 0.4
                        else "\U0001f534"
                    )
                    st.caption(f"Faithfulness: {emoji} {score:.2f}")
                    st.session_state.eval_scores.append(score)
                except Exception as e:
                    st.caption(f"평가 실패: {e}")

        # 5) 히스토리 저장
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": full_response,
                "sources": context_str or None,
                "score": score,
            }
        )
