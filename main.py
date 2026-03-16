"""
ETF 에이전트 - LangGraph 기반
사용자가 입력한 ETF 이름을 Gemini LLM이 한국어로 간단히 설명합니다.
"""

import os
import re
import time
from typing import TypedDict

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

# .env에서 GEMINI_API_KEY 로드
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY가 .env 파일에 설정되어 있지 않습니다.")


# 그래프 상태 정의
class ETFAgentState(TypedDict):
    etf_name: str
    explanation: str


def _extract_retry_after_seconds(text: str) -> int | None:
    # e.g. "Please retry in 51.2189s." or "retryDelay': '51s'"
    m = re.search(r"retry in\s+([0-9]+(?:\.[0-9]+)?)s", text, flags=re.IGNORECASE)
    if m:
        return max(0, int(float(m.group(1))))
    m = re.search(r"retryDelay'\s*:\s*'(\d+)s'", text)
    if m:
        return max(0, int(m.group(1)))
    return None


def explain_etf_node(state: ETFAgentState) -> dict:
    """ETF 이름을 받아 Gemini로 한국어 설명을 생성하는 노드."""
    etf_name = state["etf_name"]

    llm = ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
        google_api_key=GEMINI_API_KEY,
        temperature=0.3,
    )

    system_prompt = """당신은 ETF(상장지수펀드) 전문가입니다.
사용자가 입력한 ETF 이름에 대해 한국어로 2~4문장 정도로 간단하고 이해하기 쉽게 설명해 주세요.
해당 ETF가 어떤 자산/지수를 추종하는지, 어떤 투자자에게 적합한지 핵심만 요약해 주세요."""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"다음 ETF를 한국어로 간단히 설명해 주세요: {etf_name}"),
    ]

    try:
        response = llm.invoke(messages)
        explanation = response.content if hasattr(response, "content") else str(response)
        return {"explanation": explanation}
    except ChatGoogleGenerativeAIError as e:
        msg = str(e)
        if ("RESOURCE_EXHAUSTED" in msg) or ("429" in msg):
            retry_after = _extract_retry_after_seconds(msg)
            if retry_after is not None and retry_after > 0:
                time.sleep(min(retry_after, 60))
                # 한 번만 재시도
                response = llm.invoke(messages)
                explanation = response.content if hasattr(response, "content") else str(response)
                return {"explanation": explanation}

            return {
                "explanation": (
                    "현재 Gemini API 쿼터/요금제 설정 문제로 호출이 차단되어 응답을 생성할 수 없습니다(429 RESOURCE_EXHAUSTED).\n"
                    "- 해결: Google AI Studio/콘솔에서 **프로젝트 과금(결제) 연결**, **Gemini API 사용 설정**, **Rate limit/Quota**를 확인하세요.\n"
                    "- 잠시 후 다시 시도하거나, `GEMINI_MODEL`을 다른 모델로 바꿔도(예: gemini-1.5-flash) 쿼터가 0이면 동일합니다.\n"
                    f"- 입력한 ETF: {etf_name}"
                )
            }
        raise

def build_etf_agent():
    """노드 1개짜리 LangGraph 그래프를 구성하고 컴파일합니다."""
    workflow = StateGraph(ETFAgentState)

    workflow.add_node("explain_etf", explain_etf_node)
    workflow.add_edge(START, "explain_etf")
    workflow.add_edge("explain_etf", END)

    return workflow.compile()


def main():
    agent = build_etf_agent()

    print("=== ETF 에이전트 (LangGraph) ===")
    print("종료하려면 'q' 또는 'quit'를 입력하세요.\n")

    while True:
        etf_name = input("ETF 이름을 입력하세요: ").strip()
        if not etf_name:
            continue
        if etf_name.lower() in ("q", "quit"):
            print("종료합니다.")
            break

        try:
            result = agent.invoke({"etf_name": etf_name, "explanation": ""})
            print("\n[설명]\n", result["explanation"], "\n")
        except Exception as e:
            print("\n[오류]\n", str(e), "\n")


if __name__ == "__main__":
    main()
