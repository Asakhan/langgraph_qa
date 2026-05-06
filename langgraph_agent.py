"""
langgraph_agent.py
==================
LangGraph 실험 코드
논문: 대화형 문서 QA에서 에이전트 오케스트레이션 구조에 따른 정확도와 운영 비용 비교
담당: 주승현 (제3저자)

[실행 방법]
    python langgraph_agent.py

[출력 파일]
    results/results_langgraph.jsonl  ← 실험 결과 (run_log_schema 형식)

[오케스트레이션 구조]
    START
      ↓
    agent  ─── tool 호출 필요 ──→  tools
      ↑                               ↓
      └──────────── 결과 반환 ────────┘
      ↓
    END (tool 호출 없이 최종 답변 생성)

    - StateGraph + 조건부 엣지(tools_condition) 기반 그래프 구조
    - 상태(AgentState)를 통해 메시지 히스토리를 노드 간 전달
    - recursion_limit=3으로 루프 횟수 제한
"""

from __future__ import annotations

import csv
import json
import os
import re
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Sequence, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage, BaseMessage, HumanMessage, SystemMessage
)
from langchain_core.tools import tool as lc_tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
import operator

load_dotenv()

# ── 경로 설정 ──────────────────────────────────────
BASE_DIR    = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)
OUTPUT_FILE = RESULTS_DIR / "results_langgraph.jsonl"

# ── 공통 Tool API import ──────────────────────────
from tool_api import (
    retrieve_document,
    calculate,
    get_question_list,
    get_gold_answers,
    make_run_id,
)

# ── 실험 설정 (통제 조건 — 세 프레임워크 동일하게 고정) ──
FRAMEWORK        = "LangGraph"
MODEL_NAME       = "gemini-2.5-flash"
TEMPERATURE      = 0
MAX_TOKENS       = 1024
RECURSION_LIMIT  = 25
RUNS_PER_Q       = 3
RETRY_LIMIT      = 5
RETRY_DELAY      = 60


# ═══════════════════════════════════════════════════
# 1. 상태 정의 (State Schema)
#    messages: 노드 간 공유되는 메시지 히스토리
#    각 노드가 반환한 메시지를 operator.add로 누적
# ═══════════════════════════════════════════════════

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


# ═══════════════════════════════════════════════════
# 2. 시스템 프롬프트 (common_system_prompt.md와 동일)
# ═══════════════════════════════════════════════════

SYSTEM_PROMPT = """You are an expert Document Question-Answering (Document QA) agent.
Your task is to read the provided document context and answer the given question
accurately, concisely, and in the required JSON format.

# Primary Role
- Use retrieve_document_tool to get the document for the given question_id.
- Perform calculations using calculate_tool if the question requires numeric reasoning.
- Generate a final answer strictly grounded in the document.

# Constraints
- Do NOT use any knowledge outside the provided document context.
- Do NOT fabricate evidence or make assumptions beyond what the document states.
- Your FINAL response must be ONLY a valid JSON object — no markdown fences, no extra text.

# Task Types
- single_doc_extraction: Extract facts directly from text. No calculation needed.
- complex_condition_judgment: Summarize conditions/procedures. 2 sentences max.
- single_evidence_numeric: Single arithmetic operation on table values.
- multi_step_calculation: Multi-step calculation. Show each step clearly.

# Output Format (return ONLY this JSON, nothing else)
{
  "question_id":  "<from input>",
  "framework":    "LangGraph",
  "dataset":      "<from input>",
  "task_type":    "<from input>",
  "final_answer": "<answer string with units>",
  "evidence":     ["<quote from doc 1>"],
  "calculation":  "<Step 1: A - B = C> or null",
  "confidence":   0.95,
  "is_answered":  true
}

# Rules for Korean Admin QA (실험1·2)
- Answer in KOREAN.
- Use exact text from the document for single_doc_extraction.
- Keep original units (천원, 원, date format).

# Rules for FinQA (실험3·4)
- Answer in ENGLISH.
- single_evidence_numeric: "5829 - 5735 = 94"
- multi_step_calculation: "Step 1: 153.7 - 139.9 = 13.8 / Step 2: 13.8 / 139.9 = 0.09864"
- Round ratios to 5 decimal places.

# Prohibited
- Do NOT output text outside the JSON.
- Do NOT use external knowledge.
- Do NOT list multiple answers in final_answer."""


# ═══════════════════════════════════════════════════
# 3. Tool 정의
#    run_id를 클로저로 주입하기 위해 팩토리 함수로 생성
# ═══════════════════════════════════════════════════

def _build_tools(run_id: str) -> list:

    @lc_tool
    def retrieve_document_tool(question_id: str) -> str:
        """
        question_id에 해당하는 문서 전체 텍스트와 메타데이터를 반환한다.
        반드시 이 도구를 먼저 호출하여 문서를 가져온 뒤 답변을 생성한다.
        Args:
            question_id: 질문 ID (예: 'Q001')
        """
        result = retrieve_document(question_id, framework=FRAMEWORK, run_id=run_id)
        return json.dumps(result, ensure_ascii=False)

    @lc_tool
    def calculate_tool(expression: str, steps: list[str] | None = None) -> str:
        """
        수식을 안전하게 계산하고 결과를 반환한다.
        단일: expression='5829 - 5735'
        다단계: steps=['153.7 - 139.9', '#0 / 139.9'] 형태로 전달.
        #N은 N번째 단계(0-indexed) 결과를 참조한다.
        Args:
            expression: 계산할 수식 문자열
            steps: 다단계 수식 리스트 (선택)
        """
        result = calculate(expression, steps=steps, framework=FRAMEWORK, run_id=run_id)
        return json.dumps(result, ensure_ascii=False)

    return [retrieve_document_tool, calculate_tool]


# ═══════════════════════════════════════════════════
# 4. 그래프 구성
#    agent 노드 → (조건부) tools 노드 → agent 노드 반복
#    recursion_limit으로 최대 반복 횟수 제한
# ═══════════════════════════════════════════════════

def _build_graph(llm_with_tools, tools: list):
    """StateGraph를 구성하고 컴파일된 앱을 반환한다."""

    def call_agent(state: AgentState) -> dict:
        """agent 노드: LLM을 호출하여 다음 행동(tool 호출 또는 최종 답변)을 결정한다."""
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    tool_node = ToolNode(tools)

    graph = StateGraph(AgentState)
    graph.add_node("agent", call_agent)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        tools_condition,          # tool_calls 있으면 tools, 없으면 END
    )
    graph.add_edge("tools", "agent")

    return graph.compile()


# ═══════════════════════════════════════════════════
# 5. 정답 판정 (langchain_agent.py / crewai_agent.py와 동일)
# ═══════════════════════════════════════════════════

def _normalize(text: str) -> str:
    t = str(text).strip()
    t = re.sub(r"[,\s]", "", t)
    t = re.sub(r"[%$원천만억]", "", t)
    return t.lower()


def _judge_success(expected: str, generated: str, answer_type: str) -> int:
    if not generated or generated.strip() == "":
        return 0

    if answer_type in ("numeric_calculation", "numeric_multi_step"):
        nums = re.findall(r"-?\d+\.?\d*", generated.replace(",", ""))
        if not nums:
            return 0
        try:
            exp_val = float(expected)
            for n in nums:
                gen_val  = float(n)
                abs_diff = abs(gen_val - exp_val)
                rel_diff = abs_diff / abs(exp_val) if exp_val != 0 else abs_diff
                tol_abs  = 0.5 if abs(exp_val) >= 1 else 0.01
                if abs_diff <= tol_abs or rel_diff <= 0.01:
                    return 1
            return 0
        except ValueError:
            return 0

    exp_norm = _normalize(expected)
    gen_norm = _normalize(generated)
    if exp_norm in gen_norm or gen_norm in exp_norm:
        return 1

    if answer_type == "span_with_condition":
        keywords = [w for w in re.split(r"\s+|[,()·]", expected) if len(w) >= 2]
        matched  = sum(1 for k in keywords if _normalize(k) in gen_norm)
        if keywords and matched / len(keywords) >= 0.5:
            return 1

    return 0


# ═══════════════════════════════════════════════════
# 6. 단일 실행 함수
# ═══════════════════════════════════════════════════

def _extract_content(msg) -> str:
    """AIMessage의 content를 문자열로 추출한다. (gemini-2.5-flash는 리스트로 올 수 있음)"""
    c = getattr(msg, "content", "")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        parts = [p.get("text", "") if isinstance(p, dict) else str(p) for p in c]
        return "".join(parts)
    return str(c)


def run_single(
    question_item: dict,
    run_index: int,
    gold_answers: dict,
    gold_types: dict,
    llm,
) -> dict:
    """질문 1건 × 1회 실행 → run_log_schema 딕셔너리 반환."""

    qid       = question_item["question_id"]
    run_id    = make_run_id(qid, FRAMEWORK, run_index)
    t_start   = time.perf_counter()
    timestamp = datetime.now(timezone.utc).isoformat()

    input_tokens     = 0
    output_tokens    = 0
    tool_calls_count = 0
    generated_answer = ""
    error_message    = None
    agent_response   = None

    try:
        # ── 도구 및 그래프 생성 (매 실행마다 새로 생성 → 히스토리 초기화) ──
        tools          = _build_tools(run_id)
        llm_with_tools = llm.bind_tools(tools)
        app            = _build_graph(llm_with_tools, tools)

        # ── 초기 메시지 구성 ──────────────────────────
        user_content = json.dumps({
            "question_id": qid,
            "task_type":   question_item["task_type"],
            "dataset":     question_item["dataset"],
            "question":    question_item["question"],
            "instruction": (
                f"Call retrieve_document_tool with question_id='{qid}' to get the document, "
                "then answer the question and return ONLY the JSON output."
            ),
        }, ensure_ascii=False)

        initial_state = {
            "messages": [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=user_content),
            ]
        }

        # ── 그래프 실행 ──────────────────────────────
        final_state = app.invoke(
            initial_state,
            config={"recursion_limit": RECURSION_LIMIT},
        )

        # ── 결과 추출 ───────────────────────────────
        messages = final_state.get("messages", [])

        # 토큰 집계 — 모든 AIMessage의 usage_metadata 합산
        for msg in messages:
            if isinstance(msg, AIMessage):
                um = getattr(msg, "usage_metadata", None)
                if um:
                    input_tokens  += um.get("input_tokens", 0)
                    output_tokens += um.get("output_tokens", 0)
                # tool_calls 횟수 집계
                if getattr(msg, "tool_calls", None):
                    tool_calls_count += len(msg.tool_calls)

        # 마지막 AIMessage에서 최종 답변 추출
        final_content = ""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                c = _extract_content(msg)
                if c and not getattr(msg, "tool_calls", None):
                    final_content = c
                    break

        # JSON 파싱
        cleaned = re.sub(r"```(?:json)?", "", final_content).replace("```", "").strip()
        try:
            agent_response   = json.loads(cleaned)
            generated_answer = agent_response.get("final_answer", "")
        except json.JSONDecodeError:
            m = re.search(r'"final_answer"\s*:\s*"([^"]+)"', cleaned)
            generated_answer = m.group(1) if m else cleaned[:200]
            agent_response   = {"raw": cleaned}

    except Exception as exc:
        error_message = traceback.format_exc()
        print(f"    ⚠ 오류: {str(exc)[:80]}")

    exec_time    = round(time.perf_counter() - t_start, 3)
    total_tokens = input_tokens + output_tokens
    expected     = gold_answers.get(qid, "")
    answer_type  = gold_types.get(qid, "span_extraction")
    success      = _judge_success(expected, generated_answer, answer_type)

    return {
        "run_id"            : run_id,
        "question_id"       : qid,
        "framework"         : FRAMEWORK,
        "dataset"           : question_item["dataset"],
        "task_type"         : question_item["task_type"],
        "run_index"         : run_index,
        "expected_answer"   : expected,
        "generated_answer"  : generated_answer,
        "success"           : success,
        "input_tokens"      : input_tokens,
        "output_tokens"     : output_tokens,
        "total_tokens"      : total_tokens,
        "execution_time_sec": exec_time,
        "tool_calls"        : tool_calls_count,
        "timestamp"         : timestamp,
        "agent_response"    : agent_response,
        "error_message"     : error_message,
        "manual_review"     : None,
    }


# ═══════════════════════════════════════════════════
# 7. 메인 실험 루프
# ═══════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("LangGraph 실험 시작")
    print("=" * 60)

    # ── API 키 확인 ──────────────────────────────
    api_key = os.getenv("GOOGLE_API_KEY", "")
    if not api_key:
        raise ValueError(".env 파일에 GOOGLE_API_KEY가 없습니다.")

    os.environ["GOOGLE_API_KEY"] = api_key

    # ── LLM 초기화 ──────────────────────────────
    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        max_output_tokens=MAX_TOKENS,
    )

    # ── 데이터 로드 ──────────────────────────────
    questions  = get_question_list()
    gold_ans   = get_gold_answers()
    gold_types: dict[str, str] = {}
    with open(BASE_DIR / "gold_answers.csv", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            gold_types[row["question_id"]] = row["answer_type"]

    total_runs = len(questions) * RUNS_PER_Q
    print(f"총 {len(questions)}문항 × {RUNS_PER_Q}회 반복 = {total_runs}회 실행")
    print(f"결과 저장: {OUTPUT_FILE}")
    print()

    # ── 완료/실패 run_id 분리 (재시작 지원) ────────
    completed_run_ids: set[str] = set()
    failed_run_ids:    set[str] = set()

    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if rec.get("success") == 1:
                        completed_run_ids.add(rec["run_id"])
                    else:
                        failed_run_ids.add(rec["run_id"])
                except Exception:
                    pass

        # 실패 기록 제거 → 재실행 대상
        if failed_run_ids:
            all_records = []
            with open(OUTPUT_FILE, encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        if rec.get("run_id") not in failed_run_ids:
                            all_records.append(line)
                    except Exception:
                        pass
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                f.writelines(all_records)
            print(f"실패 기록 제거 후 재실행 대상: {len(failed_run_ids)}건")

        if completed_run_ids:
            print(f"이미 성공한 실행 {len(completed_run_ids)}건 → 건너뜀")

    # ── 실험 루프 ────────────────────────────────
    for q in questions:
        qid = q["question_id"]
        for run_idx in range(1, RUNS_PER_Q + 1):
            run_id = make_run_id(qid, FRAMEWORK, run_idx)

            if run_id in completed_run_ids:
                continue

            print(f"[{run_id}] 실행 중...", end=" ", flush=True)

            log = None
            for attempt in range(RETRY_LIMIT + 1):
                try:
                    log = run_single(q, run_idx, gold_ans, gold_types, llm)
                    break
                except Exception as exc:
                    err_msg = str(exc)
                    if attempt < RETRY_LIMIT:
                        wait = RETRY_DELAY if "503" in err_msg or "429" in err_msg else 10
                        print(f"\n  재시도 {attempt+1}/{RETRY_LIMIT} ({wait}초 대기)...", end=" ")
                        time.sleep(wait)
                    else:
                        log = {
                            "run_id"            : run_id,
                            "question_id"       : qid,
                            "framework"         : FRAMEWORK,
                            "dataset"           : q["dataset"],
                            "task_type"         : q["task_type"],
                            "run_index"         : run_idx,
                            "expected_answer"   : gold_ans.get(qid, ""),
                            "generated_answer"  : "",
                            "success"           : 0,
                            "input_tokens"      : 0,
                            "output_tokens"     : 0,
                            "total_tokens"      : 0,
                            "execution_time_sec": 0.0,
                            "tool_calls"        : 0,
                            "timestamp"         : datetime.now(timezone.utc).isoformat(),
                            "agent_response"    : None,
                            "error_message"     : err_msg,
                            "manual_review"     : None,
                        }

            with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(log, ensure_ascii=False) + "\n")

            icon = "✓" if log["success"] == 1 else "✗"
            print(
                f"{icon} "
                f"({log['execution_time_sec']:.1f}초, "
                f"{log['total_tokens']}tokens, "
                f"success={log['success']})"
            )

            time.sleep(3)

    print()
    print("=" * 60)
    print("실험 완료")
    print("=" * 60)
    _print_summary()


def _print_summary():
    if not OUTPUT_FILE.exists():
        return

    records = []
    with open(OUTPUT_FILE, encoding="utf-8") as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except Exception:
                pass

    if not records:
        return

    total     = len(records)
    success   = sum(r["success"] for r in records)
    tsr       = success / total if total else 0
    toks      = [r["total_tokens"] for r in records if r["total_tokens"] > 0]
    avg_tok   = sum(toks) // len(toks) if toks else 0
    avg_time  = sum(r["execution_time_sec"] for r in records) / total if total else 0
    tok_per_s = sum(r["total_tokens"] for r in records) / success if success else float("inf")

    print(f"총 실행       : {total}건")
    print(f"성공 (TSR)    : {success}/{total} = {tsr:.4f} ({tsr*100:.1f}%)")
    print(f"평균 토큰     : {avg_tok:,} tokens/query")
    print(f"평균 실행시간 : {avg_time:.2f} sec/query")
    print(f"Tokens/Success: {tok_per_s:.0f}")
    print()

    from collections import defaultdict
    task_stats: dict[str, list] = defaultdict(list)
    for r in records:
        task_stats[r["task_type"]].append(r["success"])

    print("과업 유형별 TSR:")
    for task, results in sorted(task_stats.items()):
        t = sum(results) / len(results)
        print(f"  {task:35s}: {t:.3f} ({sum(results)}/{len(results)})")

    print()
    print(f"결과 파일: {OUTPUT_FILE}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--summary":
        print("=== LangGraph 실험 결과 요약 ===")
        _print_summary()
    else:
        main()
