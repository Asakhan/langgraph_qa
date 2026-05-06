"""
tool_api.py
===========
에이전트 오케스트레이션 비교 실험 — 공통 Tool API
논문: 대화형 문서 QA에서 에이전트 오케스트레이션 구조에 따른 정확도와 운영 비용 비교
저자: 문근영·김희경·주승현·허대영 (국민대학교)

[설계 원칙]
- 세 프레임워크(LangChain Agent / CrewAI / LangGraph)가 동일한 함수를 import하여 사용한다.
- Tool 함수는 3개로 고정한다.
  1. retrieve_document  : question_id를 받아 관련 문서 전체 텍스트를 반환한다.
  2. calculate          : 수식 문자열을 받아 안전하게 계산한다.
  3. validate_output    : 에이전트 최종 응답 딕셔너리를 받아 스키마를 검증하고 JSON 문자열을 반환한다.
- 각 도구 호출은 자동으로 로그에 기록된다(call_log.jsonl).
- 도구 내부에서 LLM을 추가 호출하지 않는다.
"""

from __future__ import annotations

import json
import math
import operator
import os
import re
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ─────────────────────────────────────────────
# 0. 경로 설정
# ─────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "experiment_data"
LOG_DIR    = BASE_DIR / "logs"
LOG_FILE   = LOG_DIR / "call_log.jsonl"

DATA_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────
# 1. 실험 데이터베이스 초기화
#    questions.csv + 원본 데이터셋을 로드하여
#    question_id → document_text 매핑을 메모리에 구성한다.
# ─────────────────────────────────────────────

def _build_document_db() -> dict[str, dict]:
    """
    questions.csv와 원본 데이터셋을 결합하여
    { question_id: { question, document, task_type, dataset, ... } } 딕셔너리를 반환한다.
    """
    import csv

    db: dict[str, dict] = {}

    # ── 1-A. questions.csv 로드 ──────────────────
    q_csv = BASE_DIR / "questions.csv"
    if not q_csv.exists():
        raise FileNotFoundError(f"questions.csv를 찾을 수 없습니다: {q_csv}")

    with open(q_csv, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    # ── 1-B. 한국어 QA 원본: question → context 매핑 ──
    ko_context_map: dict[str, str] = {}
    ko_files = [
        "TL_span_extraction.json",
        "TL_span_extraction_how.json",
        "TL_tableqa.json",
        "TL_text_entailment.json",
        "TS_span_extraction.json",
        "TS_span_extraction_how.json",
        "TS_tableqa.json",
        "TS_text_entailment.json",
    ]
    ko_data_dir = DATA_DIR / "한국어데이터" / "sample"
    for fname in ko_files:
        fpath = ko_data_dir / fname
        if not fpath.exists():
            continue
        with open(fpath, encoding="utf-8") as f:
            raw = json.load(f)
        for doc in raw.get("data", []):
            for para in doc.get("paragraphs", []):
                ctx = para.get("context", "")
                for qa in para.get("qas", []):
                    ko_context_map[qa["question"]] = ctx

    # ── 1-C. FinQA 원본: filename → item 매핑 ──
    finqa_path = DATA_DIR / "FinQA-main" / "code" / "evaluate" / "test.json"
    if not finqa_path.exists():
        raise FileNotFoundError(f"FinQA test.json을 찾을 수 없습니다: {finqa_path}")

    with open(finqa_path, encoding="utf-8") as f:
        finqa_data = json.load(f)
    finqa_by_file: dict[str, dict] = {item["filename"]: item for item in finqa_data}

    # ── 1-D. question_id별 DB 구성 ──
    for row in rows:
        qid      = row["question_id"]
        dataset  = row["dataset"]
        task     = row["task_type"]
        q_ko     = row["question_ko"]
        q_en     = row["question_en"]
        src_doc  = row["source_doc"]

        if dataset == "Korean_Admin_QA":
            question  = q_ko
            context   = ko_context_map.get(q_ko, row.get("context_snippet", ""))
            document  = _format_ko_document(context, row.get("source_org", ""))
        else:  # FinQA
            question  = q_en
            finqa_item = finqa_by_file.get(src_doc, {})
            document  = _format_finqa_document(finqa_item)

        db[qid] = {
            "question_id": qid,
            "dataset":     dataset,
            "task_type":   task,
            "question":    question,
            "document":    document,
            "source_doc":  src_doc,
        }

    return db


def _format_ko_document(context: str, org: str) -> str:
    """한국어 행정 문서 컨텍스트를 에이전트가 읽기 쉬운 형태로 정제한다."""
    header = f"[출처: {org}]\n\n" if org else ""
    # HTML 테이블 태그를 마크다운 형태로 단순 변환
    text = re.sub(r"<br\s*/?>", "\n", context)
    text = re.sub(r"<td[^>]*>", " | ", text)
    text = re.sub(r"</td>", "", text)
    text = re.sub(r"<tr[^>]*>", "\n", text)
    text = re.sub(r"</tr>", "", text)
    text = re.sub(r"<[^>]+>", "", text)          # 나머지 태그 제거
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return header + text


def _format_finqa_document(item: dict) -> str:
    if not item:
        return ""
    
    parts = []

    # gold_inds만 추출 (정답 근거 문장만 전달)
    gold = item.get("qa", {}).get("gold_inds", {})
    if gold:
        parts.append("[EVIDENCE]\n" + "\n".join(gold.values()))

    # 표는 전체 유지 (수치 계산에 필요)
    table = item.get("table", [])
    if table:
        md = []
        for i, row in enumerate(table):
            md.append(" | ".join(str(c) for c in row))
            if i == 0:
                md.append(" | ".join(["---"] * len(row)))
        parts.append("\n[TABLE]\n" + "\n".join(md) + "\n[/TABLE]")

    return "\n\n".join(parts)


# 모듈 로드 시 DB를 한 번만 구성한다.
try:
    _DOC_DB: dict[str, dict] = _build_document_db()
except Exception as _e:
    _DOC_DB = {}
    print(f"[tool_api] 경고: 문서 DB 초기화 실패 — {_e}")


# ─────────────────────────────────────────────
# 2. 도구 호출 로거
# ─────────────────────────────────────────────

def _log_call(
    tool_name: str,
    framework: str,
    run_id: str,
    inputs: dict,
    output: Any,
    elapsed_sec: float,
    error: str | None = None,
) -> None:
    """도구 호출 1건을 call_log.jsonl에 한 줄로 기록한다."""
    record = {
        "timestamp":   datetime.now(timezone.utc).isoformat(),
        "run_id":      run_id,
        "tool":        tool_name,
        "framework":   framework,
        "inputs":      inputs,
        "output_summary": str(output)[:200] if output else None,
        "elapsed_sec": round(elapsed_sec, 4),
        "error":       error,
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ─────────────────────────────────────────────
# 3. Tool 1 — retrieve_document
# ─────────────────────────────────────────────

def retrieve_document(
    question_id: str,
    framework: str = "unknown",
    run_id: str = "",
) -> dict:
    """
    question_id에 해당하는 문서 전체 텍스트와 메타데이터를 반환한다.

    Parameters
    ----------
    question_id : str
        실험 질문 ID. 예: "Q001"
    framework   : str
        호출 프레임워크 식별자. 로그 기록용. 예: "LangChain"
    run_id      : str
        실행 고유 ID. 예: "Q001_LangChain_01"

    Returns
    -------
    dict
        {
          "question_id" : str,
          "dataset"     : str,    # "Korean_Admin_QA" | "FinQA"
          "task_type"   : str,
          "question"    : str,    # 질문 원문
          "document"    : str,    # 관련 문서 전체 텍스트
          "source_doc"  : str,    # 원본 파일명 또는 문서 제목
          "status"      : "ok" | "not_found"
        }

    Notes
    -----
    - 도구 호출은 질의 1건당 최대 1회로 제한한다(통제 조건 RULE-13).
    - 문서 텍스트는 원본 그대로 반환하며 추가 가공 없이 에이전트에게 전달한다.
    """
    t0 = time.perf_counter()
    inputs = {"question_id": question_id}
    error  = None

    try:
        if question_id not in _DOC_DB:
            result = {
                "question_id": question_id,
                "status":      "not_found",
                "document":    "",
                "question":    "",
                "dataset":     "",
                "task_type":   "",
                "source_doc":  "",
            }
        else:
            entry  = _DOC_DB[question_id]
            result = {**entry, "status": "ok"}

    except Exception as exc:
        error  = traceback.format_exc()
        result = {"question_id": question_id, "status": "error", "error": str(exc)}

    elapsed = time.perf_counter() - t0
    _log_call("retrieve_document", framework, run_id, inputs, result.get("status"), elapsed, error)
    return result


# ─────────────────────────────────────────────
# 4. Tool 2 — calculate
# ─────────────────────────────────────────────

# 허용 연산자 및 함수 목록 (외부 코드 실행 방지)
_SAFE_OPS: dict[str, Any] = {
    "abs":   abs,
    "round": round,
    "min":   min,
    "max":   max,
    "sum":   sum,
    "sqrt":  math.sqrt,
    "log":   math.log,
    "pow":   math.pow,
    "__builtins__": {},
}

_STEP_PATTERN = re.compile(
    r"(?:step\s*\d+\s*:\s*)?"          # "Step 1: " 접두사 (선택)
    r"([\d,.\-+*/()%\s]+)"             # 수식 (숫자·연산자·괄호)
    r"(?:\s*=\s*([\d,.\-]+))?",        # "= 결과" (선택)
    re.IGNORECASE,
)


def _safe_eval(expr: str) -> float:
    """
    수식 문자열을 안전하게 평가한다.
    eval 대신 ast + operator를 사용하여 코드 인젝션을 방지한다.
    """
    import ast

    # 콤마(천 단위 구분자) 제거, 달러·퍼센트 기호 제거
    expr = expr.replace(",", "").replace("$", "").replace("%", "").strip()

    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"수식 파싱 오류: {e}") from e

    _BINOPS = {
        ast.Add:  operator.add,
        ast.Sub:  operator.sub,
        ast.Mult: operator.mul,
        ast.Div:  operator.truediv,
        ast.Pow:  operator.pow,
        ast.Mod:  operator.mod,
    }
    _UNOPS = {
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        elif isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        elif isinstance(node, ast.BinOp):
            op_fn = _BINOPS.get(type(node.op))
            if op_fn is None:
                raise ValueError(f"지원하지 않는 연산자: {type(node.op).__name__}")
            left  = _eval(node.left)
            right = _eval(node.right)
            if op_fn is operator.truediv and right == 0:
                raise ZeroDivisionError("0으로 나눌 수 없습니다.")
            return op_fn(left, right)
        elif isinstance(node, ast.UnaryOp):
            op_fn = _UNOPS.get(type(node.op))
            if op_fn is None:
                raise ValueError(f"지원하지 않는 단항 연산자: {type(node.op).__name__}")
            return op_fn(_eval(node.operand))
        else:
            raise ValueError(f"허용되지 않는 수식 요소: {ast.dump(node)}")

    return _eval(tree)


def calculate(
    expression: str,
    steps: list[str] | None = None,
    round_digits: int = 5,
    framework: str = "unknown",
    run_id: str = "",
) -> dict:
    """
    수식 또는 다단계 계산 목록을 받아 결과를 반환한다.

    Parameters
    ----------
    expression  : str
        단일 수식 문자열. 예: "5829 - 5735"
        다단계의 경우 첫 번째 수식 또는 전체 수식을 "/" 또는 "\\n"으로 연결한 문자열.
    steps       : list[str] | None
        다단계 계산 목록. 각 원소는 "153.7 - 139.9" 형태의 수식 문자열.
        None이면 expression만 계산한다.
    round_digits : int
        결과 반올림 자릿수. 기본값 5 (논문 RULE-FIN-05 준수).
    framework   : str
        호출 프레임워크 식별자. 로그용.
    run_id      : str
        실행 고유 ID. 로그용.

    Returns
    -------
    dict
        {
          "status"         : "ok" | "error",
          "expression"     : str,          # 입력 수식 원문
          "result"         : float | None, # 최종 계산 결과
          "result_rounded" : float | None, # round_digits 자리로 반올림한 결과
          "step_results"   : list[dict],   # 각 단계별 중간 결과 (다단계 시)
          "calculation_str": str,          # 논문 출력 스키마용 문자열
          "error"          : str | None
        }

    Examples
    --------
    단일 계산:
        calculate("5829 - 5735")
        → {"result": 94.0, "calculation_str": "5829 - 5735 = 94.0"}

    다단계 계산:
        calculate("153.7 - 139.9", steps=["153.7 - 139.9", "#0 / 139.9"])
        → step_results[0]["result"] = 13.8
           step_results[1]["result"] = 0.09864
           calculation_str = "Step 1: 153.7 - 139.9 = 13.8 / Step 2: 13.8 / 139.9 = 0.09864"

    Notes
    -----
    - 다단계에서 "#N"은 N번째 단계(0-indexed)의 결과값을 참조한다.
      예: "#0 / 139.9" → step_results[0]["result"] / 139.9
    - 외부 코드 실행 없이 ast 기반 안전 평가만 허용한다.
    """
    t0     = time.perf_counter()
    inputs = {"expression": expression, "steps": steps}
    error  = None

    step_results: list[dict] = []
    calc_parts:   list[str]  = []

    try:
        if steps:
            # ── 다단계 계산 ──────────────────────────
            prev_results: list[float] = []
            for i, step_expr in enumerate(steps):
                # #N 참조 치환
                resolved = step_expr
                for ref_idx, ref_val in enumerate(prev_results):
                    resolved = resolved.replace(f"#{ref_idx}", str(ref_val))
                val   = _safe_eval(resolved)
                val_r = round(val, round_digits)
                prev_results.append(val_r)
                step_results.append({
                    "step":       i + 1,
                    "expression": resolved,
                    "result":     val_r,
                })
                calc_parts.append(f"Step {i+1}: {resolved} = {val_r}")

            final_result  = prev_results[-1]
            final_rounded = round(final_result, round_digits)

        else:
            # ── 단일 계산 ────────────────────────────
            # "/" 또는 개행으로 구분된 다중 수식은 마지막 수식만 사용
            expr_clean = expression.strip().split("/")[-1].strip()
            val        = _safe_eval(expr_clean)
            final_result  = val
            final_rounded = round(val, round_digits)
            calc_parts.append(f"{expr_clean} = {final_rounded}")

        calculation_str = " / ".join(calc_parts)
        result_dict = {
            "status":          "ok",
            "expression":      expression,
            "result":          final_result,
            "result_rounded":  final_rounded,
            "step_results":    step_results,
            "calculation_str": calculation_str,
            "error":           None,
        }

    except Exception as exc:
        error       = traceback.format_exc()
        result_dict = {
            "status":          "error",
            "expression":      expression,
            "result":          None,
            "result_rounded":  None,
            "step_results":    [],
            "calculation_str": "",
            "error":           str(exc),
        }

    elapsed = time.perf_counter() - t0
    _log_call("calculate", framework, run_id, inputs, result_dict.get("result_rounded"), elapsed, error)
    return result_dict


# ─────────────────────────────────────────────
# 5. Tool 3 — validate_output
# ─────────────────────────────────────────────

# 필수 필드 정의
_REQUIRED_FIELDS = {
    "question_id", "framework", "dataset", "task_type",
    "final_answer", "evidence", "confidence", "is_answered",
}
_OPTIONAL_FIELDS = {"calculation"}

_VALID_FRAMEWORKS = {"LangChain", "CrewAI", "LangGraph"}
_VALID_DATASETS   = {"Korean_Admin_QA", "FinQA"}
_VALID_TASK_TYPES = {
    "single_doc_extraction",
    "complex_condition_judgment",
    "single_evidence_numeric",
    "multi_step_calculation",
}


def validate_output(
    response: dict,
    framework: str = "unknown",
    run_id: str = "",
) -> dict:
    """
    에이전트 최종 응답 딕셔너리를 받아 출력 스키마를 검증하고
    정규화된 JSON 문자열을 반환한다.

    Parameters
    ----------
    response  : dict
        에이전트가 생성한 응답 딕셔너리.
        문자열로 전달된 경우 JSON 파싱을 시도한다.
    framework : str
        호출 프레임워크 식별자. 로그용.
    run_id    : str
        실행 고유 ID. 로그용.

    Returns
    -------
    dict
        {
          "status"       : "ok" | "error",
          "valid"        : bool,
          "errors"       : list[str],        # 스키마 위반 항목 목록
          "warnings"     : list[str],        # 권장 사항 위반 목록
          "output_json"  : str,              # 검증 통과 시 정규화된 JSON 문자열
          "output_dict"  : dict | None       # 검증 통과 시 딕셔너리
        }

    Validation Rules
    ----------------
    ERROR (invalid → 재시도 또는 무효 처리)
      - 필수 필드 누락
      - framework 값이 허용 목록 외
      - dataset / task_type 값이 허용 목록 외
      - evidence가 리스트가 아니거나 비어 있음
      - confidence가 0.0~1.0 범위 외
      - is_answered가 bool이 아님
      - final_answer가 빈 문자열

    WARNING (기록하되 무효 처리하지 않음)
      - evidence 항목이 3개 초과
      - final_answer 길이가 200자 초과 (너무 장문)
      - calculation이 numeric task임에도 null
    """
    t0     = time.perf_counter()
    inputs = {"run_id": run_id}
    error  = None

    errors:   list[str] = []
    warnings: list[str] = []

    try:
        # ── 문자열 → dict 파싱 ──────────────────────
        if isinstance(response, str):
            # 마크다운 펜스 제거
            cleaned = re.sub(r"```(?:json)?", "", response).replace("```", "").strip()
            response = json.loads(cleaned)

        if not isinstance(response, dict):
            raise TypeError(f"응답이 dict 또는 JSON 문자열이어야 합니다. 실제 타입: {type(response)}")

        # ── 필수 필드 존재 검사 ─────────────────────
        for field in _REQUIRED_FIELDS:
            if field not in response:
                errors.append(f"필수 필드 누락: '{field}'")

        # ── 필드별 값 검사 ──────────────────────────
        fw = response.get("framework", "")
        if fw and fw not in _VALID_FRAMEWORKS:
            errors.append(f"framework 값 오류: '{fw}' (허용: {_VALID_FRAMEWORKS})")

        ds = response.get("dataset", "")
        if ds and ds not in _VALID_DATASETS:
            errors.append(f"dataset 값 오류: '{ds}' (허용: {_VALID_DATASETS})")

        tt = response.get("task_type", "")
        if tt and tt not in _VALID_TASK_TYPES:
            errors.append(f"task_type 값 오류: '{tt}' (허용: {_VALID_TASK_TYPES})")

        ev = response.get("evidence")
        if ev is not None:
            if not isinstance(ev, list):
                errors.append("evidence는 list 타입이어야 합니다.")
            elif len(ev) == 0:
                errors.append("evidence 리스트가 비어 있습니다. 최소 1개 이상의 근거를 포함해야 합니다.")
            elif len(ev) > 3:
                warnings.append(f"evidence 항목이 {len(ev)}개입니다 (권장: 1~3개).")

        conf = response.get("confidence")
        if conf is not None:
            try:
                conf_f = float(conf)
                if not (0.0 <= conf_f <= 1.0):
                    errors.append(f"confidence 범위 오류: {conf_f} (허용: 0.0~1.0)")
                else:
                    response["confidence"] = conf_f   # float 정규화
            except (TypeError, ValueError):
                errors.append(f"confidence 타입 오류: '{conf}' (float이어야 합니다)")

        is_ans = response.get("is_answered")
        if is_ans is not None and not isinstance(is_ans, bool):
            # "true"/"false" 문자열 허용
            if str(is_ans).lower() == "true":
                response["is_answered"] = True
            elif str(is_ans).lower() == "false":
                response["is_answered"] = False
            else:
                errors.append(f"is_answered 타입 오류: '{is_ans}' (bool이어야 합니다)")

        fa = response.get("final_answer", "")
        if fa == "" or fa is None:
            errors.append("final_answer가 비어 있습니다.")
        elif len(str(fa)) > 200:
            warnings.append(f"final_answer 길이({len(str(fa))}자)가 200자를 초과합니다 (2문장 이내 권장).")

        calc = response.get("calculation")
        if tt in ("single_evidence_numeric", "multi_step_calculation") and (calc is None or calc == ""):
            warnings.append("수치 계산 과업(실험3·4)임에도 calculation 필드가 null입니다.")

        # ── 결과 조합 ───────────────────────────────
        valid = len(errors) == 0

        if valid:
            # "calculation" 필드가 없으면 null로 채움
            if "calculation" not in response:
                response["calculation"] = None

            output_json = json.dumps(response, ensure_ascii=False, indent=2)
            result_dict = {
                "status":      "ok",
                "valid":       True,
                "errors":      [],
                "warnings":    warnings,
                "output_json": output_json,
                "output_dict": response,
            }
        else:
            result_dict = {
                "status":      "ok",
                "valid":       False,
                "errors":      errors,
                "warnings":    warnings,
                "output_json": "",
                "output_dict": None,
            }

    except Exception as exc:
        error       = traceback.format_exc()
        result_dict = {
            "status":      "error",
            "valid":       False,
            "errors":      [str(exc)],
            "warnings":    [],
            "output_json": "",
            "output_dict": None,
        }

    elapsed = time.perf_counter() - t0
    _log_call("validate_output", framework, run_id, inputs, result_dict.get("valid"), elapsed, error)
    return result_dict


# ─────────────────────────────────────────────
# 6. 프레임워크별 도구 정의 헬퍼
#    각 프레임워크가 import하여 바로 사용할 수 있도록
#    LangChain Tool / CrewAI Tool / LangGraph ToolNode 형태로 래핑한다.
# ─────────────────────────────────────────────

def get_langchain_tools(framework: str = "LangChain", run_id: str = ""):
    """
    LangChain Tool 객체 리스트를 반환한다.
    AgentExecutor의 tools 인자에 직접 전달한다.

    Usage
    -----
    from tool_api import get_langchain_tools
    tools = get_langchain_tools(framework="LangChain", run_id=run_id)
    agent_executor = AgentExecutor(agent=agent, tools=tools, max_iterations=3)
    """
    try:
        from langchain.tools import Tool as LCTool

        def _retrieve(q_id: str) -> str:
            result = retrieve_document(q_id.strip(), framework=framework, run_id=run_id)
            return json.dumps(result, ensure_ascii=False)

        def _calculate(expr: str) -> str:
            # "expression|step1|step2" 형식으로 다단계 전달 허용
            parts = [p.strip() for p in expr.split("|")]
            if len(parts) > 1:
                result = calculate(parts[0], steps=parts[1:], framework=framework, run_id=run_id)
            else:
                result = calculate(parts[0], framework=framework, run_id=run_id)
            return json.dumps(result, ensure_ascii=False)

        def _validate(resp_str: str) -> str:
            try:
                resp = json.loads(resp_str)
            except Exception:
                resp = resp_str
            result = validate_output(resp, framework=framework, run_id=run_id)
            return json.dumps(result, ensure_ascii=False)

        return [
            LCTool(
                name="retrieve_document",
                func=_retrieve,
                description=(
                    "question_id를 입력받아 해당 문서 전체 텍스트와 메타데이터를 반환한다. "
                    "Input: question_id 문자열 (예: 'Q001'). "
                    "Output: JSON 문자열 (question, document, task_type, dataset 포함)."
                ),
            ),
            LCTool(
                name="calculate",
                func=_calculate,
                description=(
                    "수식을 계산한다. "
                    "단일: '5829 - 5735' 형태. "
                    "다단계: '첫수식|#0 / 139.9' 형태로 '|'로 구분하여 입력. "
                    "#N은 N번째 단계 결과를 참조. "
                    "Output: JSON 문자열 (result, calculation_str 포함)."
                ),
            ),
            LCTool(
                name="validate_output",
                func=_validate,
                description=(
                    "에이전트 최종 응답 JSON 문자열을 받아 출력 스키마를 검증한다. "
                    "Input: Section 4 출력 스키마를 따르는 JSON 문자열. "
                    "Output: JSON 문자열 (valid, errors, output_json 포함)."
                ),
            ),
        ]
    except ImportError:
        raise ImportError("langchain 패키지가 설치되어 있지 않습니다: pip install langchain")


def get_crewai_tools(framework: str = "CrewAI", run_id: str = ""):
    """
    CrewAI BaseTool 인스턴스 리스트를 반환한다.
    Agent의 tools 인자에 직접 전달한다.

    Usage
    -----
    from tool_api import get_crewai_tools
    tools = get_crewai_tools(framework="CrewAI", run_id=run_id)
    researcher = Agent(role="Researcher", tools=tools, ...)
    """
    try:
        from crewai.tools import BaseTool
        from pydantic import BaseModel, Field

        class RetrieveInput(BaseModel):
            question_id: str = Field(description="질문 ID. 예: 'Q001'")

        class CalculateInput(BaseModel):
            expression: str = Field(description="계산할 수식. 다단계는 '|'로 구분.")
            steps: list[str] | None = Field(default=None, description="다단계 수식 리스트.")

        class ValidateInput(BaseModel):
            response_json: str = Field(description="검증할 JSON 문자열.")

        class RetrieveDocumentTool(BaseTool):
            name: str = "retrieve_document"
            description: str = (
                "question_id를 입력받아 해당 문서 전체 텍스트와 메타데이터를 반환한다. "
                "반드시 이 도구를 통해서만 문서에 접근한다."
            )
            args_schema: type[BaseModel] = RetrieveInput

            def _run(self, question_id: str) -> str:
                result = retrieve_document(question_id, framework=framework, run_id=run_id)
                return json.dumps(result, ensure_ascii=False)

        class CalculateTool(BaseTool):
            name: str = "calculate"
            description: str = (
                "수식 문자열을 안전하게 계산한다. "
                "단일: '5829 - 5735'. "
                "다단계: steps=['153.7 - 139.9', '#0 / 139.9'] 형태로 전달."
            )
            args_schema: type[BaseModel] = CalculateInput

            def _run(self, expression: str, steps: list[str] | None = None) -> str:
                result = calculate(expression, steps=steps, framework=framework, run_id=run_id)
                return json.dumps(result, ensure_ascii=False)

        class ValidateOutputTool(BaseTool):
            name: str = "validate_output"
            description: str = (
                "에이전트 최종 응답 JSON 문자열의 출력 스키마를 검증한다. "
                "최종 답변을 생성한 뒤 반드시 이 도구로 검증한다."
            )
            args_schema: type[BaseModel] = ValidateInput

            def _run(self, response_json: str) -> str:
                try:
                    resp = json.loads(response_json)
                except Exception:
                    resp = response_json
                result = validate_output(resp, framework=framework, run_id=run_id)
                return json.dumps(result, ensure_ascii=False)

        return [RetrieveDocumentTool(), CalculateTool(), ValidateOutputTool()]

    except ImportError:
        raise ImportError("crewai 패키지가 설치되어 있지 않습니다: pip install crewai")


def get_langgraph_tool_functions():
    """
    LangGraph ToolNode에 등록할 함수 리스트를 반환한다.
    각 함수는 @tool 데코레이터 없이 직접 LangGraph의 tools 리스트에 전달한다.

    Usage
    -----
    from tool_api import get_langgraph_tool_functions
    from langgraph.prebuilt import ToolNode

    tool_fns = get_langgraph_tool_functions()
    tool_node = ToolNode(tool_fns)
    """
    try:
        from langchain_core.tools import tool as lc_tool

        @lc_tool
        def retrieve_document_tool(question_id: str, framework: str = "LangGraph", run_id: str = "") -> str:
            """
            question_id에 해당하는 문서 텍스트와 메타데이터를 반환한다.
            Args:
                question_id: 질문 ID (예: 'Q001')
                framework: 프레임워크 이름 (로그용)
                run_id: 실행 ID (로그용)
            """
            result = retrieve_document(question_id, framework=framework, run_id=run_id)
            return json.dumps(result, ensure_ascii=False)

        @lc_tool
        def calculate_tool(expression: str, steps: list[str] | None = None,
                           framework: str = "LangGraph", run_id: str = "") -> str:
            """
            수식을 계산한다. 다단계는 steps 리스트로 전달하고 #N으로 이전 결과를 참조한다.
            Args:
                expression: 계산할 수식 문자열
                steps: 다단계 수식 리스트 (선택)
                framework: 프레임워크 이름 (로그용)
                run_id: 실행 ID (로그용)
            """
            result = calculate(expression, steps=steps, framework=framework, run_id=run_id)
            return json.dumps(result, ensure_ascii=False)

        @lc_tool
        def validate_output_tool(response_json: str, framework: str = "LangGraph", run_id: str = "") -> str:
            """
            에이전트 최종 응답 JSON 문자열을 검증하고 정규화된 출력을 반환한다.
            Args:
                response_json: 검증할 JSON 문자열
                framework: 프레임워크 이름 (로그용)
                run_id: 실행 ID (로그용)
            """
            try:
                resp = json.loads(response_json)
            except Exception:
                resp = response_json
            result = validate_output(resp, framework=framework, run_id=run_id)
            return json.dumps(result, ensure_ascii=False)

        return [retrieve_document_tool, calculate_tool, validate_output_tool]

    except ImportError:
        raise ImportError("langchain_core 패키지가 설치되어 있지 않습니다: pip install langchain-core")


# ─────────────────────────────────────────────
# 7. 유틸리티 함수
# ─────────────────────────────────────────────

def get_question_list() -> list[dict]:
    """
    실험 전체 40문항 목록을 반환한다.
    각 프레임워크 실험 루프에서 반복 대상으로 사용한다.

    Returns
    -------
    list[dict]
        [{ question_id, dataset, task_type, question, source_doc }, ...]
    """
    return [
        {
            "question_id": qid,
            "dataset":     entry["dataset"],
            "task_type":   entry["task_type"],
            "question":    entry["question"],
            "source_doc":  entry["source_doc"],
        }
        for qid, entry in _DOC_DB.items()
    ]


def get_gold_answers() -> dict[str, str]:
    """
    gold_answers.csv를 로드하여 { question_id: expected_answer } 딕셔너리를 반환한다.
    자동 평가(Task Success Rate 계산)에서 사용한다.
    """
    import csv
    ga_csv = BASE_DIR / "gold_answers.csv"
    if not ga_csv.exists():
        return {}
    with open(ga_csv, encoding="utf-8-sig") as f:
        return {row["question_id"]: row["expected_answer"] for row in csv.DictReader(f)}


def make_run_id(question_id: str, framework: str, run_index: int) -> str:
    """표준 run_id 문자열을 생성한다. 예: 'Q001_LangChain_01'"""
    return f"{question_id}_{framework}_{run_index:02d}"


# ─────────────────────────────────────────────
# 8. 빠른 동작 확인 (직접 실행 시)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Tool API 동작 확인")
    print("=" * 60)

    # DB 로드 확인
    print(f"\n[1] 문서 DB 로드: {len(_DOC_DB)}개 문항")

    # Tool 1: retrieve_document
    print("\n[2] retrieve_document('Q001')")
    r1 = retrieve_document("Q001", framework="Test", run_id="test_001")
    print(f"  status   : {r1['status']}")
    print(f"  dataset  : {r1.get('dataset')}")
    print(f"  task_type: {r1.get('task_type')}")
    print(f"  question : {r1.get('question', '')[:60]}")
    print(f"  document : {r1.get('document', '')[:80]}...")

    print("\n[3] retrieve_document('Q021') — FinQA")
    r2 = retrieve_document("Q021", framework="Test", run_id="test_002")
    print(f"  status   : {r2['status']}")
    print(f"  dataset  : {r2.get('dataset')}")
    print(f"  question : {r2.get('question', '')[:80]}")
    print(f"  document : {r2.get('document', '')[:120]}...")

    # Tool 2: calculate
    print("\n[4] calculate('5829 - 5735')  — 단일")
    r3 = calculate("5829 - 5735", framework="Test", run_id="test_003")
    print(f"  result          : {r3['result']}")
    print(f"  calculation_str : {r3['calculation_str']}")

    print("\n[5] calculate — 다단계 (Step 1: 153.7-139.9, Step 2: #0/139.9)")
    r4 = calculate("153.7 - 139.9", steps=["153.7 - 139.9", "#0 / 139.9"],
                   framework="Test", run_id="test_004")
    print(f"  step1 result    : {r4['step_results'][0]['result']}")
    print(f"  step2 result    : {r4['step_results'][1]['result']}")
    print(f"  calculation_str : {r4['calculation_str']}")

    # Tool 3: validate_output
    print("\n[6] validate_output — 정상 케이스")
    good_resp = {
        "question_id":  "Q001",
        "framework":    "LangChain",
        "dataset":      "Korean_Admin_QA",
        "task_type":    "single_doc_extraction",
        "final_answer": "44.9%",
        "evidence":     ["70대 이상 스마트폰 보유율은 44.9% 불과 ('20년)"],
        "calculation":  None,
        "confidence":   0.98,
        "is_answered":  True,
    }
    r5 = validate_output(good_resp, framework="Test", run_id="test_005")
    print(f"  valid  : {r5['valid']}")
    print(f"  errors : {r5['errors']}")

    print("\n[7] validate_output — 오류 케이스 (필드 누락 + 범위 오류)")
    bad_resp = {
        "question_id":  "Q001",
        "framework":    "UnknownFW",
        "dataset":      "Korean_Admin_QA",
        "task_type":    "single_doc_extraction",
        "final_answer": "",
        "evidence":     [],
        "confidence":   1.5,
        "is_answered":  True,
    }
    r6 = validate_output(bad_resp, framework="Test", run_id="test_006")
    print(f"  valid  : {r6['valid']}")
    print(f"  errors : {r6['errors']}")

    print("\n[8] 전체 문항 목록 — 앞 3개")
    for q in get_question_list()[:3]:
        print(f"  {q['question_id']} | {q['dataset']} | {q['task_type']}")

    print(f"\n[9] 콜 로그 저장 위치: {LOG_FILE}")
    print("\n모든 동작 확인 완료.")
