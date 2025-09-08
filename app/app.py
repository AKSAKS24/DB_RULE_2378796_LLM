from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import os, re, json

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Need OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-nano")

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

app = FastAPI(title="SAP Note 2378796 Offset Error Correction Planner")

# --------- ABAP Detection & Data Models ---------
SENSITIVE_TABLES = {"MARC"}
FORBIDDEN_FIELDS = {
    "STAWN": "Create instance of /SAPSLL/CL_MM_CLS_SERVICE and call ->GET_COMMODITY_CODE_CLS",
    "EXPME": "Create instance of /SAPSLL/CL_MM_CLS_SERVICE and call ->GET_COMMODITY_CODE_DETAILS",
}
SQL_SELECT_BLOCK_RE = re.compile(
    r"\bSELECT\b(?P<select>.+?)\bFROM\b\s+(?P<table>\w+)(?P<rest>.*?)(?=(\bSELECT\b|$))",
    re.IGNORECASE | re.DOTALL,
)
JOIN_RE = re.compile(r"\bJOIN\s+(?P<table>\w+)", re.IGNORECASE)

class Finding(BaseModel):
    pgm_name: Optional[str] = None
    inc_name: Optional[str] = None
    type: Optional[str] = None
    name: Optional[str] = None
    class_implementation: Optional[str] = None
    issue_type: Optional[str] = None
    severity: Optional[str] = None
    message: Optional[str] = None
    suggestion: Optional[str] = None
    snippet: Optional[str] = None

class Unit(BaseModel):
    pgm_name: str
    inc_name: str
    type: str
    name: Optional[str] = ""
    class_implementation: Optional[str] = ""
    start_line: Optional[int] = 0
    end_line: Optional[int] = 0
    code: Optional[str] = ""
    findings: Optional[List[Finding]] = Field(default=None)

def detect_code_snippets(code: Optional[str], meta: Dict[str, str]) -> List[Finding]:
    if not code:
        return []
    findings = []
    for stmt in SQL_SELECT_BLOCK_RE.finditer(code):
        table = stmt.group("table").upper()
        stmt_text = stmt.group(0)
        if table in SENSITIVE_TABLES:
            for field, sugg in FORBIDDEN_FIELDS.items():
                if re.search(rf"\b{field}\b", stmt_text, re.IGNORECASE):
                    findings.append(Finding(
                        snippet=stmt_text.strip(),
                        suggestion=sugg,
                        pgm_name=meta.get('pgm_name'), inc_name=meta.get('inc_name'),
                        type=meta.get('type'), name=meta.get('name'),
                        class_implementation=meta.get('class_implementation'), 
                        issue_type="offset error", severity="high",
                        message=f"Direct usage of {field} in {table} not allowed by SAP Note 2378796."
                    ))
        for jm in JOIN_RE.finditer(stmt.group("rest")):
            jtable = jm.group("table").upper()
            if jtable in SENSITIVE_TABLES:
                j_text = stmt.group("rest")
                for field, sugg in FORBIDDEN_FIELDS.items():
                    if re.search(rf"\b{field}\b", j_text, re.IGNORECASE):
                        findings.append(Finding(
                            snippet=j_text.strip(),
                            suggestion=sugg,
                            pgm_name=meta.get('pgm_name'), inc_name=meta.get('inc_name'),
                            type=meta.get('type'), name=meta.get('name'),
                            class_implementation=meta.get('class_implementation'),
                            issue_type="offset error", severity="high",
                            message=f"Direct usage of {field} in joined {jtable} not allowed by SAP Note 2378796."
                        ))
    return findings

# -------- LLM PROMPT --------
SYSTEM_MSG = """
You are a senior ABAP expert. Output ONLY JSON as response.
For every provided payload .findings[].snippet,
write a bullet point that:
- Displays the exact offending code
- Explains the necessary action to fix the offset error using the provided .message text (if available).
- Bullet points should contain both offending code snippet and the fix (no numbering or referencing like "snippet[1]": display the code inline).
- Do NOT omit any snippet; all must be covered, no matter how many there are.
- Only show actual ABAP code for each snippet with its specific action.
""".strip()

USER_TEMPLATE = """
Unit metadata:
Program: {pgm_name}
Include: {inc_name}
Unit type: {unit_type}
Unit name: {unit_name}
Start line: {start_line}
End line: {end_line}

ABAP code context (optional):
{code}

findings (JSON list of findings, each with .snippet and .message if present for offset errors):
{findings_json}

Instructions:
1. Write a 1-paragraph assessment summarizing offset-related SAP Note 2378796 risks in human language.
2. Write a llm_prompt field: for every finding, add a bullet point with
   - The exact code (snippet field)
   - The action required for correction (taken from suggestion field, if any).
   - Do not compress, omit, or refer to them by index; always display code inline.

Return valid JSON with:
{{
  "assessment": "<paragraph>",
  "llm_prompt": "<action bullets>"
}}
""".strip()

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_MSG),
    ("user", USER_TEMPLATE),
])
llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.0)
parser = JsonOutputParser()
chain = prompt | llm | parser

def llm_assess_and_prompt(unit: Unit) -> Dict[str, str]:
    findings_json = json.dumps([f.model_dump() for f in (unit.findings or [])], ensure_ascii=False, indent=2)
    try:
        return chain.invoke({
            "pgm_name": unit.pgm_name,
            "inc_name": unit.inc_name,
            "unit_type": unit.type,
            "unit_name": unit.name or "",
            "start_line": unit.start_line or 0,
            "end_line": unit.end_line or 0,
            "code": unit.code or "",
            "findings_json": findings_json,
        })
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM call failed: {e}")

@app.post("/assess-2378796")
async def assess_2378796_snippet(units: List[Unit]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for u in units:
        meta = {
            'pgm_name': u.pgm_name, 'inc_name': u.inc_name,
            'type': u.type, 'name': u.name,
            'class_implementation': getattr(u, 'class_implementation', None),
        }
        u.findings = detect_code_snippets(u.code, meta)

        # Strict: skip negative/empty scenario
        if not u.findings:
            continue

        obj = u.model_dump()
        llm_out = llm_assess_and_prompt(u)
        obj["assessment"] = llm_out.get("assessment", "")
        prompt_out = llm_out.get("llm_prompt", "")
        if isinstance(prompt_out, list):
            prompt_out = "\n".join(str(x) for x in prompt_out if x is not None)
        obj["llm_prompt"] = prompt_out
        obj.pop("findings", None)
        out.append(obj)
    return out

@app.get("/health")
def health():
    return {"ok": True, "note": "2378796", "model": OPENAI_MODEL}