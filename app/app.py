# app_2378796_strict.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any, Optional
import os, re, json

# ---- LLM Setup ----
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable is required.")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")  # safer default

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

app = FastAPI(title="SAP Note 2378796 Assessment - Strict Schema")

# ===== Tables & Fields impacted by Note 2378796 =====
SENSITIVE_TABLES = {"MARC"}
FORBIDDEN_FIELDS = {
    "STAWN": "Create instance of /SAPSLL/CL_MM_CLS_SERVICE and call ->GET_COMMODITY_CODE_CLS",
    "EXPME": "Create instance of /SAPSLL/CL_MM_CLS_SERVICE and call ->GET_COMMODITY_CODE_DETAILS",
}

# ===== Regex patterns =====
SQL_SELECT_BLOCK_RE = re.compile(
    r"\bSELECT\b(?P<select>.+?)\bFROM\b\s+(?P<table>\w+)(?P<rest>.*?)(?=(\bSELECT\b|$))",
    re.IGNORECASE | re.DOTALL,
)
JOIN_RE = re.compile(r"\bJOIN\s+(?P<table>\w+)", re.IGNORECASE)

# ===== Strict Models =====
class SelectItem(BaseModel):
    table: str
    target_type: str
    target_name: str
    used_fields: List[str]
    suggested_fields: List[str]
    suggested_statement: str

    @field_validator("used_fields", "suggested_fields")
    @classmethod
    def filter_none(cls, v: List[str]) -> List[str]:
        return [x for x in v if x is not None]


class Unit(BaseModel):
    pgm_name: str
    inc_name: str
    type: str
    name: str
    code: Optional[str] = ""
    selects: List[SelectItem] = Field(default_factory=list)


# ===== Detection logic =====
def comment_sql_field(f: str) -> str:
    return f"* TODO: Field {f} must NOT be read directly from MARC (SAP Note 2378796). {FORBIDDEN_FIELDS[f]} instead."


def parse_and_build_selectitems(code: str) -> List[SelectItem]:
    results: List[SelectItem] = []

    for stmt in SQL_SELECT_BLOCK_RE.finditer(code):
        table = stmt.group("table").upper()
        stmt_text = stmt.group(0)

        # Only relevant for MARC
        if table in SENSITIVE_TABLES:
            for field in FORBIDDEN_FIELDS.keys():
                if re.search(rf"\b{field}\b", stmt_text, re.IGNORECASE):
                    results.append(
                        SelectItem(
                            table=table,
                            target_type="SQL_FIELD",
                            target_name=field,
                            used_fields=[field],
                            suggested_fields=[],
                            suggested_statement=comment_sql_field(field)
                        )
                    )

        # Handle JOINS with MARC
        for jm in JOIN_RE.finditer(stmt.group("rest")):
            jtable = jm.group("table").upper()
            if jtable in SENSITIVE_TABLES:
                j_text = stmt.group("rest")
                for field in FORBIDDEN_FIELDS.keys():
                    if re.search(rf"\b{field}\b", j_text, re.IGNORECASE):
                        results.append(
                            SelectItem(
                                table=jtable,
                                target_type="SQL_FIELD",
                                target_name=field,
                                used_fields=[field],
                                suggested_fields=[],
                                suggested_statement=comment_sql_field(field)
                            )
                        )
    return results


# ===== Summariser =====
def summarize_selects(unit: Unit) -> Dict[str, Any]:
    field_count: Dict[str, int] = {}
    flagged = []
    for s in unit.selects:
        for f in s.used_fields:
            field_count[f.upper()] = field_count.get(f.upper(), 0) + 1
            flagged.append({"field": f, "reason": s.suggested_statement})
    return {
        "program": unit.pgm_name,
        "include": unit.inc_name,
        "unit_type": unit.type,
        "unit_name": unit.name,
        "stats": {
            "total_occurrences": len(unit.selects),
            "fields_frequency": field_count,
            "note_2378796_flags": flagged
        }
    }


# ===== LLM prompt =====
SYSTEM_MSG = "You are a precise ABAP reviewer familiar with SAP Note 2378796. Output strict JSON only."
USER_TEMPLATE = """
You are assessing ABAP code usage in light of SAP Note 2378796 (do NOT read MARC fields STAWN/EXPME directly).

We provide metadata and findings (under "selects").
Tasks:
1) Produce a concise assessment of the impact.
2) Produce an actionable LLM remediation prompt to insert TODO comments in ABAP code.

Return ONLY strict JSON:
{{
  "assessment": "<concise 2378796 impact>",
  "llm_prompt": "<remediation prompt>"
}}

Unit metadata:
- Program: {pgm}
- Include: {inc}
- Unit type: {utype}
- Unit name: {uname}

Summary:
{plan_json}

Selects (JSON findings):
{selects_json}
""".strip()

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_MSG),
    ("user", USER_TEMPLATE)
])
llm = ChatOpenAI(model=OPENAI_MODEL)
parser = JsonOutputParser()
chain = prompt | llm | parser


def llm_assess_and_prompt(unit: Unit) -> Dict[str, str]:
    plan_json = json.dumps(summarize_selects(unit), indent=2)
    selects_json = json.dumps([s.model_dump() for s in unit.selects], indent=2)
    try:
        return chain.invoke({
            "pgm": unit.pgm_name,
            "inc": unit.inc_name,
            "utype": unit.type,
            "uname": unit.name,
            "plan_json": plan_json,
            "selects_json": selects_json
        })
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM failed: {e}")


# ===== API Endpoint =====
@app.post("/assess-2378796")
async def assess_note_2378796(units: List[Unit]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for u in units:
        if u.code:
            u.selects = parse_and_build_selectitems(u.code)

        # ✅ Skip if no restricted field usage detected
        if not u.selects:
            continue

        # Otherwise, process with LLM
        llm_out = llm_assess_and_prompt(u)
        obj = u.model_dump()
        obj.pop("selects", None)
        obj.pop("code", None)
        obj["assessment"] = llm_out.get("assessment", "")
        obj["llm_prompt"] = llm_out.get("llm_prompt", "")
        out.append(obj)
    return out


@app.get("/health")
def health():
    return {"ok": True, "note": "2378796", "model": OPENAI_MODEL}