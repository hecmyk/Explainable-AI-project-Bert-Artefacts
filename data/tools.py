"""LangChain tools exposing DQ artifacts and the proposal queue to the agent."""
import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from langchain_core.tools import tool

from dq.bootstrap import PROJECT_ROOT, load_config
from dq import proposals
from dq import loader

_FENGO_CACHE = {}


def _artifacts_dir():
    """Path to the artifacts directory."""
    return Path(PROJECT_ROOT) / "artifacts"


def _log(tool_name, args, result, t0):
    """Append a tool call record to artifacts/agent_calls.jsonl and return result unchanged."""
    duration_ms = int((time.time() - t0) * 1000)
    log_path = _artifacts_dir() / "agent_calls.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps({
            "ts": datetime.now().isoformat(timespec="seconds"),
            "tool": tool_name,
            "args": args,
            "duration_ms": duration_ms,
            "result_preview": str(result)[:200],
        }) + "\n")
    return result


def _load_rules():
    """Load rules.json, always returning a list of specs (handles both dict-with-rules and list-directly)."""
    data = json.loads((_artifacts_dir() / "rules.json").read_text())
    if isinstance(data, dict):
        return data.get("rules", [])
    return data


def _load_run_rules_df():
    return pd.read_csv(_artifacts_dir() / "run_rules.csv")


def _load_run_records_df():
    return pd.read_csv(_artifacts_dir() / "run_records.csv")


def _load_dqi_df():
    return pd.read_csv(_artifacts_dir() / "run_dqi.csv")


def _get_fengo():
    """Load and cache (df_fengo, df_bdd, df_dqrc) via loader.load(cfg)."""
    if "data" not in _FENGO_CACHE:
        cfg = load_config()
        _FENGO_CACHE["data"] = loader.load(cfg)
    return _FENGO_CACHE["data"]


def _spec_columns(spec):
    """Extract the list of target columns from a spec, handling both scalar and list forms."""
    cols = spec.get("target_columns") or spec.get("column") or spec.get("target_column")
    if isinstance(cols, str):
        return [cols]
    return cols or []


def _spec_type(spec):
    """Extract the check type from a spec."""
    if isinstance(spec.get("check"), dict):
        return spec["check"].get("type")
    return spec.get("type") or spec.get("rule_type")


# --- Read tools ---

@tool
def list_rules(column: str = "", check_type: str = "") -> str:
    """List active rules, optionally filtered by column (substring) and check type."""
    t0 = time.time()
    rules = _load_rules()
    out = []
    for spec in rules:
        cols = _spec_columns(spec)
        typ = _spec_type(spec)
        if column and not any(column.lower() in str(c).lower() for c in cols):
            continue
        if check_type and check_type.lower() != str(typ).lower():
            continue
        out.append({
            "rule_id": spec.get("rule_id") or spec.get("id") or spec.get("name"),
            "target_columns": cols,
            "check_type": typ,
            "review_status": spec.get("review_status"),
            "source_text": spec.get("source_text") or spec.get("description"),
        })
    truncated = len(out) > 50
    out = out[:50]
    result = json.dumps(out, default=str)
    if truncated:
        result += "\n... (truncated)"
    return _log("list_rules", {"column": column, "check_type": check_type}, result, t0)


@tool
def list_uncovered_columns() -> str:
    """List FENGO columns not covered by any rule."""
    t0 = time.time()
    df_fengo, _, _ = _get_fengo()
    fengo_cols = set(df_fengo.columns)
    covered = set()
    for spec in _load_rules():
        for c in _spec_columns(spec):
            covered.add(str(c))
    uncovered = sorted(c for c in fengo_cols if c not in covered)
    result = json.dumps(uncovered)
    return _log("list_uncovered_columns", {}, result, t0)


@tool
def list_columns_without_definition() -> str:
    """List FENGO columns with no entry (by Name) in the Business Data Dictionary."""
    t0 = time.time()
    df_fengo, df_bdd, _ = _get_fengo()
    defined = {str(n).strip().lower() for n in df_bdd["Name"].dropna()}
    missing = [c for c in df_fengo.columns if str(c).strip().lower() not in defined]
    result = json.dumps(missing)
    return _log("list_columns_without_definition", {}, result, t0)


@tool
def profile_column(column: str) -> str:
    """Profile a FENGO column: nulls, uniques, top values, dtype, samples."""
    t0 = time.time()
    df_fengo, _, _ = _get_fengo()
    s = df_fengo[column]
    n_total = int(len(s))
    n_null = int(s.isna().sum())
    non_null = s.dropna()
    profile = {
        "n_total": n_total,
        "n_null": n_null,
        "pct_null": round(n_null / n_total * 100, 2) if n_total else 0.0,
        "n_unique": int(s.nunique(dropna=True)),
        "top_5_values": {str(k): int(v) for k, v in s.value_counts().head(5).items()},
        "dtype": str(s.dtype),
        "sample_values": [str(v) for v in non_null.drop_duplicates().head(5).tolist()],
    }
    result = json.dumps(profile)
    return _log("profile_column", {"column": column}, result, t0)


@tool
def get_failures(rule_id: str, limit: int = 10) -> str:
    """Return up to `limit` violation records for a rule."""
    t0 = time.time()
    df = _load_run_records_df()
    id_col = "rule_id" if "rule_id" in df.columns else df.columns[0]
    rows = df[df[id_col] == rule_id].head(limit)
    result = json.dumps(rows.to_dict(orient="records"), default=str)
    return _log("get_failures", {"rule_id": rule_id, "limit": limit}, result, t0)


@tool
def get_last_run_summary() -> str:
    """Summarize the last pipeline run: global DQI, per-dimension DQI, worst rules, rule count."""
    t0 = time.time()
    try:
        dqi = _load_dqi_df()
    except FileNotFoundError:
        return _log("get_last_run_summary", {}, "No run yet.", t0)
    run_rules = _load_run_rules_df()

    dqi_global = None
    dqi_by_dimension = {}
    if "dimension" in dqi.columns:
        val_col = next((c for c in dqi.columns if c.lower() in ("dqi", "score", "value", "rate")),
                       dqi.columns[-1])
        for _, row in dqi.iterrows():
            dim = str(row["dimension"])
            if dim.lower() in ("global", "overall", "all"):
                dqi_global = row[val_col]
            else:
                dqi_by_dimension[dim] = row[val_col]

    worst = []
    for cand in ("pass_rate", "rate", "failed"):
        if cand in run_rules.columns:
            asc = cand != "failed"  # for "failed" we want highest first
            cols = [c for c in ("rule_id", "column", "dimension", cand) if c in run_rules.columns]
            worst = run_rules.sort_values(cand, ascending=asc).head(5)[cols].to_dict(orient="records")
            break

    summary = {
        "dqi_global": dqi_global,
        "dqi_by_dimension": dqi_by_dimension,
        "worst_5_rules": worst,
        "n_rules_run": int(len(run_rules)),
    }
    result = json.dumps(summary, default=str)
    return _log("get_last_run_summary", {}, result, t0)


# --- Write tools ---

@tool
def propose_add_rule(spec_json: str, rationale: str) -> str:
    """Propose adding a new rule. spec_json is a JSON string with target_columns, check{type,params}, source_text."""
    t0 = time.time()
    spec = json.loads(spec_json)
    prop = proposals.add_proposal("add_rule", {"spec": spec}, rationale)
    result = f"Proposal {prop['id']} created (add_rule)."
    return _log("propose_add_rule", {"rationale": rationale}, result, t0)


@tool
def propose_modify_rule(rule_id: str, new_spec_json: str, rationale: str) -> str:
    """Propose modifying an existing rule."""
    t0 = time.time()
    new_spec = json.loads(new_spec_json)
    prop = proposals.add_proposal("modify_rule", {"rule_id": rule_id, "new_spec": new_spec}, rationale)
    result = f"Proposal {prop['id']} created (modify_rule)."
    return _log("propose_modify_rule", {"rule_id": rule_id}, result, t0)


@tool
def propose_delete_rule(rule_id: str, rationale: str) -> str:
    """Propose deleting a rule."""
    t0 = time.time()
    prop = proposals.add_proposal("delete_rule", {"rule_id": rule_id}, rationale)
    result = f"Proposal {prop['id']} created (delete_rule)."
    return _log("propose_delete_rule", {"rule_id": rule_id}, result, t0)


@tool
def propose_definition(column: str, name: str, definition: str, example: str, rationale: str) -> str:
    """Propose a Business Data Dictionary definition for a column."""
    t0 = time.time()
    payload = {"column": column, "name": name, "definition": definition, "example": example}
    prop = proposals.add_proposal("definition", payload, rationale)
    result = f"Proposal {prop['id']} created (definition)."
    return _log("propose_definition", {"column": column}, result, t0)


ALL_TOOLS = [list_rules, list_uncovered_columns, list_columns_without_definition,
             profile_column, get_failures, get_last_run_summary,
             propose_add_rule, propose_modify_rule, propose_delete_rule,
             propose_definition]
