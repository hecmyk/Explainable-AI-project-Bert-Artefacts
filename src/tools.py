"""LangChain tools exposing DQ artifacts and the proposal queue to the agent."""
import json

import pandas as pd
from langchain_core.tools import tool

from dq.bootstrap import PROJECT_ROOT
from dq import proposals
from dq import loader

_FENGO_CACHE = {}


def _artifacts_dir():
    """Path to the artifacts directory."""
    return PROJECT_ROOT / "artifacts"


def _load_rules():
    """Load rules.json, always returning a list of specs."""
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


def _load_coverage_df():
    return pd.read_csv(_artifacts_dir() / "coverage.csv")


def _get_fengo():
    """Load and cache (df_fengo, df_bdd, df_dqrc)."""
    if "data" not in _FENGO_CACHE:
        _FENGO_CACHE["data"] = loader.load_data()
    return _FENGO_CACHE["data"]


def _spec_get(spec, keys, default=None):
    """Return the first present key from a spec dict."""
    for k in keys:
        if isinstance(spec, dict) and k in spec:
            return spec[k]
    return default


# --- Read tools ---

@tool
def list_rules(column: str = "", dimension: str = "") -> str:
    """List active rules, optionally filtered by column and/or dimension (substring, case-insensitive)."""
    rules = _load_rules()
    out = []
    for spec in rules:
        col = _spec_get(spec, ["column", "target_column"], "")
        dim = _spec_get(spec, ["dimension"], "")
        if column and column.lower() not in str(col).lower():
            continue
        if dimension and dimension.lower() not in str(dim).lower():
            continue
        rid = _spec_get(spec, ["id", "rule_id", "name"])
        typ = _spec_get(spec, ["type", "rule_type"])
        desc = _spec_get(spec, ["description", "desc"])
        if rid is None and col == "" and dim == "":
            out.append(spec)
        else:
            out.append({"rule_id": rid, "column": col, "dimension": dim,
                        "type": typ, "description": desc})
    truncated = len(out) > 50
    out = out[:50]
    result = json.dumps(out, default=str)
    if truncated:
        result += "\n... (truncated)"
    return result


@tool
def list_uncovered_columns() -> str:
    """List FENGO columns not covered by any rule."""
    df_fengo, _, _ = _get_fengo()
    fengo_cols = set(df_fengo.columns)
    rules = _load_rules()
    covered = {str(_spec_get(s, ["column", "target_column"], "")) for s in rules}
    uncovered = sorted(c for c in fengo_cols if c not in covered)
    return json.dumps(uncovered)


@tool
def list_columns_without_definition() -> str:
    """List FENGO columns with no entry (by Name) in the Business Data Dictionary."""
    df_fengo, df_bdd, _ = _get_fengo()
    defined = {str(n).strip().lower() for n in df_bdd["Name"].dropna()}
    missing = [c for c in df_fengo.columns if str(c).strip().lower() not in defined]
    return json.dumps(missing)


@tool
def profile_column(column: str) -> str:
    """Profile a FENGO column: nulls, uniques, top values, dtype, samples."""
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
    return json.dumps(profile)


@tool
def get_failures(rule_id: str, limit: int = 10) -> str:
    """Return up to `limit` violation records for a rule."""
    df = _load_run_records_df()
    rows = df[df["rule_id"] == rule_id].head(limit)
    return json.dumps(rows.to_dict(orient="records"), default=str)


@tool
def get_last_run_summary() -> str:
    """Summarize the last pipeline run: global DQI, per-dimension DQI, worst rules, rule count."""
    dqi = _load_dqi_df()
    run_rules = _load_run_rules_df()

    dqi_global = None
    dqi_by_dimension = {}
    if "dimension" in dqi.columns:
        val_col = next((c for c in dqi.columns if c.lower() in ("dqi", "score", "value")),
                       dqi.columns[-1])
        for _, row in dqi.iterrows():
            dim = str(row["dimension"])
            if dim.lower() in ("global", "overall", "all"):
                dqi_global = row[val_col]
            else:
                dqi_by_dimension[dim] = row[val_col]

    worst = []
    if "pass_rate" in run_rules.columns:
        cols = [c for c in ("rule_id", "column", "dimension", "pass_rate")
                if c in run_rules.columns]
        worst = run_rules.sort_values("pass_rate").head(5)[cols].to_dict(orient="records")

    summary = {
        "dqi_global": dqi_global,
        "dqi_by_dimension": dqi_by_dimension,
        "worst_5_rules": worst,
        "n_rules_run": int(len(run_rules)),
    }
    return json.dumps(summary, default=str)


# --- Write tools ---

@tool
def propose_add_rule(spec_json: str, rationale: str) -> str:
    """Propose adding a new rule. spec_json is a JSON string describing the rule."""
    spec = json.loads(spec_json)
    prop = proposals.add_proposal("add_rule", {"spec": spec}, rationale)
    return f"Proposal {prop['id']} created (add_rule)."


@tool
def propose_modify_rule(rule_id: str, new_spec_json: str, rationale: str) -> str:
    """Propose modifying an existing rule. new_spec_json is a JSON string."""
    new_spec = json.loads(new_spec_json)
    prop = proposals.add_proposal(
        "modify_rule", {"rule_id": rule_id, "new_spec": new_spec}, rationale)
    return f"Proposal {prop['id']} created (modify_rule)."


@tool
def propose_delete_rule(rule_id: str, rationale: str) -> str:
    """Propose deleting a rule."""
    prop = proposals.add_proposal("delete_rule", {"rule_id": rule_id}, rationale)
    return f"Proposal {prop['id']} created (delete_rule)."


@tool
def propose_definition(column: str, name: str, definition: str, example: str, rationale: str) -> str:
    """Propose a Business Data Dictionary definition for a column."""
    payload = {"column": column, "name": name, "definition": definition, "example": example}
    prop = proposals.add_proposal("definition", payload, rationale)
    return f"Proposal {prop['id']} created (definition)."


@tool
def propose_remediation(record_id: str, column: str, suggested_value: str,
                        source: str, confidence: float, rationale: str) -> str:
    """Propose a corrected value for a failing record."""
    payload = {"record_id": record_id, "column": column, "suggested_value": suggested_value,
               "source": source, "confidence": confidence}
    prop = proposals.add_proposal("remediation", payload, rationale)
    return f"Proposal {prop['id']} created (remediation)."


ALL_TOOLS = [list_rules, list_uncovered_columns, list_columns_without_definition,
             profile_column, get_failures, get_last_run_summary,
             propose_add_rule, propose_modify_rule, propose_delete_rule,
             propose_definition, propose_remediation]
