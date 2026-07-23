"""Streamlit POC for the DQ agent: dashboard, rule engine, chat, review queue."""
import json
import subprocess
from pathlib import Path

import pandas as pd
import streamlit as st

from dq.bootstrap import PROJECT_ROOT, load_config
from dq import proposals
from dq.agent.graph import build_agent, run_agent

ARTIFACTS = PROJECT_ROOT / "artifacts"
st.set_page_config(page_title="DQ Agent POC", layout="wide")


@st.cache_resource
def get_agent():
    return build_agent()


tab_dashboard, tab_engine, tab_chat, tab_review = st.tabs(
    ["Dashboard", "Rule Engine", "Agent Chat", "Review Queue"])


# --- Tab 1: Dashboard ---
with tab_dashboard:
    st.title("Data Quality Dashboard")

    dqi_path = ARTIFACTS / "run_dqi.csv"
    coverage_path = ARTIFACTS / "coverage.csv"
    rules_path = ARTIFACTS / "rules.json"

    if not dqi_path.exists() or not rules_path.exists():
        st.info("No run yet. Go to Rule Engine tab.")
    else:
        dqi = pd.read_csv(dqi_path)
        val_col = next((c for c in dqi.columns if c.lower() in ("dqi", "score", "value")),
                       dqi.columns[-1])
        globals_mask = dqi["dimension"].astype(str).str.lower().isin(
            ["global", "overall", "all"]) if "dimension" in dqi.columns else pd.Series(
            [False] * len(dqi))

        if globals_mask.any():
            st.metric("DQI Global", round(float(dqi[globals_mask][val_col].iloc[0]), 2))

        per_dim = dqi[~globals_mask] if "dimension" in dqi.columns else dqi
        if len(per_dim):
            cols = st.columns(min(4, len(per_dim)))
            for i, (_, row) in enumerate(per_dim.iterrows()):
                cols[i % len(cols)].metric(str(row.get("dimension", f"dim{i}")),
                                           round(float(row[val_col]), 2))

        rules = json.loads(rules_path.read_text())
        rules_list = rules.get("rules", rules) if isinstance(rules, dict) else rules
        st.metric("Active rules", len(rules_list))

        if coverage_path.exists():
            cov = pd.read_csv(coverage_path)
            covered_cols = set(str(c) for c in cov.get(cov.columns[0], []))
            st.metric("Columns in coverage matrix", len(covered_cols))
            st.subheader("Coverage matrix")
            st.dataframe(cov)


# --- Tab 2: Rule Engine ---
with tab_engine:
    st.title("Rule Engine")

    if st.button("Run pipeline"):
        with st.spinner("Running pipeline..."):
            proc = subprocess.run(
                ["python", str(PROJECT_ROOT / "scripts" / "run_pipeline.py")],
                capture_output=True, text=True)
        st.code((proc.stdout or "") + "\n" + (proc.stderr or ""))

    run_rules_path = ARTIFACTS / "run_rules.csv"
    run_records_path = ARTIFACTS / "run_records.csv"

    if run_rules_path.exists():
        st.subheader("Run rules")
        rr = pd.read_csv(run_rules_path)
        c1, c2 = st.columns(2)
        dim_opts = ["(all)"] + sorted(rr["dimension"].dropna().unique().tolist()) \
            if "dimension" in rr.columns else ["(all)"]
        col_opts = ["(all)"] + sorted(rr["column"].dropna().unique().tolist()) \
            if "column" in rr.columns else ["(all)"]
        dim_f = c1.selectbox("Dimension", dim_opts, key="rr_dim")
        col_f = c2.selectbox("Column", col_opts, key="rr_col")
        view = rr
        if dim_f != "(all)":
            view = view[view["dimension"] == dim_f]
        if col_f != "(all)":
            view = view[view["column"] == col_f]
        st.dataframe(view)

    if run_records_path.exists():
        st.subheader("Run records (violations)")
        rec = pd.read_csv(run_records_path)
        c1, c2 = st.columns(2)
        rcol_opts = ["(all)"] + sorted(rec["column"].dropna().unique().tolist()) \
            if "column" in rec.columns else ["(all)"]
        rcol_f = c1.selectbox("Column", rcol_opts, key="rec_col")
        view = rec
        if rcol_f != "(all)":
            view = view[view["column"] == rcol_f]
        st.dataframe(view)


# --- Sidebar ---
with st.sidebar:
    if st.button("Clear chat"):
        st.session_state["chat"] = []
    n_pending = len(proposals.list_proposals(status="pending"))
    st.metric("Pending proposals", n_pending)


# --- Tab 3: Agent Chat ---
with tab_chat:
    st.title("Agent Chat")
    st.session_state.setdefault("chat", [])

    for msg in st.session_state.chat:
        st.chat_message(msg["role"]).write(msg["content"])

    prompt = st.chat_input("Ask the agent...")
    if prompt:
        st.session_state.chat.append({"role": "user", "content": prompt})
        result = run_agent(get_agent(), prompt, history=st.session_state.chat[:-1])
        reply = result["messages"][-1].content
        st.session_state.chat.append({"role": "assistant", "content": reply})
        st.rerun()


# --- Tab 4: Review Queue ---
with tab_review:
    st.title("Review Queue")
    st.caption("Approved proposals are logged but not auto-applied to rules.json in this POC.")

    status = st.radio("Status", ["pending", "approved", "rejected", "all"], horizontal=True)
    props = proposals.list_proposals(status=None if status == "all" else status)

    for prop in props:
        with st.expander(f"{prop['id']} — {prop['kind']}"):
            st.write(f"**Status:** {prop['status']}")
            st.write(f"**Created:** {prop['created_at']}")
            st.write(f"**Rationale:** {prop['rationale']}")
            st.json(prop["payload"])
            if prop["status"] == "pending":
                c1, c2 = st.columns(2)
                if c1.button("Approve", key=f"app_{prop['id']}"):
                    proposals.approve(prop["id"])
                    st.rerun()
                if c2.button("Reject", key=f"rej_{prop['id']}"):
                    proposals.reject(prop["id"])
                    st.rerun()
