"""Data Quality Agent — merged Streamlit app (6 tabs)."""
import json
import os

import pandas as pd
import streamlit as st

from dq import loader, executor, report, audit, proposals
from dq.bootstrap import load_config
from dq.agent.graph import build_agent, stream_agent

st.set_page_config(page_title="Data Quality Agent", layout="wide")


# --- Sidebar ---
with st.sidebar:
    if st.button("Clear chat"):
        st.session_state["chat"] = []
    n_pending = len(proposals.list_proposals(status="pending"))
    st.metric("Pending agent proposals", n_pending)


@st.cache_resource
def boot():
    cfg = load_config()
    fengo, bdd, dqrc = loader.load(cfg)
    idx = loader.build_column_index(fengo, bdd)
    links = loader.link_rules_to_columns(dqrc, idx)
    return cfg, fengo, bdd, dqrc, idx, links


@st.cache_resource
def get_agent():
    return build_agent()


cfg, fengo, bdd, dqrc, idx, links = boot()
ART = cfg["paths"]["artifacts"]
RULES = os.path.join(ART, "rules.json")
KEY = cfg["data"]["record_key"]

load = lambda: json.load(open(RULES))
save = lambda s: json.dump(s, open(RULES, "w"), indent=2, default=str)


def concern(s):
    """Higher = needs a human sooner."""
    d = s.get("dry_run") or {}
    if s["compile_status"] != "compiled":
        return 0
    if d.get("error"):
        return 1000
    r = d.get("rate")
    if r is None:
        return 400
    if r > 0.20:
        return 500 + r
    if d.get("revised"):
        return 300
    return r


def fails(spec, n=10):
    """Return (sample failing rows, error) for a spec."""
    try:
        v = executor.evaluate(fengo, spec)
    except Exception as e:
        return None, str(e)
    cols = [KEY] + [c for c in (spec.get("target_columns") or []) if c in fengo.columns]
    return fengo.loc[v == executor.FAIL, cols].head(n), None


specs = load()

tab_r, tab_c, tab_run, tab_f, tab_chat, tab_props = st.tabs(
    ["Review & approve", "Coverage", "Run & reports", "Findings",
     "Agent Chat", "Agent Proposals"]
)

# --- Review & approve ---
with tab_r:
    st.title("Review & approve")

    pending = sorted(
        [s for s in specs if s.get("review_status") == "pending"],
        key=concern, reverse=True,
    )
    st.metric("Pending rules", len(pending))

    if not pending:
        st.info("No pending rules to review.")
    else:
        overview = pd.DataFrame([{
            "rule_id": s["rule_id"],
            "columns": ", ".join(s.get("target_columns") or []),
            "check": (s.get("check") or {}).get("type"),
            "compile_status": s.get("compile_status"),
            "rate": (s.get("dry_run") or {}).get("rate"),
            "concern": round(concern(s), 3),
        } for s in pending])
        st.dataframe(overview, width="stretch", height=260)

        if st.button("Approve all plausible"):
            for s in pending:
                if s["compile_status"] == "compiled" and not (s.get("dry_run") or {}).get("error"):
                    s["review_status"] = "approved"
                    audit.record("approve", "demo", rule=s["rule_id"])
            save(specs)
            st.rerun()

        chosen = st.selectbox("Inspect rule", [s["rule_id"] for s in pending])
        spec = next(s for s in pending if s["rule_id"] == chosen)

        st.subheader("Catalog text")
        st.write(spec.get("source_text", ""))

        st.subheader("Compiled check")
        edited = st.text_area("check (JSON)", json.dumps(spec.get("check", {}), indent=2), height=200)

        dry = spec.get("dry_run") or {}
        if dry:
            st.subheader("Dry run")
            st.json(dry)

        st.subheader("Sample violations")
        sample, err = fails(spec)
        if err:
            st.error(err)
        elif sample is not None:
            st.dataframe(sample, width="stretch")

        c1, c2 = st.columns(2)
        if c1.button("Approve", type="primary"):
            try:
                spec["check"] = json.loads(edited)
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON: {e}")
            else:
                spec["review_status"] = "approved"
                save(specs)
                audit.record("approve", "demo", rule=spec["rule_id"])
                st.rerun()
        if c2.button("Reject"):
            spec["review_status"] = "rejected"
            save(specs)
            audit.record("reject", "demo", rule=spec["rule_id"])
            st.rerun()

# --- Coverage ---
with tab_c:
    cov = loader.coverage_matrix(links, fengo, idx)
    st.metric("Columns with no rule", int((cov["total"] == 0).sum()), delta=f"of {len(cov)}")
    st.dataframe(cov, width="stretch", height=560)

# --- Run & reports ---
with tab_run:
    if st.button("Run approved rules against FENGO", type="primary"):
        appr = [s for s in load() if s["review_status"] == "approved"]
        if appr:
            summ, verd = executor.run(fengo, appr, KEY)
            report.write(ART, **{
                "run_rules.csv": summ,
                "run_records.csv": report.record_view(fengo, verd, KEY),
                "run_dqi.csv": report.dqi_view(summ, appr),
                "coverage.csv": loader.coverage_matrix(links, fengo, idx),
            })
            audit.record("run", "demo", rules=len(appr))
            st.session_state["summ"] = summ
        else:
            st.warning("No approved rules to run.")

    if "summ" in st.session_state:
        st.subheader("Run summary")
        st.dataframe(st.session_state["summ"], width="stretch")

    for name in ("run_dqi.csv", "run_rules.csv", "run_records.csv"):
        path = os.path.join(ART, name)
        if os.path.exists(path):
            st.subheader(name)
            st.dataframe(pd.read_csv(path), width="stretch")

# --- Findings ---
with tab_f:
    st.markdown(
        "## Findings\n\n"
        "This section summarizes data quality observations from the latest run. "
        "Use the reports above to drill into specific rules and records."
    )

# --- Agent Chat ---
with tab_chat:
    st.title("Agent Chat")
    st.session_state.setdefault("chat", [])

    for msg in st.session_state.chat:
        st.chat_message(msg["role"]).write(msg["content"])

    prompt = st.chat_input("Ask the agent...")
    if prompt:
        st.session_state.chat.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
            status = st.status("Thinking...", expanded=True)
            final_content = ""
            try:
                for chunk in stream_agent(get_agent(), prompt,
                                          history=st.session_state.chat[:-1]):
                    for node_name, node_output in chunk.items():
                        msgs = node_output.get("messages", []) if isinstance(node_output, dict) else []
                        for m in msgs:
                            # Tool call from the agent
                            tcs = getattr(m, "tool_calls", None)
                            if tcs:
                                for tc in tcs:
                                    args_str = json.dumps(tc.get("args", {}))[:120]
                                    status.write(f"🔧 `{tc['name']}({args_str})`")
                            # Tool result
                            elif getattr(m, "type", None) == "tool":
                                name = getattr(m, "name", "tool")
                                status.write(f"📥 `{name}` returned {len(str(m.content))} chars")
                            # Final assistant message
                            elif getattr(m, "type", None) == "ai" and getattr(m, "content", ""):
                                final_content = m.content
            except Exception as e:
                status.update(label=f"Error: {e}", state="error")
                final_content = f"Error: {e}"
            else:
                status.update(label="Done", state="complete", expanded=False)
            st.write(final_content)

        st.session_state.chat.append({"role": "assistant", "content": final_content})

# --- Agent Proposals ---
with tab_props:
    st.title("Agent Proposals")
    st.caption("These are propositions from the interactive agent (chat). "
               "Approved proposals are logged but not auto-applied to rules.json in this POC.")

    status_f = st.radio("Status", ["pending", "approved", "rejected", "all"], horizontal=True)
    props = proposals.list_proposals(status=None if status_f == "all" else status_f)

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
