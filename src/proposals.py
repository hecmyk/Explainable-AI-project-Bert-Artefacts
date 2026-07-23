"""Proposal queue persisted to artifacts/proposals.json."""
import json
from datetime import datetime

from dq.bootstrap import PROJECT_ROOT

_PATH = PROJECT_ROOT / "artifacts" / "proposals.json"


def _ensure_file():
    """Create an empty proposals file if it does not exist."""
    if not _PATH.exists():
        _PATH.parent.mkdir(parents=True, exist_ok=True)
        _PATH.write_text(json.dumps({"proposals": []}, indent=2))


def load_proposals():
    """Return the list of proposals."""
    _ensure_file()
    return json.loads(_PATH.read_text())["proposals"]


def save_proposals(proposals):
    """Write the proposals list to disk."""
    _PATH.parent.mkdir(parents=True, exist_ok=True)
    _PATH.write_text(json.dumps({"proposals": proposals}, indent=2))


def add_proposal(kind, payload, rationale):
    """Create a new pending proposal and persist it."""
    proposals = load_proposals()
    n = max((int(p["id"].split("_")[1]) for p in proposals), default=0) + 1
    prop = {
        "id": f"prop_{n:04d}",
        "kind": kind,
        "status": "pending",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "rationale": rationale,
        "payload": payload,
    }
    proposals.append(prop)
    save_proposals(proposals)
    return prop


def list_proposals(status=None, kind=None):
    """Return proposals filtered by status and/or kind."""
    proposals = load_proposals()
    if status:
        proposals = [p for p in proposals if p["status"] == status]
    if kind:
        proposals = [p for p in proposals if p["kind"] == kind]
    return proposals


def get(prop_id):
    """Return a proposal by id or None."""
    for p in load_proposals():
        if p["id"] == prop_id:
            return p
    return None


def _set_status(prop_id, status):
    proposals = load_proposals()
    for p in proposals:
        if p["id"] == prop_id:
            p["status"] = status
            save_proposals(proposals)
            return p
    return None


def approve(prop_id):
    """Mark a proposal as approved."""
    return _set_status(prop_id, "approved")


def reject(prop_id):
    """Mark a proposal as rejected."""
    return _set_status(prop_id, "rejected")
