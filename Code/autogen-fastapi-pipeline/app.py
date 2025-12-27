"""
FastAPI + AutoGen AgentChat pipeline:
Planner -> Executor -> Reviewer -> Approver (human-in-the-loop)

- Planner: prefer Gemini (OpenAI-compatible) if configured, else fallback to OpenAI
- Executor: OpenAI GPT (tool-using)
- Reviewer: prefer Claude if configured, else fallback to OpenAI

Endpoints:
- POST /runs/start
- POST /runs/{run_id}/approve
- GET /health

How it works:
1) /runs/start:
   - Runs planner->executor->reviewer
   - Returns an approval packet (plan, evidence/tool outputs, reviewer verdict, draft)
2) /runs/{run_id}/approve:
   - If approved, runs executor finalization (applying reviewer notes)
   - Returns final response
"""

import asyncio
import os
import time
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination

from autogen_ext.models.openai import OpenAIChatCompletionClient

# Anthropic client is optional; we import lazily inside a helper
# from autogen_ext.models.anthropic import AnthropicChatCompletionClient


# ---------------------------
# Data models for API I/O
# ---------------------------

class StartRunRequest(BaseModel):
    invention_disclosure: str = Field(..., description="Text describing the invention / idea / claim-like features.")
    constraints: Optional[str] = Field(
        default="",
        description="Optional constraints e.g. jurisdiction, must-include risks, or internal policy notes."
    )
    top_k_docs: int = Field(default=5, ge=1, le=20, description="How many documents to retrieve (local corpus).")


class StartRunResponse(BaseModel):
    run_id: str
    status: str
    approval_request: Dict[str, Any]


class ApproveRequest(BaseModel):
    decision: str = Field(..., description="APPROVE or REJECT")
    approver_notes: Optional[str] = Field(default="", description="Optional notes to guide finalization.")


class ApproveResponse(BaseModel):
    run_id: str
    status: str
    final_output: Optional[str] = None
    message: Optional[str] = None


# ---------------------------
# In-memory run store
# Replace with Redis/DB in production
# ---------------------------

_RUNS: Dict[str, Dict[str, Any]] = {}


# ---------------------------
# Local "prior-art" index + tools
# (Works entirely locally, no external web calls required.)
# ---------------------------

@dataclass
class Doc:
    doc_id: str
    path: str
    text: str


class LocalCorpusIndex:
    """
    Lightweight TF-IDF index for local documents (TXT/MD).
    Used as a stand-in for "prior-art document scanning" or internal KB search.
    """

    def __init__(self, docs: List[Doc]) -> None:
        self.docs = docs
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=50000)
        self.doc_matrix = self.vectorizer.fit_transform([d.text for d in docs])

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Doc, float]]:
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.doc_matrix).flatten()
        ranked = sims.argsort()[::-1][:top_k]
        return [(self.docs[i], float(sims[i])) for i in ranked]

    @staticmethod
    def chunk_text(text: str, max_chars: int = 1800, overlap: int = 200) -> List[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = min(len(text), start + max_chars)
            chunks.append(text[start:end])
            start = max(0, end - overlap)
            if end == len(text):
                break
        return chunks


_INDEX: Optional[LocalCorpusIndex] = None


def _tool_ok(data: Any) -> Dict[str, Any]:
    return {"ok": True, "data": data}


def _tool_err(error_type: str, message: str, retryable: bool = False) -> Dict[str, Any]:
    return {"ok": False, "error_type": error_type, "message": message, "retryable": retryable}


def corpus_list_docs() -> Dict[str, Any]:
    """List local corpus documents available for scanning."""
    if _INDEX is None:
        return _tool_err("not_ready", "Local corpus index is not initialized.")
    return _tool_ok([{"doc_id": d.doc_id, "path": d.path} for d in _INDEX.docs])


def corpus_search(query: str, top_k: int = 5) -> Dict[str, Any]:
    """
    Search the local corpus. Returns doc_id, similarity score, and preview.
    """
    if _INDEX is None:
        return _tool_err("not_ready", "Local corpus index is not initialized.")
    try:
        results = _INDEX.search(query=query, top_k=top_k)
        payload = []
        for doc, score in results:
            preview = doc.text[:350].replace("\n", " ").strip()
            payload.append({"doc_id": doc.doc_id, "score": round(score, 4), "path": doc.path, "preview": preview})
        return _tool_ok(payload)
    except Exception as e:
        return _tool_err("search_failed", str(e), retryable=False)


def corpus_get_excerpts(doc_id: str, query_hint: str, max_chunks: int = 3) -> Dict[str, Any]:
    """
    Return the most relevant chunks from a local doc based on a query hint.
    """
    if _INDEX is None:
        return _tool_err("not_ready", "Local corpus index is not initialized.")
    try:
        doc = next((d for d in _INDEX.docs if d.doc_id == doc_id), None)
        if not doc:
            return _tool_err("not_found", f"doc_id '{doc_id}' not found.")
        chunks = _INDEX.chunk_text(doc.text)

        # Rank chunks by TF-IDF similarity to query_hint (local-only).
        vec = TfidfVectorizer(stop_words="english", max_features=20000)
        mat = vec.fit_transform(chunks + [query_hint])
        sims = cosine_similarity(mat[-1], mat[:-1]).flatten()
        ranked = sims.argsort()[::-1][:max_chunks]
        selected = [chunks[i] for i in ranked]

        return _tool_ok({"doc_id": doc_id, "chunks": selected})
    except Exception as e:
        return _tool_err("excerpt_failed", str(e), retryable=False)


# ---------------------------
# Model client helpers (multi-model wiring)
# ---------------------------

def _openai_client(model: str) -> OpenAIChatCompletionClient:
    """
    Standard OpenAI client for GPT execution/fallback.
    """
    return OpenAIChatCompletionClient(model=model)


def _gemini_openai_compat_client() -> Optional[OpenAIChatCompletionClient]:
    """
    Gemini planning via an OpenAI-compatible endpoint (optional).

    Requirements:
    - GEMINI_BASE_URL: base URL ending with /v1 (or equivalent)
    - GEMINI_API_KEY
    - GEMINI_MODEL

    If not configured, return None and the app falls back to OpenAI for planning.
    """
    base_url = os.getenv("GEMINI_BASE_URL", "").strip()
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    model = os.getenv("GEMINI_MODEL", "").strip()
    if not (base_url and api_key and model):
        return None

    # OpenAIChatCompletionClient supports custom base_url & api_key
    # so it can target OpenAI-compatible gateways.
    return OpenAIChatCompletionClient(model=model, base_url=base_url, api_key=api_key)


def _claude_client_if_available() -> Optional[Any]:
    """
    Return an AnthropicChatCompletionClient if ANTHROPIC_API_KEY is present,
    otherwise None (fallback to OpenAI reviewer).

    We import Anthropic client lazily so the app can run even if the extra isn't installed.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    model = os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-latest").strip()
    if not api_key:
        return None

    try:
        from autogen_ext.models.anthropic import AnthropicChatCompletionClient
        return AnthropicChatCompletionClient(model=model, api_key=api_key)
    except Exception:
        # If extension is not installed or import fails, fallback.
        return None


# ---------------------------
# Prompts: stage contracts + markers
# ---------------------------

PLANNER_PROMPT = """You are the PLANNER in a pipeline: planner -> executor -> reviewer -> approver.

Domain: Legal/IP prior-art scan assistant.
Goal: produce an actionable plan to scan a local corpus of prior-art-like documents and summarize risk.

You MUST output a plan in JSON with keys:
- "claim_like_features": [..]   (list of features you will look for)
- "search_queries": [..]        (queries to use against the local corpus tool)
- "mapping_strategy": ".."      (how you will map features -> excerpts)
- "risk_dimensions": [..]       (e.g., novelty overlap, obviousness/combinations, scope ambiguity)
- "success_criteria": [..]      (how we know the run is good)
- "handoff_to_executor": ".."   (very explicit next-step instructions)

Rules:
- Do NOT call tools.
- Do NOT draft the final memo.
- End your message with the exact marker: PLAN_READY
"""

EXECUTOR_PROMPT = """You are the EXECUTOR in a pipeline: planner -> executor -> reviewer -> approver.

You MUST:
1) Read the planner's JSON plan.
2) Use tools to search the local corpus:
   - corpus_list_docs
   - corpus_search
   - corpus_get_excerpts
3) Build an evidence pack:
   - top hits (doc_id + score + why)
   - excerpts mapped to claim-like features
4) Draft an INTERNAL risk memo (not legal advice), with hedging language.

Output in JSON with keys:
- "evidence": {
    "search_results": [...],
    "feature_to_excerpts": [{"feature": "..", "doc_id": "..", "excerpts": ["..."]}, ...]
  }
- "draft_memo": "..."
- "open_questions": [...]
- "limits": "..."

Rules:
- Never invent document contents; only use excerpts returned by tools.
- Keep outputs compact but complete.
- End your message with the exact marker: EXEC_DONE
"""

REVIEWER_PROMPT = """You are the REVIEWER (quality gate) in a pipeline: planner -> executor -> reviewer -> approver.

You MUST review the executor's JSON:
- Are the key features reasonable?
- Is each risk claim supported by excerpts?
- Any missing synonyms / queries that should be run?
- Any overconfident legal conclusions? (must be hedged)
- Any obvious gaps?

Output JSON with keys:
- "verdict": "PASS" or "FAIL"
- "issues": [{"severity":"high|med|low","problem":"...","fix":"..."}...]
- "suggested_extra_queries": [...]
- "approval_summary": "short summary for the human approver"

Rules:
- Be strict; prefer FAIL if evidence is weak.
- End your message with the exact marker: REVIEW_DONE
"""

FINALIZE_PROMPT = """You are the EXECUTOR FINALIZER after human approval.

Input:
- Planner plan
- Executor evidence/draft
- Reviewer issues (if any)
- Human approver notes

Task:
- Produce the FINAL internal risk memo.
- If reviewer verdict was FAIL, apply the fixes where possible WITHOUT inventing evidence.
- If evidence is missing, explicitly call that out and recommend next steps.

Output plain text (not JSON), formatted with headings:
1) Invention Summary
2) Key Claim-Like Features
3) Most Relevant Prior Art (doc_id ranked)
4) Risk Assessment (Novelty-style overlap; obviousness/combinations; uncertainty)
5) Recommended Next Steps

End with the exact marker: FINAL_READY
"""


# ---------------------------
# Core pipeline runner
# ---------------------------

async def run_stage_team(task: str, agents: List[AssistantAgent], stop_marker: str, max_turns: int = 10) -> str:
    """
    Run a RoundRobin team until stop_marker is mentioned or max_turns is hit.
    Returns the final combined transcript text (AutoGen internally maintains messages).

    In production you might want a structured capture of each agent's final message.
    For simplicity, we use the team's final output text.
    """
    termination = TextMentionTermination(stop_marker) | MaxMessageTermination(max_turns)
    team = RoundRobinGroupChat(participants=agents, termination_condition=termination)

    # AgentChat team.run returns a result object; but to keep compatibility across versions,
    # we treat it as returning a final string-ish output via str().
    result = await team.run(task=task)
    return str(result)


def _init_local_corpus_index() -> None:
    """
    Initializes the local corpus index from ./prior_art (txt/md).
    If folder doesn't exist or empty, index remains None (tools will return a clear error).
    """
    global _INDEX
    folder = Path("prior_art")
    if not folder.exists():
        _INDEX = None
        return

    docs: List[Doc] = []
    for fp in sorted(folder.glob("**/*")):
        if fp.is_file() and fp.suffix.lower() in {".txt", ".md"}:
            text = fp.read_text(encoding="utf-8", errors="ignore")
            docs.append(Doc(doc_id=fp.stem, path=str(fp), text=text))

    if docs:
        _INDEX = LocalCorpusIndex(docs)
    else:
        _INDEX = None


def _build_agents() -> Dict[str, Any]:
    """
    Build model clients and agents according to env vars and fallback rules.
    Returns dict with clients + agents.
    """
    exec_model = os.getenv("OPENAI_EXEC_MODEL", "gpt-4o").strip()

    # 1) Planner: Gemini (OpenAI-compatible) if configured else OpenAI fallback
    planner_client = _gemini_openai_compat_client() or _openai_client(exec_model)

    # 2) Executor: OpenAI (tool-using)
    executor_client = _openai_client(exec_model)

    # 3) Reviewer: Claude if available else OpenAI fallback
    reviewer_client = _claude_client_if_available() or _openai_client(exec_model)

    planner = AssistantAgent(
        name="planner",
        model_client=planner_client,
        system_message=PLANNER_PROMPT,
    )

    executor = AssistantAgent(
        name="executor",
        model_client=executor_client,
        system_message=EXECUTOR_PROMPT,
        tools=[corpus_list_docs, corpus_search, corpus_get_excerpts],
        reflect_on_tool_use=True,
    )

    reviewer = AssistantAgent(
        name="reviewer",
        model_client=reviewer_client,
        system_message=REVIEWER_PROMPT,
    )

    finalizer = AssistantAgent(
        name="finalizer",
        model_client=executor_client,  # finalize with GPT execution model
        system_message=FINALIZE_PROMPT,
        tools=[corpus_search, corpus_get_excerpts],  # allow extra lookups if needed
        reflect_on_tool_use=True,
    )

    return {
        "clients": [planner_client, executor_client, reviewer_client],
        "agents": {"planner": planner, "executor": executor, "reviewer": reviewer, "finalizer": finalizer},
    }


# ---------------------------
# FastAPI app
# ---------------------------

load_dotenv()
_init_local_corpus_index()

app = FastAPI(title="AutoGen Pipeline (Planner→Executor→Reviewer→Approver)")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "local_corpus_ready": _INDEX is not None,
        "time": int(time.time()),
    }


@app.post("/runs/start", response_model=StartRunResponse)
async def start_run(req: StartRunRequest):
    """
    Runs planner -> executor -> reviewer.
    Returns an approval packet for the human approver.
    """
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("GEMINI_API_KEY"):
        raise HTTPException(status_code=400, detail="No model API keys configured (need OPENAI_API_KEY at minimum).")

    bundle = _build_agents()
    agents = bundle["agents"]

    # Compose the task to include optional constraints.
    task = f"""Invention disclosure:
{req.invention_disclosure}

Constraints / Notes:
{req.constraints}

Use the local corpus tools for prior-art scanning (if available).
Return an approval request after review.
"""

    # Stage 1: planner -> executor -> reviewer
    # We run them in a single deterministic RoundRobin team and stop when reviewer is done.
    # This keeps the "pipeline" in one coherent context.
    team_agents = [agents["planner"], agents["executor"], agents["reviewer"]]
    transcript = await run_stage_team(task=task, agents=team_agents, stop_marker="REVIEW_DONE", max_turns=12)

    run_id = uuid.uuid4().hex[:12]
    approval_packet = {
        "transcript": transcript,
        "instructions": "Review the packet. If you approve, call /runs/{run_id}/approve with decision=APPROVE.",
        "note": "If the reviewer verdict was FAIL, approval should typically be REJECT or request revisions.",
    }

    _RUNS[run_id] = {
        "status": "PENDING_APPROVAL",
        "request": req.model_dump(),
        "pipeline_transcript": transcript,
        "created_at": time.time(),
    }

    return StartRunResponse(run_id=run_id, status="PENDING_APPROVAL", approval_request=approval_packet)


@app.post("/runs/{run_id}/approve", response_model=ApproveResponse)
async def approve_run(run_id: str, req: ApproveRequest):
    """
    Human-in-the-loop approval step.
    If approved: run finalizer stage and return final memo.
    If rejected: store reason and end.
    """
    run = _RUNS.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="run_id not found")

    if run["status"] != "PENDING_APPROVAL":
        raise HTTPException(status_code=409, detail=f"Run is not pending approval (status={run['status']})")

    decision = req.decision.strip().upper()
    if decision not in {"APPROVE", "REJECT"}:
        raise HTTPException(status_code=400, detail="decision must be APPROVE or REJECT")

    if decision == "REJECT":
        run["status"] = "REJECTED"
        run["approver_notes"] = req.approver_notes
        return ApproveResponse(run_id=run_id, status="REJECTED", message="Run rejected by approver.")

    # APPROVE: finalize
    bundle = _build_agents()
    agents = bundle["agents"]

    # Finalization task includes the prior transcript and the human notes.
    finalize_task = f"""You are finalizing after human approval.

Here is the pipeline transcript (planner/executor/reviewer):
{run['pipeline_transcript']}

Human approver notes:
{req.approver_notes}

Now produce the final internal memo per instructions.
"""

    transcript = await run_stage_team(
        task=finalize_task,
        agents=[agents["finalizer"]],
        stop_marker="FINAL_READY",
        max_turns=4,
    )

    run["status"] = "APPROVED"
    run["final_output"] = transcript
    run["approved_at"] = time.time()

    return ApproveResponse(run_id=run_id, status="APPROVED", final_output=transcript)

