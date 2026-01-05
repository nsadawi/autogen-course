"""
ip_prior_art_assistant_hitl.py

A local prior-art assistant with:
- Local TF-IDF retrieval tools (pa_list_docs, pa_search, pa_get_excerpt)
- Multi-agent orchestration (Planner → Researcher → Critic → Human Approver → Drafter)
- True Human-in-the-Loop (HITL): the run pauses for your input:
    APPROVE
    MORE: <instructions>
    STOP

How to run
----------
1) Create a folder: prior_art/
   Put .txt or .md files inside it.

2) Set your OpenAI key (example using .env):
   Create a file named .env in the same directory:
     OPENAI_API_KEY=your_key_here

3) Install deps (adjust package names if your environment differs):
   pip install python-dotenv scikit-learn autogen-agentchat autogen-ext

4) Run:
   python ip_prior_art_assistant_hitl.py
"""

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient


# -------------------------
# 1) Document store + retrieval
# -------------------------

@dataclass
class Doc:
    doc_id: str
    path: str
    text: str


class PriorArtIndex:
    """
    Simple TF-IDF index over local text/markdown files.
    Lightweight and local-only.
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
        """
        Chunk text by characters (simple + robust).
        """
        chunks: List[str] = []
        start = 0
        while start < len(text):
            end = min(len(text), start + max_chars)
            chunks.append(text[start:end])
            start = max(0, end - overlap)
            if end == len(text):
                break
        return chunks


def load_prior_art(folder: str = "prior_art") -> List[Doc]:
    p = Path(folder)
    if not p.exists():
        raise FileNotFoundError(
            f"Folder '{folder}' not found. Create it and add .txt/.md files."
        )

    docs: List[Doc] = []
    for fp in sorted(p.glob("**/*")):
        if fp.is_file() and fp.suffix.lower() in {".txt", ".md"}:
            text = fp.read_text(encoding="utf-8", errors="ignore")
            docs.append(Doc(doc_id=fp.stem, path=str(fp), text=text))

    if not docs:
        raise ValueError(f"No .txt/.md files found in '{folder}'.")
    return docs


# -------------------------
# 2) Tools exposed to AutoGen
# -------------------------

_INDEX: Optional[PriorArtIndex] = None


def pa_search(query: str, top_k: int = 5) -> str:
    """
    Search prior-art corpus for relevant documents.
    Returns ranked docs with scores and short previews.
    """
    assert _INDEX is not None, "Index not initialized."
    results = _INDEX.search(query, top_k=top_k)
    lines: List[str] = []
    for doc, score in results:
        preview = doc.text[:400].replace("\n", " ").strip()
        lines.append(
            f"- doc_id={doc.doc_id} score={score:.3f} path={doc.path}\n"
            f"  preview={preview}..."
        )
    return "Search results:\n" + "\n".join(lines)


def pa_get_excerpt(doc_id: str, query_hint: str = "", max_chunks: int = 3) -> str:
    """
    Return the most relevant excerpt chunks from a given doc_id,
    using query_hint for local scoring.
    """
    assert _INDEX is not None, "Index not initialized."

    doc = next((d for d in _INDEX.docs if d.doc_id == doc_id), None)
    if not doc:
        return f"Document not found: {doc_id}"

    chunks = _INDEX.chunk_text(doc.text)
    if not query_hint.strip():
        chosen = chunks[:max_chunks]
    else:
        # Rank chunks by similarity to the hint using a local TF-IDF fit over chunks
        vec = TfidfVectorizer(stop_words="english", max_features=20000)
        mat = vec.fit_transform(chunks + [query_hint])
        sims = cosine_similarity(mat[-1], mat[:-1]).flatten()
        ranked = sims.argsort()[::-1][:max_chunks]
        chosen = [chunks[i] for i in ranked]

    excerpt = "\n\n---\n\n".join(chosen)
    return f"Excerpts from {doc_id}:\n{excerpt}"


def pa_list_docs() -> str:
    """List all available prior-art document IDs."""
    assert _INDEX is not None, "Index not initialized."
    return "Available prior-art docs:\n" + "\n".join(
        [f"- {d.doc_id} ({d.path})" for d in _INDEX.docs]
    )


# -------------------------
# 3) Multi-agent orchestration + HITL gate
# -------------------------

PLANNER_PROMPT = """You are the PLANNER for a legal/IP prior-art risk scan.

Task:
- Given an invention disclosure, produce a structured plan for:
  (1) identifying key claim-like features,
  (2) searching prior art,
  (3) extracting best matching excerpts,
  (4) drafting a risk summary (novelty + obviousness style concerns).

Rules:
- Do NOT draft the final memo. Output only a plan.
- The plan must explicitly mention which tools to use at each step:
  pa_list_docs, pa_search, pa_get_excerpt.
- If the human (approver) requested MORE research/synonyms, incorporate that into the plan.
- End with the exact marker: PLAN_READY
"""

RESEARCHER_PROMPT = """You are the RESEARCHER for a legal/IP prior-art scan.

You must:
- Follow the planner's steps.
- Use tools to search and extract excerpts.
- Return:
  1) extracted claim-like features (bullets),
  2) top prior-art hits (doc_id + why relevant),
  3) excerpts (quoted blocks) to support each mapping.

Rules:
- Use tools; do not invent document contents.
- Keep citations as doc_id references (since these are local files).
- End with the exact marker: RESEARCH_READY
"""

CRITIC_PROMPT = """You are the CRITIC (quality gate) for a legal/IP prior-art scan.

Check:
- Are key features clearly identified?
- Are risks supported by excerpts (doc_id + quotes)?
- Are there missing searches / missed synonyms?
- Are conclusions appropriately hedged (no legal certainty)?
- Is there a clear next step (e.g., deeper search, claim charting)?

Always do BOTH:
1) Provide your critique/verdict (short but specific).
2) Ask the human for a decision using EXACT commands:

   APPROVE
   MORE: <what to do next>
   STOP

End your message with: HITL_PROMPT
"""

DRAFTER_PROMPT = """You are the DRAFTER producing the final internal research memo.

Hard rules:
- Look for the most recent human decision message (from the 'approver' agent).
- If the human decision is STOP, output exactly: HUMAN_STOP
- If the human decision starts with MORE:, do NOT draft the memo.
  Instead, summarize the additional research requested in 3-6 bullets and end with: LOOP_AGAIN
- Only draft the final memo if the human decision is exactly APPROVE.

If approved, output requirements:
- Title: Prior-Art Scan & Risk Summary (Internal)
- Sections:
  1) Invention summary (2-4 sentences)
  2) Key features (bullets)
  3) Most relevant prior art (ranked list with doc_id)
  4) Risk assessment:
     - Novelty risk (what appears disclosed)
     - Obviousness-style risk (combinations / common patterns)
     - Confidence + assumptions
  5) Recommended next steps (3-6 bullets)

Rules:
- Do NOT claim legal determination; use "indicates", "suggests", "may".
- Only reference prior art using doc_id and quoted excerpts already surfaced.
- End with the exact marker: FINAL_READY
"""


def console_input(prompt: str) -> str:
    """
    Called by UserProxyAgent when it's the human's turn.
    """
    return input(prompt)


async def main() -> None:
    load_dotenv()

    # Initialize local index
    global _INDEX
    docs = load_prior_art("prior_art")
    _INDEX = PriorArtIndex(docs)

    # Model client (OpenAI hosted). Uses OPENAI_API_KEY from env.
    model_client = OpenAIChatCompletionClient(model="gpt-4o")

    planner = AssistantAgent(
        name="planner",
        model_client=model_client,
        system_message=PLANNER_PROMPT,
        model_client_stream=True,
    )

    researcher = AssistantAgent(
        name="researcher",
        model_client=model_client,
        system_message=RESEARCHER_PROMPT,
        tools=[pa_list_docs, pa_search, pa_get_excerpt],  # tool use is here
        reflect_on_tool_use=True,
        model_client_stream=True,
    )

    critic = AssistantAgent(
        name="critic",
        model_client=model_client,
        system_message=CRITIC_PROMPT,
        model_client_stream=True,
    )

    # HITL agent (no system_message argument!)
    approver = UserProxyAgent(
        name="approver",
        description="Human approver. Type APPROVE, MORE: ..., or STOP when prompted.",
        input_func=console_input,
    )

    drafter = AssistantAgent(
        name="drafter",
        model_client=model_client,
        system_message=DRAFTER_PROMPT,
        model_client_stream=True,
    )

    # Termination:
    # - stop on FINAL_READY (successful completion),
    # - stop if the drafter outputs HUMAN_STOP,
    # - or max messages safety cap.
    #
    # Note: LOOP_AGAIN is NOT a termination marker; it will naturally continue
    # into another round (planner→researcher→critic→approver→drafter) using the
    # conversation history, including the human's MORE: instructions.
    termination = (
        TextMentionTermination("FINAL_READY")
        | TextMentionTermination("HUMAN_STOP")
        | MaxMessageTermination(24)
    )

    team = RoundRobinGroupChat(
        participants=[planner, researcher, critic, approver, drafter],
        termination_condition=termination,
    )

    invention_disclosure = """\
Invention disclosure:
A system that scans incoming customer support chats and automatically routes them to the right policy module.
It uses: (1) intent classification, (2) retrieval of policy passages, (3) structured response generation with citations,
and (4) an audit log that records which policy text influenced the response. The system can escalate to a human.
We want to assess prior-art overlap and novelty/obviousness-style risks using the local corpus.
"""

    await Console(team.run_stream(task=invention_disclosure))
    await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())
