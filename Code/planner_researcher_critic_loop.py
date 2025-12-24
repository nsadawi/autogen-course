import asyncio
from typing import Any, Optional

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient


def _extract_last_text(result: Any) -> str:
    """
    Best-effort extractor for the last assistant message content from an AutoGen run result.

    AutoGen result objects can vary by version; this tries common shapes:
    - result.messages[-1].content
    - result.chat_history[-1].content
    - result[-1].content (if list-like)
    Falls back to str(result).
    """
    for attr in ("messages", "chat_history"):
        seq = getattr(result, attr, None)
        if seq:
            last = seq[-1]
            content = getattr(last, "content", None)
            if isinstance(content, str) and content.strip():
                return content

    # If result itself is list-like of messages
    try:
        last = result[-1]
        content = getattr(last, "content", None)
        if isinstance(content, str) and content.strip():
            return content
    except Exception:
        pass

    return str(result)


async def main():
    model = OpenAIChatCompletionClient(model="gpt-4o")

    # --- Agents ---
    planner = AssistantAgent(
        name="planner",
        model_client=model,
        system_message=(
            "You are the PLANNER.\n"
            "Create a numbered plan for the task.\n"
            "For each step include:\n"
            "  - What to do\n"
            "  - What evidence/tools are needed\n"
            "Do NOT answer the user; output ONLY the plan.\n"
            "End your message with: PLAN_READY"
        ),
    )

    researcher = AssistantAgent(
        name="researcher",
        model_client=model,
        system_message=(
            "You are the RESEARCHER.\n"
            "Follow the approved plan.\n"
            "Use tools to gather evidence.\n"
            "Return bullet points of findings.\n"
            "If evidence is missing, say exactly what is missing.\n"
            "End your message with: RESEARCH_READY"
        )
    )

    critic = AssistantAgent(
        name="critic",
        model_client=model,
        system_message=(
            "You are the CRITIC.\n"
            "Check for unsupported claims, missing steps, risky assumptions, and policy violations.\n"
            "If issues exist, request a revision and state what must change.\n"
            "If satisfied, produce a short 'approved' summary and end with: DONE"
        ),
    )

    user_task = "Design a customer-support workflow for password reset and escalation."

    # --- HITL loop: human approves the plan before research/review ---
    approved_plan: Optional[str] = None
    human_notes: str = ""

    while approved_plan is None:
        # 1) Run PLANNER only, stop when PLAN_READY appears (or a small cap)
        plan_termination = TextMentionTermination("PLAN_READY") | MaxMessageTermination(2)
        planning_team = RoundRobinGroupChat([planner], termination_condition=plan_termination)

        planning_prompt = user_task
        if human_notes.strip():
            planning_prompt += (
                "\n\nHuman feedback / constraints to incorporate:\n"
                f"{human_notes.strip()}"
            )

        print("\n" + "=" * 80)
        print("PLANNING PHASE (auto) — will pause for human approval")
        print("=" * 80)

        plan_result = await Console(planning_team.run_stream(task=planning_prompt))

        plan_text = _extract_last_text(plan_result)

        # 2) Human approval gate (real HITL)
        print("\n" + "-" * 80)
        print("HUMAN-IN-THE-LOOP CHECKPOINT")
        print("-" * 80)
        print("Planner produced the plan above.")
        print("Type one of the following:")
        print("  - APPROVE  (continue to research/review)")
        print("  - EDIT: <your changes>  (re-plan with your feedback)")
        print("  - QUIT  (exit)\n")

        user_input = input("Your decision: ").strip()

        if user_input.upper() == "APPROVE":
            approved_plan = plan_text
            break

        if user_input.upper().startswith("EDIT:"):
            human_notes = user_input[len("EDIT:"):].strip()
            if not human_notes:
                human_notes = "Please revise the plan (human requested changes but did not specify details)."
            continue

        if user_input.upper() == "QUIT":
            print("Exiting without proceeding to execution.")
            await model.close()
            return

        # default: treat anything else as feedback and re-plan
        human_notes = user_input or "Please revise the plan (human did not approve)."

    # 3) With approved plan, run researcher + critic (and optionally planner for iterations)
    print("\n" + "=" * 80)
    print("EXECUTION PHASE (auto) — using the human-approved plan")
    print("=" * 80)

    exec_termination = TextMentionTermination("DONE") | MaxMessageTermination(12)
    exec_team = RoundRobinGroupChat([researcher, critic], termination_condition=exec_termination)

    exec_task = (
        f"{user_task}\n\n"
        "Human-approved plan (must follow this):\n"
        f"{approved_plan}\n\n"
        "Proceed with research and critique."
    )

    await Console(exec_team.run_stream(task=exec_task))

    await model.close()


if __name__ == "__main__":
    asyncio.run(main())

