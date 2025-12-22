import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

# ---- (Optional) tools the researcher can call ----
def kb_lookup(query: str) -> str:
    """Search internal KB for relevant policy/process info."""
    return f"(mock kb) Top KB result for: {query}"

async def main():
    model = OpenAIChatCompletionClient(model="gpt-4o")

    planner = AssistantAgent(
        name="planner",
        model_client=model,
        system_message=(
            "You are the PLANNER.\n"
            "Create a numbered plan. For each step: what to do, and what evidence/tools are needed.\n"
            "Do NOT answer the user; output only the plan.\n"
            "End your message with: PLAN_READY"
        ),
    )

    researcher = AssistantAgent(
        name="researcher",
        model_client=model,
        system_message=(
            "You are the RESEARCHER.\n"
            "Follow the plan. Use tools to gather evidence. Return bullet points of findings.\n"
            "If evidence is missing, say exactly what is missing.\n"
            "End your message with: RESEARCH_READY"
        ),
        tools=[kb_lookup],  # AutoGen wraps Python functions as tools automatically
    )

    critic = AssistantAgent(
        name="critic",
        model_client=model,
        system_message=(
            "You are the CRITIC.\n"
            "Check for unsupported claims, missing steps, risky assumptions, and policy violations.\n"
            "If issues exist, request a revision and say which agent should act next.\n"
            "If satisfied, produce a short 'approved' summary and end with: DONE"
        ),
    )

    # Terminate when critic says DONE OR after a safety cap on turns
    termination = TextMentionTermination("DONE") | MaxMessageTermination(12)

    team = RoundRobinGroupChat([planner, researcher, critic], termination_condition=termination)

    await Console(team.run_stream(task="Design a customer-support workflow for password reset and escalation."))

    await model.close()

if __name__ == "__main__":
    asyncio.run(main())

