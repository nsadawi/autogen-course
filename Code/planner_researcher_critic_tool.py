import asyncio
import os
import requests

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient


# ---- REAL WEB SEARCH TOOL (SerpAPI) ----
def web_search(query: str, top_k: int = 5) -> str:
    """
    Perform a real web search using SerpAPI (Google Search).
    Returns a compact, structured summary suitable for LLM reasoning.
    """
    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        return "ERROR: SERPAPI_API_KEY not set."

    params = {
        "engine": "google",
        "q": query,
        "num": top_k,
        "api_key": api_key,
    }

    try:
        response = requests.get(
            "https://serpapi.com/search",
            params=params,
            timeout=10
        )
        response.raise_for_status()
        data = response.json()

        organic = data.get("organic_results", [])[:top_k]
        if not organic:
            return "No relevant search results found."

        results = []
        for i, item in enumerate(organic, start=1):
            title = item.get("title", "No title")
            snippet = item.get("snippet", "No snippet")
            link = item.get("link", "No link")
            results.append(
                f"{i}. {title}\n"
                f"   {snippet}\n"
                f"   Source: {link}"
            )

        return "\n".join(results)

    except Exception as e:
        return f"Web search failed: {str(e)}"


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
            "Follow the plan and gather evidence.\n"
            "You MUST use web_search() at least once to find best practices, "
            "security requirements, or escalation models.\n"
            "Summarise findings as bullet points.\n"
            "If evidence is missing, say exactly what is missing.\n"
            "End your message with: RESEARCH_READY"
        ),
        tools=[web_search],
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

    termination = (
        TextMentionTermination("DONE")
        | MaxMessageTermination(6)
    )

    team = RoundRobinGroupChat(
        [planner, researcher, critic],
        termination_condition=termination
    )

    await Console(
        team.run_stream(
            task="Design a customer-support workflow for password reset and escalation."
        )
    )

    await model.close()


if __name__ == "__main__":
    asyncio.run(main())

