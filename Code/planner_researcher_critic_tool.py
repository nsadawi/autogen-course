import asyncio
import os
import requests

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

from tools import web_search

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

