# This is an example to show Pattern B HITL: “Escalation + clarification”
# See slides for more details

import asyncio
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

async def main():
    model = OpenAIChatCompletionClient(model="gpt-4o")

    writer = AssistantAgent(
        name="writer",
        model_client=model,
        system_message="Draft the final customer response. End with: DRAFT_READY"
    )

    approver = UserProxyAgent(
        name="approver",
        # by default, it requests user input; you can supply a custom input function for web/telegram, etc.
    )

    termination = TextMentionTermination("APPROVED")  # you can implement approval by having user type APPROVED
    team = RoundRobinGroupChat([writer, approver], termination_condition=termination)

    await Console(team.run_stream(task="Write a refund response for a customer (policy-compliant). Ask for approval."))

    await model.close()

if __name__ == "__main__":
    asyncio.run(main())

