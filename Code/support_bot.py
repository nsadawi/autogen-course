# support_bot.py
import asyncio
import os
import uuid
from dataclasses import dataclass, asdict
from typing import Optional

from dotenv import load_dotenv

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

from kb import kb_search

load_dotenv()

@dataclass
class Ticket:
    ticket_id: str
    user_summary: str
    category: str
    priority: str = "normal"
    contact_email: Optional[str] = None

def create_ticket(user_summary: str, category: str, priority: str = "normal", contact_email: str | None = None) -> str:
    """
    Simulate ticket creation (in real life: call Zendesk/Freshdesk/ServiceNow API).
    """
    t = Ticket(
        ticket_id=f"TCK-{uuid.uuid4().hex[:8].upper()}",
        user_summary=user_summary.strip(),
        category=category.strip().lower(),
        priority=priority.strip().lower(),
        contact_email=contact_email,
    )
    # In production, you would persist this to a DB or send to a ticketing API.
    return f"Created ticket: {asdict(t)}"

SYSTEM_MESSAGE = """You are a customer support assistant for Qaswa Telecom.

Goals:
- Resolve the user's issue accurately and efficiently.
- Use tools to look up policies/FAQs and to create support tickets.

Constraints:
- Do not invent account-specific facts (refund eligibility, order status, etc.).
- Never ask for passwords, full card numbers, or one-time codes.
- If you can't resolve confidently, escalate by creating a ticket.

Tool rules:
- Use kb_search for policy/FAQ questions.
- Use create_ticket when the issue likely requires human support or the user asks for escalation.

Style:
- Friendly, concise, and structured. If giving steps, number them.
"""

async def main():
    # Model client (OpenAI). Reads OPENAI_API_KEY from env by default.
    # You can swap to other providers by changing the model client per AutoGen "Models" docs.
    model_client = OpenAIChatCompletionClient(model="gpt-4o")

    agent = AssistantAgent(
        name="support_agent",
        model_client=model_client,
        system_message=SYSTEM_MESSAGE,
        tools=[kb_search, create_ticket],
        reflect_on_tool_use=True,
        model_client_stream=True,
    )

    print("\nQaswa Support Bot (type 'exit' to quit)\n")
    while True:
        user_text = input("You: ").strip()
        if user_text.lower() in {"exit", "quit"}:
            break

        # Run one turn (task = user's latest message).
        await Console(agent.run_stream(task=user_text))

    await model_client.close()

if __name__ == "__main__":
    # Basic key check (optional)
    if not os.getenv("OPENAI_API_KEY"):
        print("Missing OPENAI_API_KEY. Set it in your environment or a .env file.")
    asyncio.run(main())

