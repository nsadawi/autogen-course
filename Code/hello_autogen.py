import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

async def main():
    model_client = OpenAIChatCompletionClient(model="gpt-4o")  # uses OPENAI_API_KEY env var
    agent = AssistantAgent(
        name="hello_agent",
        model_client=model_client,
        system_message="You are a helpful assistant."
    )
    await Console(agent.run_stream(task="Say hello in one sentence."))
    await model_client.close()

if __name__ == "__main__":
    asyncio.run(main())

