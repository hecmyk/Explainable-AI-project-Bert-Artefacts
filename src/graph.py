"""Builds the ReAct DQ agent over the tools."""
import os

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from dq.bootstrap import load_config
from dq.agent.tools import ALL_TOOLS
from dq.agent.prompts import SYSTEM_PROMPT


def build_agent():
    cfg = load_config()
    model = cfg["llm"].get("agent_model") or cfg["llm"]["compiler_model"]
    base_url = os.environ.get("LLM_BASE_URL") or cfg["llm"].get("base_url") \
        or "http://datacamp-llm-api.datacamp-llm-api-production/v1/openai"
    api_key = os.environ.get("LLM_API_KEY") or cfg["llm"].get("api_key") or "None"

    llm = ChatOpenAI(model=model, base_url=base_url, api_key=api_key, temperature=0)
    agent = create_react_agent(llm, ALL_TOOLS, prompt=SYSTEM_PROMPT)
    return agent


def run_agent(agent, user_message: str, history: list | None = None) -> dict:
    """Run one turn. history is a list of {'role', 'content'} dicts. Returns the final state."""
    messages = list(history or [])
    messages.append({"role": "user", "content": user_message})
    result = agent.invoke({"messages": messages})
    return result
