
from dotenv import load_dotenv
from langchain import hub
from langchain.agents.react.agent import create_react_agent
from langchain.agents import AgentExecutor
from langchain_tavily import TavilySearch
from langchain_openai import ChatOpenAI
import os

load_dotenv()

tools = [TavilySearch()]
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
react_prompt = hub.pull("hwchase17/react")
agent = create_react_agent(llm, tools, prompt=react_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
chain = agent_executor
# @tool
# def search_tavily(query: str) -> str:
#     """Search the web for information."""
#     return TavilySearch().run(query)

def main():
    response = chain.invoke(
        input={
            "input": "search for 3 job postings for AI engineer on linkedin in Singapore and list their details"
        }
    )
    print(response)

if __name__ == "__main__":
    main()