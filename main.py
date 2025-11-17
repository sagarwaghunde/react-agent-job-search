import os

## from series : https://www.udemy.com/course/langchain/learn/lecture/52107227#overview The Original Langchain React Agent
from dotenv import load_dotenv
## to keep the imports stable, import from hub
# from langchain import hub
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
## can use @tool decorator to create tools from your own functions or classes. These tools will be available to the agent.
# from langchain_core.output_parsers.pydantic import PydanticOutputParser
# from langchain_core.prompts import PromptTemplate
# from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage

from prompt import REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS
from schemas import AgentResponse

load_dotenv()

tools = [TavilySearch()]
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
# structured_llm = llm.with_structured_output(AgentResponse)

# prompt template from hub
# react_prompt = hub.pull("hwchase17/react")
# output_parser = PydanticOutputParser(pydantic_object=AgentResponse)
# react_prompt_with_format_instructions = PromptTemplate(
#     template=REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS,
#     input_variables=["input", "agent_scratchpad", "tool_names"],
#     # partial_variables={"format_instructions": output_parser.get_format_instructions()},
#     partial_variables={"format_instructions": ""}, ## can use empty strings with structured llm for format instructions
# )
agent = create_agent(
    llm,
    tools=tools,
    # system_prompt=REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS,
    response_format=AgentResponse
)
# agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
# extract_output = RunnableLambda(lambda x: x.get("output"))
# parse_output = RunnableLambda(lambda x: output_parser.parse(x))
# chain = agent_executor | extract_output | parse_output
chain = agent.with_config({"run_name": "Agent"})
# @tool
# def search_tavily(query: str) -> str:
#     """Search the web for information."""
#     return TavilySearch().run(query)


def main():
    response = chain.invoke(
        input={
            "messages": [HumanMessage(content="search for 3 job postings for AI engineer on linkedin in Singapore and list their details")]
        }
    )
    print(response)


if __name__ == "__main__":
    main()
