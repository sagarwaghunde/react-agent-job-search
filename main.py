import os

from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor
from langchain.agents.react.agent import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
# from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

from prompt import REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS
from schemas import AgentResponse

load_dotenv()

tools = [TavilySearch()]
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
structured_llm = llm.with_structured_output(AgentResponse)
react_prompt = hub.pull("hwchase17/react")
# output_parser = PydanticOutputParser(pydantic_object=AgentResponse)
react_prompt_with_format_instructions = PromptTemplate(
    template=REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS,
    input_variables=["input", "agent_scratchpad", "tool_names"],
    # partial_variables={"format_instructions": output_parser.get_format_instructions()},
    partial_variables={"format_instructions": ""}, ## can use empty strings with structured llm for format instructions
)
agent = create_react_agent(
    llm=llm, 
    tools=tools, 
    prompt=react_prompt_with_format_instructions
)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
extract_output = RunnableLambda(lambda x: x.get("output"))
# parse_output = RunnableLambda(lambda x: output_parser.parse(x))
# chain = agent_executor | extract_output | parse_output
chain = agent_executor | extract_output | structured_llm
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
