import json
import re
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import chain
from prompts import reona_json2
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage
from langchain.output_parsers import PydanticOutputParser
from traceloop.sdk import Traceloop

Traceloop.init(api_endpoint="http://127.0.0.1:4318")
load_dotenv()

workflow = StateGraph(state_schema=MessagesState)


class Result(BaseModel):
    move: str
    response: str


model = ChatOllama(
    model="deepseek-r1:32b-qwen-distill-q4_K_M",
    temperature=0.7,
    max_tokens=8000,
)


@chain
def dynamic_system_prompt(input: str) -> ChatPromptTemplate:
    prompt_base = reona_json2.character_description
    prompt_extra = ""
    dynamic_prompt = f"{prompt_base} {prompt_extra}"

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                dynamic_prompt,
            ),
            (
                "human",
                "{input}",
            ),
        ]
    )
    return prompt


# structured_model = model.with_structured_output(Result, include_raw=True)
parser = PydanticOutputParser(pydantic_object=Result)


def call_model(state: MessagesState):
    chain = dynamic_system_prompt | model
    messages = state["messages"]
    result = chain.invoke(messages, config={"configurable": {"thread_id": "1"}})
    print(result)
    # タグで囲まれたすべての内容を削除
    # <think> セクションをすべて削除
    cleaned_content = re.sub(r"<think>.*?</think>", "", result.content, flags=re.DOTALL)
    res = parser.parse(cleaned_content)
    print(res)
    return {"messages": res.response}


workflow.add_node("model", call_model)
workflow.add_edge(START, "model")

# Add simple in-memory checkpointer
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

while True:
    user_input = input("You: ")
    result = app.invoke(
        {
            "messages": [HumanMessage(content=user_input)],
        },
        config={"configurable": {"thread_id": "1"}},
    )
    print(f"Ollama:{result['messages'][-1].content}")
