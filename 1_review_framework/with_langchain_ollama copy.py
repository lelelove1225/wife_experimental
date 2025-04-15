import os
import asyncio
from dotenv import load_dotenv
from typing_extensions import Annotated, TypedDict
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import chain
from prompts import reona_json
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

workflow = StateGraph(state_schema=MessagesState)


# TypedDictでストリーミングに対応
class Result(TypedDict):
    move: Annotated[str, ..., "発言内容に応じた動作"]
    response: Annotated[str, ..., "キャラクターの応答"]


model = ChatOllama(
    model=os.environ["MODEL"],
    temperature=0.3,
    max_tokens=1000,
)


@chain
def dynamic_system_prompt(input: str) -> ChatPromptTemplate:
    prompt_base = reona_json.character_description
    dynamic_prompt = f"{prompt_base} 次のプロンプトに従って応答してください。"

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", dynamic_prompt),
            ("human", "{messages}"),
        ]
    )
    return prompt


structured_model = model.with_structured_output(Result, include_raw=True)


async def call_model(state: MessagesState):
    chain = dynamic_system_prompt | structured_model
    messages = state["messages"]

    async for chunk in chain.astream(
        {"messages": messages}, config={"configurable": {"thread_id": "1"}}
    ):
        if isinstance(chunk, AIMessage):
            print(f"Ollama (streaming): {chunk.content}")

    messages.append(AIMessage(content="応答が終了しました。"))
    return {"messages": messages}


workflow.add_node("model", call_model)
workflow.add_edge(START, "model")

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)


async def main():
    while True:
        user_input = input("You: ")
        config = {"configurable": {"thread_id": "1"}}

        async for event in app.astream_events(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
            version="v1",
        ):
            kind = event["event"]
            if kind == "on_chat_model_stream":
                content = event["data"]["chunk"].content
                if content:
                    print(f"Ollama (streaming): {content}")
            elif kind == "on_chain_end":
                print(f"Final Response: {event['data']}")


if __name__ == "__main__":
    asyncio.run(main())
