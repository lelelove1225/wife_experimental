import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import chain
from prompts import reona_json
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

workflow = StateGraph(state_schema=MessagesState)


class Result(BaseModel):
    move: str = Field(description="発言内容に応じた動作")
    response: str = Field(description="キャラクターの応答")


model = ChatOllama(
    model=os.environ["MODEL"],
    temperature=0.8,
    max_tokens=1000,
)


@chain
def dynamic_system_prompt(input: str) -> ChatPromptTemplate:
    prompt_base = reona_json.character_description
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


structured_model = model.with_structured_output(Result)


def call_model(state: MessagesState):
    chain = dynamic_system_prompt | structured_model
    messages = state["messages"]
    for _ in range(5):  # 最大3回リトライ
        result: Result = chain.invoke(
            messages, config={"configurable": {"thread_id": "1"}}
        )
        if result:
            # print(messages)
            print(result)
            return {"messages": result.response}


workflow.add_node("model", call_model)
workflow.add_edge(START, "model")

# Add simple in-memory checkpointer
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

while True:
    user_input = input("You: ")
    result = app.invoke(
        {
            "messages": [("human", user_input)],
        },
        config={"configurable": {"thread_id": "1"}},
    )
    print(f"Ollama:{result}")
