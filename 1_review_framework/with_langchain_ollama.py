import json
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import chain
from prompts import reona_json2
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage
from traceloop.sdk import Traceloop

Traceloop.init(api_endpoint="http://127.0.0.1:4318")
load_dotenv()

workflow = StateGraph(state_schema=MessagesState)


class Result(BaseModel):
    move: str = Field(description="発言内容に応じた動作")
    response: str = Field(description="キャラクターの応答")


model = ChatOllama(
    model="qwen2.5:32b",
    temperature=0.3,
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


structured_model = model.with_structured_output(Result, include_raw=True)


def call_model(state: MessagesState):
    chain = dynamic_system_prompt | structured_model
    messages = state["messages"]
    for _ in range(10):  # 最大10回リトライ
        result = chain.invoke(messages, config={"configurable": {"thread_id": "1"}})

        try:
            parsed: Result = result["parsed"]
            print(parsed)
            messages.append(AIMessage(content=parsed.response))
            return {"messages": messages}
        except Exception as e:
            print(e)
            print(result)
            raw = result["raw"]
            print(raw)

            # 直接JSONを解析してみる
            try:
                print("try to parse JSON")
                raw_content = json.loads(raw.content)
                parsed = Result(**raw_content)  # pydanticでResultに直接マッピング
                messages.append(AIMessage(content=parsed.response))
                return {"messages": messages}
            except json.JSONDecodeError as je:
                print("JSONパースエラー:", je)
                continue


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
