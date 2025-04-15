from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from prompts import reona_json


class Result(BaseModel):
    move: str = Field(description="発言内容に応じた動作")
    response: str = Field(description="キャラクターの応答")


model = ChatOllama(
    model="qwen2.5:32b",
    temperature=0.8,
    max_tokens=1000,
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            reona_json.character_description,
        ),
        (
            "human",
            "{input}",
        ),
    ]
)

structured_model = model.with_structured_output(Result)

chain = prompt | structured_model


result = chain.invoke(input="こんにちは")

print(result)
