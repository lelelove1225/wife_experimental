from pydantic_ai import Agent
from prompts import reona_json

from pydantic import BaseModel, Field


class Result(BaseModel):
    move: str = Field(description="発言内容に応じた動作")
    response: str = Field(description="キャラクターの応答")


agent = Agent(
    "ollama:qwen2.5:32b",
    result_type=Result,
    system_prompt=reona_json.character_description,
)

result = agent.run_sync("こんにちは")
print(result.new_messages_json)
