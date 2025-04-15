import logfire
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from prompts.talk import reona_talk
from prompts.move import reona_move

logfire.configure()


class ResultSet(BaseModel):
    move: str = Field(description="発言内容に応じた動作")
    response: str = Field(description="キャラクターの応答")


class TalkResult(BaseModel):
    response: str = Field(description="キャラクターの応答")


class MoveResult(BaseModel):
    move: str = Field(description="発言内容に応じた動作")


agent_talk = Agent(
    "ollama:phi4",
    system_prompt=reona_talk.response_prompt,
    retries=5,
)

agent_move = Agent(
    "ollama:phi4",
    system_prompt=reona_move.action_prompt,
    retries=5,
)


def main():
    messages = []
    print("対話を開始します。終了するには 'exit' と入力してください。")

    while True:
        user_input = input("\nあなた: ").strip()

        if user_input.lower() == "exit":
            print("\n対話を終了します。")
            break

        if not user_input:
            continue

        try:
            result_talk = agent_talk.run_sync(user_input, message_history=messages)
            result_move = agent_move.run_sync(result_talk.data)
            # result_talkとresult_moveの結果を結合して表示

            result = ResultSet(move=result_move.data, response=result_talk.data)
            messages = result_talk.all_messages()
            print(result)
        except Exception as e:
            print(f"\nエラーが発生しました: {e}")


if __name__ == "__main__":
    main()
