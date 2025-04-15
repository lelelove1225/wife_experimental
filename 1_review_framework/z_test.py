from datetime import date

import logfire
from pydantic import ValidationError
from typing_extensions import TypedDict
from pydantic import BaseModel

from pydantic_ai import Agent
from openai import AsyncAzureOpenAI
from pydantic_ai.models.openai import OpenAIModel

import os

traces_endpoint = "http://localhost:4318/v1/traces"
os.environ["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"] = traces_endpoint

logfire.configure(
    service_name="z_test",
    send_to_logfire=False,
)


# class UserProfile(TypedDict, total=False):
#     name: str
#     dob: date
#     bio: str


class Response(BaseModel):
    response: str


client = AsyncAzureOpenAI(
    azure_endpoint="https://mitsuhiko-ota.openai.azure.com/",
    api_version="2024-09-01-preview",
    api_key="c53dc498b2cc420a950f73b7adfcbe46",
)

model = OpenAIModel("gpt-4o", openai_client=client)

agent = Agent(
    model,
    result_type=Response,
    result_retries=10,
)

result = agent.run_sync("こんにちは")

print(result.new_messages_json())


# async def main():
#     user_input = "My name is Ben, I was born on January 28th 1990, I like the chain the dog and the pyramid."
#     async with agent.run_stream(user_input) as result:
#         async for message, last in result.stream_structured(debounce_by=0.01):
#             # print(message)
#             # print(last)
#             try:
#                 profile = await result.validate_structured_result(
#                     message,
#                     allow_partial=not last,
#                 )
#                 print(profile)
#             except ValidationError:
#                 print("Validation error")
#                 continue
#             print(profile)


# if __name__ == "__main__":
#     import asyncio

#     asyncio.run(main())
