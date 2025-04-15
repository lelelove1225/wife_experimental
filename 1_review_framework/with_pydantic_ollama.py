import logfire
from pydantic_ai import Agent
from prompts import hikari_json
from pydantic import BaseModel, Field
import os

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter


# Tracerの設定
traces_endpoint = "http://localhost:4318/v1/traces"
os.environ["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"] = traces_endpoint
logfire.configure(send_to_logfire=False, service_name="hinata_hikari")
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)
span_processor = SimpleSpanProcessor(ConsoleSpanExporter())
trace.get_tracer_provider().add_span_processor(span_processor)

with tracer.start_as_current_span("main"):
    logfire.info("Started")

    class Result(BaseModel):
        move: str = Field(description="発言内容に応じた動作")
        response: str = Field(description="キャラクターの応答")

    agent = Agent(
        "ollama:qwen2.5:32b",
        result_type=Result,
        system_prompt=hikari_json.character_description,
        retries=5,
    )

    result = agent.run_sync("こんにちは")
    # logfire.info(result.new_messages_json)

    res = result.data
    logfire.info(res.move)
    logfire.info(res.response)
    logfire.info("Finished")
