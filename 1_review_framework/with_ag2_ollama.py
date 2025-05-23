from autogen import AssistantAgent, UserProxyAgent

config_list = [
    {
        "model": "phi4",
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
    }
]

assistant = AssistantAgent("assistant", llm_config={"config_list": config_list})
user_proxy = UserProxyAgent("user_proxy", code_execution_config=False)

# Start the chat
user_proxy.initiate_chat(
    assistant,
    message="こんにちは",
)
