"""LWE chat google genai provider integration."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from google import genai
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.runnables import Runnable
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    HarmBlockThreshold,
    HarmCategory,
)
from typing_extensions import override

from lwe.core.async_compat import ensure_event_loop
from lwe.core.provider import PresetValue, Provider

MODEL_FILTERS = [
    "models/gemini-",
]
HARM_BLOCK_THRESHOLD_OPTIONS = [
    "BLOCK_NONE",
    "BLOCK_LOW_AND_ABOVE",
    "BLOCK_MEDIUM_AND_ABOVE",
    "BLOCK_ONLY_HIGH",
    "HARM_BLOCK_THRESHOLD_UNSPECIFIED",
]


class CustomChatGoogleGenerativeAI(ChatGoogleGenerativeAI):

    def __init__(self, **kwargs: Any) -> None:
        ensure_event_loop()
        super().__init__(**kwargs)

    @property
    @override
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "chat_google_genai"


class ProviderChatGoogleGenai(Provider):
    """
    Access to chat Google GenAI models
    """

    @property
    @override
    def model_property_name(self) -> str:
        return "model"

    @property
    @override
    def capabilities(self) -> dict[str, bool]:
        return {
            "chat": True,
            "validate_models": True,
        }

    @property
    @override
    def default_model(self) -> str:
        return list(self.available_models)[0]

    def fetch_models(self) -> dict[str, dict[str, int | None]]:
        try:
            client = genai.Client()
            model_data = client.models.list()
            if not model_data:
                raise ValueError("Could not retrieve models")
            models = {
                model.name: {"max_tokens": model.input_token_limit}
                for model in model_data
                if model.name is not None
                and any(substring in model.name for substring in MODEL_FILTERS)
            }
            return models
        except Exception as e:
            raise ValueError(f"Could not retrieve models: {e}")

    @override
    def default_customizations(
        self, defaults: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        defaults = defaults or {}
        # NOTE: Seems kinda dumb that the class doesn't provide a default
        # model, but it requires one, so...
        defaults.setdefault("model", self.default_model)
        return super().default_customizations(defaults)

    @override
    def prepare_messages_method(
        self,
    ) -> Callable[[list[dict[str, Any]]], list[BaseMessage]]:
        return self.prepare_messages_for_llm_chat

    @override
    def llm_factory(self) -> type[CustomChatGoogleGenerativeAI]:
        return CustomChatGoogleGenerativeAI

    def format_responses_content(self, content_list: list[str | dict[str, Any]]) -> str:
        parts: list[str] = []
        for content in content_list:
            text = content if isinstance(content, str) else content.get("text")
            if isinstance(text, str):
                parts.append(text)
        return "".join(parts)

    def handle_non_streaming_response(self, response: AIMessage) -> AIMessage:
        if isinstance(response.content, list):
            response.content = self.format_responses_content(response.content)
        return response

    def handle_streaming_chunk(
        self,
        chunk: AIMessageChunk | str,
        _previous_chunks: list[AIMessageChunk | str],
    ) -> str:
        if isinstance(chunk, str):
            return chunk
        elif isinstance(chunk.content, str):
            return chunk.content
        return self.format_responses_content(chunk.content)

    def configure_safety_settings(
        self, configured_safety_settings: dict[str, str]
    ) -> dict[HarmCategory, HarmBlockThreshold]:
        safety_settings = {}
        for category, threshold in configured_safety_settings.items():
            try:
                harm_category = getattr(HarmCategory, category)
                harm_block_threshold = getattr(HarmBlockThreshold, threshold)
            except AttributeError as e:
                raise ValueError(f"Invalid HarmCategory or HarmBlockThreshold: {e}")
            safety_settings[harm_category] = harm_block_threshold
        return safety_settings

    @override
    def make_llm(
        self,
        customizations: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        use_defaults: bool = False,
    ) -> Runnable[Any, Any]:
        customizations = customizations or {}
        final_customizations = self.get_customizations()
        final_customizations.update(customizations)
        configured_safety_settings = final_customizations.pop("safety_settings", None)
        if configured_safety_settings:
            final_customizations["safety_settings"] = self.configure_safety_settings(
                configured_safety_settings
            )
        return super().make_llm(
            final_customizations,
            tools=tools,
            tool_choice=tool_choice,
            use_defaults=use_defaults,
        )

    @override
    def customization_config(self) -> dict[str, Any]:
        return {
            "model": PresetValue(str, options=self.available_models),
            "google_api_key": PresetValue(str, private=True),
            "temperature": PresetValue(float, min_value=0.0, max_value=1.0),
            "max_output_tokens": PresetValue(int, min_value=1, include_none=True),
            "thinking_budget": PresetValue(int, min_value=0, include_none=True),
            "top_k": PresetValue(int, min_value=1, max_value=40),
            "top_p": PresetValue(float, min_value=0.0, max_value=1.0),
            "n": PresetValue(int, 1, 10),
            "safety_settings": {
                "HARM_CATEGORY_DANGEROUS_CONTENT": PresetValue(
                    str,
                    options=HARM_BLOCK_THRESHOLD_OPTIONS,
                    include_none=True,
                ),
                "HARM_CATEGORY_HATE_SPEECH": PresetValue(
                    str,
                    options=HARM_BLOCK_THRESHOLD_OPTIONS,
                    include_none=True,
                ),
                "HARM_CATEGORY_HARASSMENT": PresetValue(
                    str,
                    options=HARM_BLOCK_THRESHOLD_OPTIONS,
                    include_none=True,
                ),
                "HARM_CATEGORY_SEXUALLY_EXPLICIT": PresetValue(
                    str,
                    options=HARM_BLOCK_THRESHOLD_OPTIONS,
                    include_none=True,
                ),
            },
            "tools": None,
            "tool_choice": None,
        }
