import google.generativeai as genai

from langchain_google_genai import ChatGoogleGenerativeAI, HarmBlockThreshold, HarmCategory

from lwe.core.provider import Provider, PresetValue
from lwe.core.async_compat import ensure_event_loop


MODEL_FILTERS = [
    "models/gemini-1.5",
    "models/gemini-2.0",
    "models/gemini-exp-",
]
HARM_BLOCK_THRESHOLD_OPTIONS = [
    'BLOCK_NONE',
    'BLOCK_LOW_AND_ABOVE',
    'BLOCK_MEDIUM_AND_ABOVE',
    'BLOCK_ONLY_HIGH',
    'HARM_BLOCK_THRESHOLD_UNSPECIFIED',
]


class CustomChatGoogleGenerativeAI(ChatGoogleGenerativeAI):

    def __init__(self, **kwargs):
        ensure_event_loop()
        super().__init__(**kwargs)

    @property
    def _llm_type(self):
        """Return type of llm."""
        return "chat_google_genai"


class ProviderChatGoogleGenai(Provider):
    """
    Access to chat Google GenAI models
    """

    @property
    def model_property_name(self):
        return 'model'

    @property
    def capabilities(self):
        return {
            "chat": True,
            'validate_models': True,
        }

    @property
    def default_model(self):
        return list(self.available_models)[0]

    def fetch_models(self):
        try:
            model_data = genai.list_models()
            if not model_data:
                raise ValueError('Could not retrieve models')
            models = {
                model.name: {'max_tokens': model.input_token_limit}
                for model in model_data
                if any(substring in model.name for substring in MODEL_FILTERS)
            }
            return models
        except Exception as e:
            raise ValueError(f"Could not retrieve models: {e}")

    def default_customizations(self, defaults=None):
        defaults = defaults or {}
        # NOTE: Seems kinda dumb that the class doesn't provide a default
        # model, but it requires one, so...
        defaults.setdefault('model', self.default_model)
        return super().default_customizations(defaults)

    def prepare_messages_method(self):
        return self.prepare_messages_for_llm_chat

    def llm_factory(self):
        return CustomChatGoogleGenerativeAI

    def configure_safety_settings(self, configured_safety_settings):
        safety_settings = {}
        for category, threshold in configured_safety_settings.items():
            try:
                harm_category = getattr(HarmCategory, category)
                harm_block_threshold = getattr(HarmBlockThreshold, threshold)
            except AttributeError as e:
                raise ValueError(f"Invalid HarmCategory or HarmBlockThreshold: {e}")
            safety_settings[harm_category] = harm_block_threshold
        return safety_settings

    def make_llm(self, customizations=None, tools=None, tool_choice=None, use_defaults=False):
        customizations = customizations or {}
        final_customizations = self.get_customizations()
        final_customizations.update(customizations)
        configured_safety_settings = final_customizations.pop('safety_settings', None)
        if configured_safety_settings:
            final_customizations['safety_settings'] = self.configure_safety_settings(configured_safety_settings)
        return super().make_llm(final_customizations, tools=tools, tool_choice=tool_choice, use_defaults=use_defaults)

    def customization_config(self):
        return {
            'model': PresetValue(str, options=self.available_models),
            'google_api_key': PresetValue(str, private=True),
            'temperature': PresetValue(float, min_value=0.0, max_value=1.0),
            'max_output_tokens': PresetValue(int, min_value=1, max_value=2048, include_none=True),
            'top_k': PresetValue(int, min_value=1, max_value=40),
            'top_p': PresetValue(float, min_value=0.0, max_value=1.0),
            'n': PresetValue(int, 1, 10),
            'safety_settings': {
                'HARM_CATEGORY_DANGEROUS_CONTENT': PresetValue(str, options=HARM_BLOCK_THRESHOLD_OPTIONS, include_none=True),
                'HARM_CATEGORY_HATE_SPEECH': PresetValue(str, options=HARM_BLOCK_THRESHOLD_OPTIONS, include_none=True),
                'HARM_CATEGORY_HARASSMENT': PresetValue(str, options=HARM_BLOCK_THRESHOLD_OPTIONS, include_none=True),
                'HARM_CATEGORY_SEXUALLY_EXPLICIT': PresetValue(str, options=HARM_BLOCK_THRESHOLD_OPTIONS, include_none=True),
            },
            "tools": None,
            "tool_choice": None,
        }
