from langchain_google_genai import ChatGoogleGenerativeAI

from lwe.core.provider import Provider, PresetValue


DEFAULT_GOOGLE_GENAI_MODEL = 'gemini-pro'


class CustomChatGoogleGenerativeAI(ChatGoogleGenerativeAI):

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
            'models': {
                'chat-bison': {
                    'max_tokens': 4096,
                },
                'codechat-bison': {
                    'max_tokens': 6144,
                },
                'gemini-pro': {
                    'max_tokens': 32768,
                },
            },
        }

    @property
    def default_model(self):
        return DEFAULT_GOOGLE_GENAI_MODEL

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

    # TODO: Remove this if Gemni Pro starts supporting system messages.
    def make_llm(self, customizations=None, use_defaults=False):
        customizations = customizations or {}
        customizations.update({'convert_system_message_to_human': True})
        return super().make_llm(customizations, use_defaults)

    def customization_config(self):
        return {
            'model': PresetValue(str, options=self.available_models),
            'google_api_key': PresetValue(str, private=True),
            'temperature': PresetValue(float, min_value=0.0, max_value=1.0),
            'max_output_tokens': PresetValue(int, min_value=1, max_value=2048, include_none=True),
            'top_k': PresetValue(int, min_value=1, max_value=40),
            'top_p': PresetValue(float, min_value=0.0, max_value=1.0),
            'n': PresetValue(int, 1, 10),
        }
