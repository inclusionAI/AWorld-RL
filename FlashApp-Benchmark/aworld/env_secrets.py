from pydantic_settings import BaseSettings, SettingsConfigDict


class Secrets(BaseSettings):
    # android agent
    android_model_id: str = ""
    android_openai_api_key: str = ""
    android_base_url: str = ""

    # model key
    claude_api_key: str = ""
    deep_seek_api_key: str = ""
    mistral_api_key: str = ""
    openai_api_key: str = ""
    google_api_key: str = ""
    azure_openai_api_key: str = ""
    qwen_api_key: str = ""
    moonshot_api_key: str = ""

    model_config = SettingsConfigDict(case_sensitive=False, extra='ignore')


secrets: 'Secrets' = Secrets()
