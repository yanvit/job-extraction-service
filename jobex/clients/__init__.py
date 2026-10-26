from .base import FatalError, LLMClient, LLMResponse, RetryableError, Sampling


def make_client(model_cfg: dict, defaults: dict) -> LLMClient:
    """Фабрика клієнтів за конфігом моделі (config.yaml -> models[])."""
    kw = {
        "timeout_s": model_cfg.get("timeout_s", defaults.get("timeout_s", 60)),
        "max_retries": model_cfg.get("max_retries", defaults.get("max_retries", 3)),
        "backoff_base_s": defaults.get("backoff_base_s", 1.0),
        "max_tokens": model_cfg.get("max_tokens", defaults.get("max_tokens", 2048)),
    }
    provider = model_cfg["provider"]
    if provider == "anthropic":
        from .anthropic_client import AnthropicClient
        return AnthropicClient(model_cfg["model"],
                               structured_mode=model_cfg.get("structured_mode", "json_schema"),
                               supports_sampling=model_cfg.get("supports_sampling", True), **kw)
    if provider == "ollama":
        from .ollama_client import OllamaClient
        return OllamaClient(model_cfg["model"], base_url=model_cfg.get("base_url"),
                            num_ctx=model_cfg.get("num_ctx", 8192), think=model_cfg.get("think"), **kw)
    raise ValueError(f"unknown provider: {provider}")


__all__ = ["FatalError", "LLMClient", "LLMResponse", "RetryableError", "Sampling", "make_client"]
