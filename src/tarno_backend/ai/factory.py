"""Central factory for creating LLM providers, including fallback chains."""

from __future__ import annotations

import functools
import logging
import threading
from typing import TYPE_CHECKING

from tarno_backend.ai.fallback_provider import FallbackProvider
from tarno_backend.ai.openai_compatible import OpenAICompatibleProvider
from tarno_backend.ai.provider import RateLimiter
from tarno_backend.core.config import OpenAICompatibleConfig
from tarno_backend.security.secrets import SecretsVault

if TYPE_CHECKING:
    from tarno_backend.ai.provider import LLMProvider
    from tarno_backend.core.config import TarnoConfig

log = logging.getLogger(__name__)


def _set_rate_limit(provider: "LLMProvider", rps: float) -> None:
    if rps and rps > 0:
        provider._rate_limiter = RateLimiter(rps)


# Agent-Pool-Rate-Limit-Fix: mehrere Pool-Agenten koennen denselben Provider
# (z.B. 2-3 Mistral-Modelle) nutzen, bekommen dabei aber JEDER eine eigene
# LLMProvider-Instanz (siehe PoolWorker.__init__) - RateLimiter oben ist
# ausserdem nur ein Requests-pro-Sekunde-Drosselr pro INSTANZ, keine
# Nebenlaeufigkeits-Sperre. Ohne etwas, das ueber Instanzen hinweg pro
# Provider-NAME zaehlt, feuern mehrere gleichzeitig aktive Agenten desselben
# Providers ihre echten HTTP-Calls parallel ab und loesen den Anbieter-
# seitigen 429-Rate-Limiter aus (live bestaetigt: 3 gleichzeitige
# Mistral-Agenten). threading.Semaphore statt asyncio.Semaphore, weil jeder
# Pool-Lauf in _run_pool_task (tarno/grpc/server.py) seine eigene
# asyncio.run()-Event-Loop in einem eigenen Executor-Thread bekommt - ein
# asyncio.Semaphore waere ueber Threads hinweg nicht sicher teilbar, ein
# threading.Semaphore dagegen schon (und der eigentliche blockierende
# provider.send()-Call laeuft sowieso schon in einem Executor-Thread, nicht
# im Event-Loop-Thread, ein Blockieren dort ist also unproblematisch).
_provider_semaphores: dict[str, threading.Semaphore] = {}
_provider_semaphores_lock = threading.Lock()


def get_provider_semaphore(name: str, max_concurrent: int) -> threading.Semaphore:
    """Liefert das (pro Prozesslaufzeit einmalige) Semaphore fuer einen
    Provider-Namen - alle PoolWorker-Instanzen, die denselben Provider-Namen
    verwenden, teilen sich dasselbe Objekt, unabhaengig davon, in welchem
    Pool-Lauf/Worker sie erzeugt wurden."""
    name = name.lower()
    sem = _provider_semaphores.get(name)
    if sem is not None:
        return sem
    with _provider_semaphores_lock:
        sem = _provider_semaphores.get(name)
        if sem is None:
            sem = threading.Semaphore(max(1, max_concurrent))
            _provider_semaphores[name] = sem
        return sem


def _with_concurrency_limit(send_fn, sem: threading.Semaphore):
    @functools.wraps(send_fn)
    def _wrapped(*args, **kwargs):
        with sem:
            return send_fn(*args, **kwargs)
    return _wrapped


def _apply_concurrency_limit(provider: "LLMProvider", name: str, max_concurrent: int) -> "LLMProvider":
    sem = get_provider_semaphore(name, max_concurrent)
    # Beide HTTP-ausloesenden Methoden wrappen - send_tool_result() wird von
    # PoolWorker._run_tool_loop's Tool-Calling-Schleife fuer jede
    # Folgeantwort nach einem Tool-Aufruf aufgerufen; ohne diesen zweiten
    # Wrap wuerde die Nebenlaeufigkeits-Bremse genau dort greifen, wo sie am
    # noetigsten ist (mehrere Worker desselben Providers, die parallel
    # read_file-Tool-Calls abarbeiten), und der urspruengliche 429-Bug waere
    # nur fuer den allerersten Aufruf pro Worker behoben.
    provider.send = _with_concurrency_limit(provider.send, sem)
    provider.send_tool_result = _with_concurrency_limit(provider.send_tool_result, sem)
    return provider

# Maps a provider name to the secret name used to look it up in the vault.
# Same names as the environment variables the providers used to read
# directly, so existing setups that only ever set an env var keep working
# unchanged (SecretsVault.get() falls back to the env var automatically).
_PROVIDER_SECRET_NAMES = {
    "mistral": "MISTRAL_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "groq": "GROQ_API_KEY",
    "huggingface": "HF_TOKEN",
    "claude": "ANTHROPIC_API_KEY",
}

# Older secret names that docs/api-keys.md and the legacy first-start wizard
# (tarno/first_start/pages.py) used to point at before they were corrected to
# match _PROVIDER_SECRET_NAMES - checked as a fallback so a key already set
# under the old name keeps working instead of silently locking the user out.
_LEGACY_SECRET_ALIASES: dict[str, str] = {
    "claude": "CLAUDE_API_KEY",
}

# Defaults for OpenAI-compatible providers. Each entry is:
#   (base_url, api_key_env, default_model, context_window)
_OPENAI_COMPATIBLE_DEFAULTS: dict[str, tuple[str, str, str, int]] = {
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY", "gpt-4o-mini", 128_000),
    "glm": ("https://open.bigmodel.cn/api/paas/v4", "GLM_API_KEY", "glm-4.7-flash", 200_000),
    "perplexity": ("https://api.perplexity.ai", "PERPLEXITY_API_KEY", "sonar", 128_000),
    "meta": ("https://api.meta.ai/v1", "META_API_KEY", "llama-3.3-70b-instruct", 128_000),
    "deepseek": ("https://api.deepseek.com/v1", "DEEPSEEK_API_KEY", "deepseek-v4-flash", 1_000_000),
    "moonshot": ("https://api.moonshot.ai/v1", "MOONSHOT_API_KEY", "kimi-k3", 1_000_000),
    "qwen": ("https://dashscope.aliyuncs.com/compatible-mode/v1", "QWEN_API_KEY", "qwen3.7-plus", 1_000_000),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "openai/gpt-4o-mini", 128_000),
}


def _resolve_api_key(vault: SecretsVault, name: str) -> str:
    if name in _PROVIDER_SECRET_NAMES:
        key = vault.get(_PROVIDER_SECRET_NAMES[name]) or ""
        if not key and name in _LEGACY_SECRET_ALIASES:
            key = vault.get(_LEGACY_SECRET_ALIASES[name]) or ""
        return key
    if name in _OPENAI_COMPATIBLE_DEFAULTS:
        return vault.get(_OPENAI_COMPATIBLE_DEFAULTS[name][1]) or ""
    return ""


def provider_has_api_key(vault: SecretsVault, name: str) -> bool:
    """True if the provider needs no key at all (not in
    _PROVIDER_SECRET_NAMES or _OPENAI_COMPATIBLE_DEFAULTS, e.g. Ollama) or
    already has one configured (including legacy aliases)."""
    if name not in _PROVIDER_SECRET_NAMES and name not in _OPENAI_COMPATIBLE_DEFAULTS:
        return True
    return bool(_resolve_api_key(vault, name))


def _build_single_provider(
    config: "TarnoConfig",
    name: str,
    vault: SecretsVault | None = None,
    model_override: str | None = None,
) -> "LLMProvider":
    """Instantiate one LLM provider by name from the config.

    API keys are resolved once here via the SecretsVault (keyring /
    encrypted file / env, per config.security.secrets_backend) instead of
    each provider reading os.environ itself - see ADR discussion in the
    Phase 2 stabilization plan ("Rejected: Key-Injection per
    Dependency-Injection-Framework -> over-engineering for a single-dev
    project"). A plain resolve-and-pass-as-parameter is enough.

    model_override (Phase 1-6): picks a specific model from the curated
    catalog (tarno.ai.model_catalog) instead of the provider's configured
    default - None keeps existing behavior unchanged. For Mistral the chosen
    model is used directly; the difficulty_tiers remain active when reasoning
    is enabled and use_high_tier is toggled.
    """
    name = name.lower()
    vault = vault or SecretsVault(backend=config.security.secrets_backend)

    if name == "mistral":
        from tarno_backend.ai.mistral_client import MistralProvider
        tiers = config.llm.mistral.difficulty_tiers
        provider = MistralProvider(
            api_key=_resolve_api_key(vault, "mistral"),
            model=model_override or config.llm.mistral.model,
            max_tokens=config.llm.mistral.max_tokens,
            temperature=config.llm.mistral.temperature,
            frequency_penalty=config.llm.mistral.frequency_penalty,
            presence_penalty=config.llm.mistral.presence_penalty,
            timeout=config.llm.mistral.timeout,
            max_retries=config.llm.mistral.max_retries,
            base_delay=config.llm.mistral.base_delay,
            difficulty_tiers={"low": tiers.low, "medium": tiers.medium, "high": tiers.high},
            difficulty_enabled=tiers.enabled,
            rate_limit_rps=config.llm.mistral.rate_limit_rps,
        )
        return provider

    if name == "claude":
        from dataclasses import replace
        from tarno_backend.ai.claude_client import ClaudeProvider
        claude_config = config.llm.claude
        if model_override:
            claude_config = replace(claude_config, model=model_override)
        return ClaudeProvider(claude_config, api_key=_resolve_api_key(vault, "claude"))

    if name == "ollama":
        from tarno_backend.ai.ollama_client import OllamaProvider
        return OllamaProvider(
            model=model_override or config.llm.ollama.model,
            base_url=config.llm.ollama.base_url,
            num_ctx=config.llm.ollama.num_ctx,
            num_gpu=config.llm.ollama.num_gpu,
            temperature=config.llm.ollama.temperature,
            timeout=config.llm.ollama.timeout,
            max_retries=config.llm.ollama.max_retries,
            base_delay=config.llm.ollama.base_delay,
        )

    if name == "groq":
        from tarno_backend.ai.groq_client import GroqProvider
        provider = GroqProvider(
            api_key=_resolve_api_key(vault, "groq"),
            model=model_override or config.llm.groq.model,
            max_tokens=config.llm.groq.max_tokens,
            temperature=config.llm.groq.temperature,
            timeout=config.llm.groq.timeout,
            max_retries=config.llm.groq.max_retries,
            base_delay=config.llm.groq.base_delay,
        )
        _set_rate_limit(provider, config.llm.groq.rate_limit_rps)
        return provider

    if name == "huggingface":
        from tarno_backend.ai.huggingface_client import HuggingFaceProvider
        return HuggingFaceProvider(
            api_key=_resolve_api_key(vault, "huggingface"),
            model=model_override or config.llm.huggingface.model,
            max_tokens=config.llm.huggingface.max_tokens,
            temperature=config.llm.huggingface.temperature,
            timeout=config.llm.huggingface.timeout,
            max_retries=config.llm.huggingface.max_retries,
            base_delay=config.llm.huggingface.base_delay,
        )

    if name == "gemini":
        from tarno_backend.ai.gemini_client import GeminiProvider
        provider = GeminiProvider(
            api_key=_resolve_api_key(vault, "gemini"),
            model=model_override or config.llm.gemini.model,
            max_tokens=config.llm.gemini.max_tokens,
            temperature=config.llm.gemini.temperature,
            timeout=config.llm.gemini.timeout,
            max_retries=config.llm.gemini.max_retries,
            base_delay=config.llm.gemini.base_delay,
        )
        _set_rate_limit(provider, config.llm.gemini.rate_limit_rps)
        return provider

    if name in _OPENAI_COMPATIBLE_DEFAULTS:
        defaults = _OPENAI_COMPATIBLE_DEFAULTS[name]
        cfg = config.llm.openai_compatible.get(name) or OpenAICompatibleConfig(
            base_url=defaults[0],
            api_key_env=defaults[1],
            model=defaults[2],
            context_window=defaults[3],
        )
        api_key = _resolve_api_key(vault, name) or vault.get(cfg.api_key_env) or ""
        return OpenAICompatibleProvider(
            api_key=api_key,
            base_url=cfg.base_url or defaults[0],
            model=model_override or cfg.model or defaults[2],
            provider_name=name,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            timeout=cfg.timeout,
            max_retries=cfg.max_retries,
            base_delay=cfg.base_delay,
            rate_limit_rps=cfg.rate_limit_rps,
            context_window=cfg.context_window or defaults[3],
        )

    raise ValueError(
        f"Unbekannter LLM-Provider: {name}. "
        "Nutze 'mistral', 'claude', 'gemini', 'groq', 'huggingface', 'ollama' "
        "oder einen openai_compatible Schlüssel."
    )


def create_provider(
    config: "TarnoConfig", provider_name: str | None = None, model: str | None = None
) -> "LLMProvider":
    """Create a single LLM provider.

    If ``provider_name`` is None, the configured primary provider is used.
    ``model`` (Phase 1-6) picks a specific model from the curated catalog
    instead of the provider's configured default.
    """
    vault = SecretsVault(backend=config.security.secrets_backend)
    name = provider_name or config.llm.provider
    provider = _build_single_provider(config, name, vault=vault, model_override=model)
    return _apply_concurrency_limit(provider, name, config.coding.pool_provider_max_concurrent)


def create_provider_with_fallback(config: "TarnoConfig") -> "LLMProvider":
    """Create the primary provider plus any configured fallback providers.

    If a single provider can be built, it is returned directly. Otherwise a
    ``FallbackProvider`` is returned that tries providers in order.
    """
    primary = config.llm.provider.lower()
    fallback_names = [
        p for p in config.llm.fallback_providers
        if p and p.lower() != primary
    ]

    vault = SecretsVault(backend=config.security.secrets_backend)
    providers: list[LLMProvider] = []
    seen: set[str] = set()

    for name in [primary] + fallback_names:
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        try:
            built = _build_single_provider(config, key, vault=vault)
            providers.append(_apply_concurrency_limit(built, key, config.coding.pool_provider_max_concurrent))
        except Exception:
            log.exception(
                "Provider '%s' konnte nicht initialisiert werden und wird uebersprungen",
                key,
            )

    if not providers:
        raise RuntimeError("Kein LLM-Provider konnte initialisiert werden")

    if len(providers) == 1:
        return providers[0]

    return FallbackProvider(providers)
