"""
Cloud AI Provider Mixin
=======================
Multi-provider cloud API integration (OpenAI, Anthropic, Gemini)
"""

import os
import warnings

from utils.logger import get_logger

logger = get_logger("ai.providers.cloud")

# Suppress Google Generative AI deprecation warning
warnings.filterwarnings("ignore", category=FutureWarning, module="google.generativeai")

# Optional Imports with Graceful Fallback
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


class CloudProviderMixin:
    """
    Cloud AI Provider 호출 메서드 모음 (Mixin)
    OpenAI, Anthropic(Claude), Google Gemini 지원
    """

    def _check_provider_availability(self, provider: str) -> bool:
        """
        지정된 Provider가 사용 가능한지 확인

        Args:
            provider: "openai", "anthropic", "gemini"

        Returns:
            bool: 사용 가능 여부
        """
        if provider == "openai":
            return OPENAI_AVAILABLE and bool(os.getenv("OPENAI_API_KEY"))
        elif provider in ["anthropic", "claude"]:
            return ANTHROPIC_AVAILABLE and bool(os.getenv("ANTHROPIC_API_KEY"))
        elif provider == "gemini":
            return GEMINI_AVAILABLE and bool(os.getenv("GEMINI_API_KEY"))
        return False

    async def _call_cloud_ai(self, prompt: str, provider: str) -> str:
        """선택된 Provider의 API 호출 (OpenAI JSON 모드 적극 활용)"""
        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            client = openai.OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=os.getenv("AI_MODEL_VERSION_OPENAI", "gpt-4o"),
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            return response.choices[0].message.content

        # Provider 이름 -> 메서드명 매핑
        provider_method_map = {
            "anthropic": "_call_claude_api",
            "claude": "_call_claude_api",
            "gemini": "_call_gemini_api",
            "openai": "_call_openai_api",
        }
        method_name = provider_method_map.get(provider, f"_call_{provider}_api")
        method = getattr(self, method_name, None)
        if method is None:
            raise ValueError(f"Unsupported AI provider: {provider}")
        return method(prompt)

    def _call_claude_api(self, prompt: str) -> str:
        """Anthropic Claude API"""
        api_key = os.getenv("ANTHROPIC_API_KEY")
        model = os.getenv("AI_MODEL_VERSION_ANTHROPIC", "claude-3-haiku-20240307")

        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model=model,
            max_tokens=4000,
            temperature=0.0,
            system="You are a strict data validation rule parser. Output JSON only.",
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text

    def _call_openai_api(self, prompt: str) -> str:
        """OpenAI GPT API"""
        api_key = os.getenv("OPENAI_API_KEY")
        model = os.getenv("AI_MODEL_VERSION_OPENAI", "gpt-4o")

        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            temperature=0.0,
            messages=[
                {"role": "system", "content": "You are a strict data validation rule parser. Output JSON only."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content

    def _call_gemini_api(self, prompt: str) -> str:
        """Google Gemini API"""
        api_key = os.getenv("GEMINI_API_KEY")
        model = os.getenv("AI_MODEL_VERSION_GEMINI", "gemini-1.5-flash")

        genai.configure(api_key=api_key)
        gemini_model = genai.GenerativeModel(
            model,
            generation_config={"response_mime_type": "application/json"}
        )
        response = gemini_model.generate_content(prompt)
        return response.text

    def _call_cloud_ai_sync(self, prompt: str, provider: str) -> str:
        """동기적 Cloud AI 호출 (await 없이)"""
        provider_method_map = {
            "anthropic": "_call_claude_api",
            "claude": "_call_claude_api",
            "gemini": "_call_gemini_api",
            "openai": "_call_openai_api",
        }
        method_name = provider_method_map.get(provider, f"_call_{provider}_api")
        method = getattr(self, method_name, None)
        if method is None:
            raise ValueError(f"Unsupported AI provider: {provider}")
        return method(prompt)
