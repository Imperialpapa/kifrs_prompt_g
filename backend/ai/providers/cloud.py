"""
Cloud AI Provider Mixin
=======================
Multi-provider cloud API integration (OpenAI, Anthropic, Gemini)
"""

import json
import os
import re
import warnings

from utils.logger import get_logger

logger = get_logger("ai.providers.cloud")


def parse_json_response(response: str) -> dict:
    """
    AI 응답에서 JSON을 안전하게 추출하는 단계적 파서

    1단계: json.loads 직접 시도
    2단계: markdown 코드블록(```json ... ```) 추출
    3단계: 첫 번째 { 부터 마지막 } 까지 추출 (기존 방식)

    Returns:
        dict: 파싱된 JSON 딕셔너리. 실패 시 빈 dict 반환
    """
    if not response or not response.strip():
        return {}

    text = response.strip()

    # 1단계: 직접 파싱 시도
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass

    # 2단계: markdown 코드블록에서 추출
    code_block_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
    if code_block_match:
        try:
            return json.loads(code_block_match.group(1).strip())
        except (json.JSONDecodeError, ValueError):
            pass

    # 3단계: 가장 바깥쪽 중괄호 매칭 (중첩 브레이스 안전 처리)
    start_idx = text.find('{')
    if start_idx == -1:
        return {}

    depth = 0
    end_idx = -1
    in_string = False
    escape_next = False

    for i in range(start_idx, len(text)):
        c = text[i]
        if escape_next:
            escape_next = False
            continue
        if c == '\\':
            escape_next = True
            continue
        if c == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                end_idx = i
                break

    if end_idx > start_idx:
        try:
            return json.loads(text[start_idx:end_idx + 1])
        except (json.JSONDecodeError, ValueError):
            pass

    return {}

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


MAX_RETRIES = int(os.getenv("AI_MAX_RETRIES", "2"))


class CloudProviderMixin:
    """
    Cloud AI Provider 호출 메서드 모음 (Mixin)
    OpenAI, Anthropic(Claude), Google Gemini 지원

    Features:
    - Exponential backoff retry (일시적 실패 대응)
    - asyncio.to_thread 비동기 래핑 (이벤트루프 차단 방지)
    """

    # Provider → 동기 메서드 매핑
    _PROVIDER_METHOD_MAP = {
        "anthropic": "_call_claude_api",
        "claude": "_call_claude_api",
        "gemini": "_call_gemini_api",
        "openai": "_call_openai_api",
    }

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

    def _get_provider_method(self, provider: str):
        """Provider에 해당하는 동기 API 호출 메서드를 반환"""
        method_name = self._PROVIDER_METHOD_MAP.get(provider, f"_call_{provider}_api")
        method = getattr(self, method_name, None)
        if method is None:
            raise ValueError(f"Unsupported AI provider: {provider}")
        return method

    async def _call_cloud_ai(self, prompt: str, provider: str) -> str:
        """
        비동기 Cloud AI 호출 (asyncio.to_thread + exponential backoff retry)

        이벤트루프를 차단하지 않으면서 재시도 로직을 적용합니다.
        """
        import asyncio

        method = self._get_provider_method(provider)

        for attempt in range(MAX_RETRIES + 1):
            try:
                result = await asyncio.to_thread(method, prompt)
                return result
            except Exception as e:
                if attempt < MAX_RETRIES:
                    wait_time = 2 ** attempt  # 1초, 2초
                    logger.warning(
                        "Cloud AI call failed (attempt %d/%d), retrying in %ds: %s",
                        attempt + 1, MAX_RETRIES + 1, wait_time, e
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error("Cloud AI call failed after %d attempts: %s", MAX_RETRIES + 1, e)
                    raise

    async def _call_cloud_ai_async(self, prompt: str, provider: str) -> str:
        """
        비동기 Cloud AI 호출 (analyzer mixin용 별칭)

        _call_cloud_ai_sync를 대체합니다.
        """
        return await self._call_cloud_ai(prompt, provider)

    def _call_cloud_ai_sync(self, prompt: str, provider: str) -> str:
        """
        동기적 Cloud AI 호출 (하위 호환 유지)

        주의: 이 메서드는 이벤트루프를 차단합니다.
        가능하면 _call_cloud_ai() 또는 _call_cloud_ai_async()를 사용하세요.
        """
        method = self._get_provider_method(provider)
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
            max_tokens=4000,
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
