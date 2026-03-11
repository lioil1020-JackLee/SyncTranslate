from __future__ import annotations

import json
from dataclasses import dataclass
from urllib import error, request


@dataclass(slots=True)
class OllamaClient:
    backend: str = "ollama"
    base_url: str = "http://127.0.0.1:11434"
    model: str = "llama3.1:8b"
    temperature: float = 0.2
    top_p: float = 0.9
    request_timeout_sec: float = 20.0

    def health_check(self) -> tuple[bool, str]:
        try:
            if self.backend == "lm_studio":
                self._chat_completion(messages=[{"role": "user", "content": "ping"}], max_tokens=2)
            else:
                self._request_json("/api/tags", method="GET")
            return True, "ok"
        except Exception as exc:
            return False, str(exc)

    def list_models(self) -> list[str]:
        if self.backend == "lm_studio":
            payload = self._request_json("/v1/models", method="GET")
            data = payload.get("data", [])
            return [str(item.get("id", "")) for item in data if isinstance(item, dict)]
        payload = self._request_json("/api/tags", method="GET")
        models = payload.get("models", [])
        result: list[str] = []
        for item in models:
            if isinstance(item, dict):
                result.append(str(item.get("name", "")))
        return [name for name in result if name]

    def translate(self, text: str, *, source_lang: str, target_lang: str, context: list[str] | None = None) -> str:
        if not text.strip():
            return ""
        context_text = "\n".join((context or [])[-6:])
        system_prompt = (
            "You are a translation engine. "
            "Return only translated text without notes or markdown. "
            "Keep named entities."
        )
        user_prompt = (
            f"Source language: {source_lang}\n"
            f"Target language: {target_lang}\n"
            f"Context:\n{context_text}\n\n"
            f"Text:\n{text}"
        )
        if self.backend == "lm_studio":
            response = self._chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            )
            return response.strip()

        payload = {
            "model": self.model,
            "prompt": f"{system_prompt}\n\n{user_prompt}",
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "top_p": self.top_p,
            },
        }
        data = self._request_json("/api/generate", payload=payload)
        return str(data.get("response", "")).strip()

    def _chat_completion(self, *, messages: list[dict[str, str]], max_tokens: int = 512) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": max_tokens,
        }
        data = self._request_json("/v1/chat/completions", payload=payload)
        choices = data.get("choices", [])
        if not choices:
            return ""
        message = choices[0].get("message", {})
        return str(message.get("content", "")).strip()

    def _request_json(self, path: str, payload: dict[str, object] | None = None, method: str = "POST") -> dict:
        url = self.base_url.rstrip("/") + path
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {"Accept": "application/json"}
        if data is not None:
            headers["Content-Type"] = "application/json"
        req = request.Request(url=url, data=data, headers=headers, method=method)
        try:
            with request.urlopen(req, timeout=self.request_timeout_sec) as resp:
                content = resp.read().decode("utf-8", errors="replace")
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise ValueError(f"HTTP {exc.code}: {body or exc.reason}") from exc
        except error.URLError as exc:
            reason = getattr(exc, "reason", exc)
            raise ValueError(f"local llm connection failed: {reason}") from exc
        if not content.strip():
            return {}
        return json.loads(content)
