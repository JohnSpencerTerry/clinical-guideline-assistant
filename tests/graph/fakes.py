"""Lightweight duck-typed fake chat models for testing graph nodes without a real API key."""


class FakeMessage:
    def __init__(self, content: str):
        self.content = content


class FakeChatModel:
    """Fakes .invoke(prompt) -> message with .content, and .with_structured_output(Model)."""

    def __init__(self, *, text: str | None = None, structured=None):
        self._text = text
        self._structured = structured

    def invoke(self, prompt):
        return FakeMessage(self._text)

    def with_structured_output(self, model):
        return _FakeStructuredModel(self._structured)


class _FakeStructuredModel:
    def __init__(self, obj):
        self._obj = obj

    def invoke(self, prompt):
        return self._obj
