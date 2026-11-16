"""Tests for genai-nodes. No network: providers and SDK modules are faked."""

import asyncio
import json
import sys
import threading
import time
import types
from typing import Any, cast

import numpy as np
import pytest

from nonoha_core._testing import assert_all_ports_documented
from nonoha_core.node_registry import NodeRegistry

import genai_nodes
from genai_nodes import providers as prov
from genai_nodes.nodes import LLMDescribeImageNode, LLMInterpretNode
from genai_nodes.runner import DailyCounter

TRUE = [np.array([True])]
JPEG = [np.frombuffer(b"\xff\xd8fake-jpeg\xff\xd9", dtype=np.uint8)[np.newaxis]]


class FakeProvider(prov.Provider):
    name = "fake"
    default_model = "fake-1"
    default_key_env = "FAKE_API_KEY"

    def __init__(self, delay=0.0, replies=None):
        self.delay = delay
        self.calls = []
        self.replies = list(replies or [])

    async def generate(self, req):
        self.calls.append(req)
        await asyncio.sleep(self.delay)
        if self.replies:
            reply = self.replies.pop(0)
            if isinstance(reply, Exception):
                raise reply
            return reply
        text = json.dumps({"scene": "forest", "calm": 0.8}) if req.schema else "a forest"
        return prov.Response(text, prov.extract_json(text) if req.schema else None, {"input_tokens": 10})


@pytest.fixture
def fake(monkeypatch):
    provider = FakeProvider()
    monkeypatch.setitem(prov.PROVIDERS, "fake", provider)
    monkeypatch.setenv("FAKE_API_KEY", "test-key")
    return provider


def collect(port):
    got = []
    port.subscribe(id(got), lambda v: got.append(str(v[0][0]) if v[0].dtype.kind in "UTO" else v[0][0]))
    return got


def make(cls, **props):
    node = cls("n")
    node.set_properties({"provider": "fake", "min_interval_s": 0.0, **props})
    return node


async def settle(seconds=0.05):
    await asyncio.sleep(seconds)


def test_registration_documents_ports():
    registry = NodeRegistry()
    genai_nodes.register_nodes(registry)
    assert set(registry.node_definitions) == {"LLMDescribeImage", "LLMInterpret"}
    assert_all_ports_documented(registry)


def test_extract_json():
    assert prov.extract_json('{"a": 1}') == {"a": 1}
    assert prov.extract_json('Sure!\n```json\n{"a": 2}\n```') == {"a": 2}
    assert prov.extract_json('Here: {"a": 3} hope it helps') == {"a": 3}
    assert prov.extract_json("no json") is None


def test_no_trigger_no_call_and_missing_key(fake, monkeypatch):
    async def run():
        node = make(LLMDescribeImageNode)
        node._do_activate()
        errors = collect(node.outputs["error"])
        node.inputs["image"].dispatch(JPEG)
        await settle()
        assert fake.calls == []  # an image alone never sends anything
        monkeypatch.delenv("FAKE_API_KEY")
        node.inputs["trigger"].dispatch(TRUE)
        await settle()
        node._do_finalize()
        return errors

    errors = asyncio.run(run())
    assert fake.calls == [] and "FAKE_API_KEY" in errors[-1]


def test_describe_image_round_trip(fake):
    async def run():
        node = make(LLMDescribeImageNode)
        node._do_activate()
        text, busy, usage = collect(node.outputs["text"]), collect(node.outputs["busy"]), collect(node.outputs["usage"])
        node.inputs["image"].dispatch(JPEG)
        node.inputs["trigger"].dispatch(TRUE)
        await settle()
        node._do_finalize()
        return text, busy, usage

    text, busy, usage = asyncio.run(run())
    assert text == ["a forest"] and busy == [True, False]
    assert json.loads(usage[0])["input_tokens"] == 10
    req = fake.calls[0]
    assert req.image.startswith(b"\xff\xd8") and req.model == "fake-1" and req.api_key == "test-key"


def test_one_in_flight_newest_wins(fake):
    fake.delay = 0.2

    async def run():
        node = make(LLMDescribeImageNode)
        node._do_activate()
        node.inputs["image"].dispatch(JPEG)
        for prompt in ("first", "second", "third"):
            node.set_properties({"prompt": prompt})
            node.inputs["trigger"].dispatch(TRUE)
        await settle(0.6)
        node._do_finalize()

    asyncio.run(run())
    assert [c.prompt for c in fake.calls] == ["first", "third"]  # "second" was replaced


def test_min_interval_and_daily_cap(fake):
    async def run():
        node = make(LLMDescribeImageNode, min_interval_s=0.3, max_requests_per_day=2)
        node._do_activate()
        errors = collect(node.outputs["error"])
        node.inputs["image"].dispatch(JPEG)
        node.inputs["trigger"].dispatch(TRUE)
        await settle()
        node.inputs["trigger"].dispatch(TRUE)
        await settle(0.1)
        assert len(fake.calls) == 1  # still inside the interval
        await settle(0.35)
        assert len(fake.calls) == 2
        node.inputs["trigger"].dispatch(TRUE)
        await settle(0.4)
        node._do_finalize()
        return errors

    errors = asyncio.run(run())
    assert len(fake.calls) == 2 and "daily limit" in errors[-1]


def test_rate_limit_pauses_then_retries_once(fake):
    fake.replies = [prov.RateLimited("fake: 429", retry_after=0.2)]

    async def run():
        node = make(LLMDescribeImageNode)
        node._do_activate()
        errors, text = collect(node.outputs["error"]), collect(node.outputs["text"])
        node.inputs["image"].dispatch(JPEG)
        node.inputs["trigger"].dispatch(TRUE)
        await settle(0.1)
        assert len(fake.calls) == 1 and "429" in errors[-1]
        await settle(0.3)
        node._do_finalize()
        return text

    text = asyncio.run(run())
    assert len(fake.calls) == 2 and text == ["a forest"]


def test_send_images_off_blocks_requests(fake):
    async def run():
        node = make(LLMDescribeImageNode, send_images=False)
        node._do_activate()
        errors = collect(node.outputs["error"])
        node.inputs["image"].dispatch(JPEG)
        node.inputs["trigger"].dispatch(TRUE)
        await settle()
        node._do_finalize()
        return errors

    errors = asyncio.run(run())
    assert fake.calls == [] and "send_images" in errors[-1]


def test_interpret_schema_fields_become_ports(fake):
    schema = {"type": "object", "properties": {"scene": {"type": "string"}, "calm": {"type": "number"}}}

    async def run():
        node = make(LLMInterpretNode, schema=json.dumps(schema), labels="energy,speed")
        assert {"out_scene", "out_calm"} <= set(node.outputs)
        node._do_activate()
        scene, calm = collect(node.outputs["out_scene"]), collect(node.outputs["out_calm"])
        node.inputs["features"].dispatch([np.array([[1.5, 0.25]], np.float32)])
        node.inputs["context"].dispatch([np.array(["a quiet park"])])
        node.inputs["trigger"].dispatch(TRUE)
        await settle()
        node._do_finalize()
        return node, scene, calm

    node, scene, calm = asyncio.run(run())
    assert scene == ["forest"] and calm[0] == pytest.approx(0.8)
    prompt = fake.calls[0].prompt
    assert "- energy: 1.5" in prompt and "- speed: 0.25" in prompt and "a quiet park" in prompt
    assert fake.calls[0].image is None  # send_images defaults to off for Interpret

    node.set_properties({"schema": ""})
    assert "out_scene" not in node.outputs
    with pytest.raises(ValueError):
        node.set_properties({"schema": "{not json"})


def test_daily_counter_persists(tmp_path):
    path = tmp_path / "usage.json"
    a = DailyCounter(path, "node", lambda: "2026-11-15")
    a.increment()
    a.increment()
    assert DailyCounter(path, "node", lambda: "2026-11-15").count == 2
    assert DailyCounter(path, "node", lambda: "2026-11-16").count == 0  # new day


def test_gemini_provider_runs_off_the_loop(monkeypatch):
    captured = {}

    class FakeInteractions:
        def create(self, **kwargs):
            captured.update(kwargs)
            time.sleep(0.2)  # a blocking SDK call
            return types.SimpleNamespace(output_text='{"scene": "river"}', usage=None)

    class FakeClient:
        def __init__(self, api_key, http_options=None):
            captured["api_key"] = api_key
            captured["http_options"] = http_options
            self.interactions = FakeInteractions()

    genai = types.ModuleType("google.genai")
    setattr(genai, "Client", FakeClient)
    google = types.ModuleType("google")
    setattr(google, "genai", genai)
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.genai", genai)

    async def run():
        ticks = 0

        async def ticker():
            nonlocal ticks
            while True:
                await asyncio.sleep(0.01)
                ticks += 1

        task = asyncio.create_task(ticker())
        req = prov.Request(prompt="where?", api_key="k", model="gemini-x", image=b"img",
                           schema={"type": "object", "properties": {"scene": {"type": "string"}}})
        response = await prov.GeminiProvider().generate(req)
        task.cancel()
        return response, ticks

    response, ticks = asyncio.run(run())
    assert response.data == {"scene": "river"}
    assert ticks >= 5  # the event loop kept running during the blocking call (Windows timers are ~15 ms)
    assert captured["model"] == "gemini-x" and captured["api_key"] == "k"
    assert captured["input"][1]["type"] == "image" and captured["response_format"]["mime_type"] == "application/json"
    assert captured["generation_config"] == {"max_output_tokens": 1024}
    assert captured["automatic_continuation"] is False and captured["store"] is False
    assert captured["http_options"] == {"retry_options": {"attempts": 1}}  # no SDK retries


def test_sdk_import_runs_off_the_loop(monkeypatch):
    """The first import of a big SDK takes seconds: the event loop must keep running meanwhile."""
    fake_sdk = types.SimpleNamespace(Client=lambda api_key, http_options=None: types.SimpleNamespace(
        interactions=types.SimpleNamespace(create=lambda **kw: types.SimpleNamespace(output_text="ok", usage=None))))

    def slow_import(name):
        time.sleep(0.3)  # importing the SDK the first time
        return fake_sdk

    monkeypatch.setattr(prov.importlib, "import_module", slow_import)

    async def run():
        ticks = 0

        async def ticker():
            nonlocal ticks
            while True:
                await asyncio.sleep(0.01)
                ticks += 1

        task = asyncio.create_task(ticker())
        response = await prov.GeminiProvider().generate(prov.Request(prompt="p", api_key="k", model="m"))
        task.cancel()
        return response, ticks

    response, ticks = asyncio.run(run())
    assert response.text == "ok"
    assert ticks >= 10  # about 30 ticks of 10 ms while the import ran (Windows timers are ~15 ms)


def test_providers_preload_their_sdk_in_a_worker_thread(monkeypatch):
    imported = []
    main = threading.get_ident()

    def record(name):
        imported.append((name, threading.get_ident() != main))
        if name == "anthropic":
            raise ImportError("not installed here")  # reported later, by a request
        return types.SimpleNamespace()

    monkeypatch.setattr(prov.importlib, "import_module", record)

    async def run():
        loop = asyncio.get_running_loop()
        for name in ("gemini", "openrouter", "anthropic"):
            prov.PROVIDERS[name].preload(loop)
        await asyncio.sleep(0.2)

    asyncio.run(run())
    assert sorted(imported) == [("anthropic", True), ("google.genai", True), ("httpx2", True)]


def test_openrouter_provider_request_and_429(monkeypatch):
    sent = {}

    class FakeResponse:
        def __init__(self, status, payload, headers=None):
            self.status_code, self._payload, self.headers = status, payload, headers or {}
            self.text = json.dumps(payload)

        def json(self):
            return self._payload

    replies = [FakeResponse(429, {}, {"retry-after": "7"}),
               FakeResponse(200, {"choices": [{"message": {"content": "a crosswalk"}}], "usage": {"total_tokens": 5}})]

    class FakeAsyncClient:
        def __init__(self, timeout):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, url, headers, json):
            sent.update(url=url, headers=headers, body=json)
            return replies.pop(0)

    monkeypatch.setitem(sys.modules, "httpx2", types.SimpleNamespace(AsyncClient=FakeAsyncClient))
    req = prov.Request(prompt="what?", api_key="k", model="openrouter/free", image=b"img")

    with pytest.raises(prov.RateLimited) as err:
        asyncio.run(prov.OpenRouterProvider().generate(req))
    assert err.value.retry_after == 7.0
    response = asyncio.run(prov.OpenRouterProvider().generate(req))
    assert response.text == "a crosswalk"
    content = sent["body"]["messages"][-1]["content"]
    assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert sent["headers"]["Authorization"] == "Bearer k"


def test_nodes_declare_the_key_they_read():
    for cls in (LLMDescribeImageNode, LLMInterpretNode):
        env = cls.describe({"provider": "gemini"}).env
        assert [(e.name, e.url) for e in env] == [("GEMINI_API_KEY", "https://aistudio.google.com/apikey")]
        assert cls.describe({"provider": "anthropic", "api_key_env": "MY_CLAUDE_KEY"}).env[0].name == "MY_CLAUDE_KEY"
        assert cls.describe({"provider": "nope"}).env == ()
    registry = NodeRegistry()
    genai_nodes.register_nodes(registry)
    assert registry.node_definitions["LLMInterpret"].metadata.env[0].name == "GEMINI_API_KEY"


def test_key_comes_from_environment_values_set_while_running(fake, monkeypatch):
    from nonoha_core import environment

    monkeypatch.delenv("FAKE_API_KEY")

    async def run():
        node = make(LLMDescribeImageNode)
        node._do_activate()
        errors = collect(node.outputs["error"])
        node.inputs["image"].dispatch(JPEG)
        node.inputs["trigger"].dispatch(TRUE)
        await settle()
        assert fake.calls == [] and "FAKE_API_KEY is not set" in errors[-1]
        environment.update({"FAKE_API_KEY": "entered-in-the-editor"})  # no restart
        node.inputs["trigger"].dispatch(TRUE)
        await settle()
        node._do_finalize()

    with environment.using({}):
        asyncio.run(run())
    assert [c.api_key for c in fake.calls] == ["entered-in-the-editor"]


def test_errors_and_repr_never_carry_the_key(fake):
    fake.replies = [prov.ProviderError("bad request for key test-key"), RuntimeError("boom test-key")]

    async def run():
        node = make(LLMDescribeImageNode)
        node._do_activate()
        errors = collect(node.outputs["error"])
        node.inputs["image"].dispatch(JPEG)
        node.inputs["trigger"].dispatch(TRUE)
        await settle()
        node.inputs["trigger"].dispatch(TRUE)
        await settle()
        node._do_finalize()
        return errors

    errors = asyncio.run(run())
    assert errors and all("test-key" not in e for e in errors)
    assert any("***" in e for e in errors)
    assert "test-key" not in repr(fake.calls[0])


class FakeGraph:
    """Records the busy outputs dispatched before each commit (flush_outputs)."""

    persist = None

    def __init__(self):
        self.commits = []
        self.busy = []

    def flush_outputs(self):
        self.commits.append(list(self.busy))


def test_async_paths_commit_once_and_the_trigger_does_not(fake):
    async def run():
        node = make(LLMDescribeImageNode, min_interval_s=0.2)
        graph = FakeGraph()
        node.set_parent_graph(cast(Any, graph))  # only flush_outputs and persist are used
        node._do_activate()
        node.outputs["busy"].subscribe(id(graph), lambda v: graph.busy.append(bool(v[0][0])))
        node.inputs["image"].dispatch(JPEG)
        node.inputs["trigger"].dispatch(TRUE)
        at_trigger = list(graph.commits)
        await settle()
        node.inputs["trigger"].dispatch(TRUE)  # inside the interval: the timer sends it
        await settle(0.3)
        node._do_finalize()
        return at_trigger, graph.commits

    at_trigger, commits = asyncio.run(run())
    assert at_trigger == []  # inside the trigger's dispatch: its originator commits
    assert commits == [
        [True, False],  # the request task, after busy went back to False
        [True, False, True],  # the timer that started the second request
        [True, False, True, False],  # the second request task
    ]


def test_interpret_keeps_a_copy_of_the_features(fake):
    async def run():
        node = make(LLMInterpretNode, labels="a")
        node._do_activate()
        buffer = np.array([[1.5]], np.float32)
        node.inputs["features"].dispatch([buffer])
        buffer[0, 0] = 9.0  # the sender reuses its buffer after the dispatch
        node.inputs["trigger"].dispatch(TRUE)
        await settle()
        node._do_finalize()

    asyncio.run(run())
    assert "- a: 1.5" in fake.calls[0].prompt
