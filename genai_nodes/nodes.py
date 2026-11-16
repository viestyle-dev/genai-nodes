"""LLMDescribeImage and LLMInterpret nodes."""

from __future__ import annotations

import asyncio
import json
import logging

import numpy as np

from nonoha_core.data_type import DataType, PortValueType, PropertyDataType
from nonoha_core import environment
from nonoha_core.node import BaseNode, EnvDoc, NodeDoc
from nonoha_core.port import PortValue, StaticTypeInputPort, StaticTypeOutputPort

from .providers import PROVIDERS, Provider, Request, Response
from .runner import DailyCounter, RequestRunner, today_utc

logger = logging.getLogger(__name__)

STR = PortValueType(DataType.String)
BOOL = PortValueType(DataType.Bool)
F32 = PortValueType(DataType.Float32)
ANY = PortValueType(DataType.Any)
BYTES = PortValueType(DataType.Uint8, (-1,))

_SCHEMA_PORT_TYPES = {"number": F32, "integer": F32, "string": STR, "boolean": BOOL}

COMMON_PROPERTIES = {
    "provider": {"description": "gemini (free tier), openrouter (free models) or anthropic (paid)", "default_value": "gemini"},
    "model": {"description": "Model id; empty = the provider's default", "default_value": ""},
    "api_key_env": {"description": "Name of the environment value holding the API key; empty = the provider's default (e.g. GEMINI_API_KEY). Set the value in the editor's Environment settings, the Hub or .env, never in the graph", "default_value": ""},
    "system": {"description": "System instruction", "default_value": ""},
    "min_interval_s": {"description": "Minimum seconds between requests", "default_value": 10.0},
    "max_requests_per_day": {"description": "Requests allowed per UTC day (caps cost)", "default_value": 200},
    "timeout_s": {"description": "Seconds before a request is abandoned", "default_value": 30.0},
    "max_tokens": {"description": "Maximum output tokens", "default_value": 1024},
}

_PROVIDER_NOTE = (
    " Requests start only on 'trigger'; one runs at a time (a newer trigger "
    "replaces a waiting one), they are spaced by min_interval_s and capped "
    "per day. The API key is an environment value (see 'env'), read when "
    "a request is made and never stored in the graph."
)


def _env_docs(properties) -> tuple[EnvDoc, ...]:
    """The API key this node reads with these properties."""
    props = properties or {}
    provider = PROVIDERS.get(str(props.get("provider") or "gemini"))
    if provider is None:
        return ()
    name = str(props.get("api_key_env") or "") or provider.default_key_env
    return (EnvDoc(name, f"API key for {provider.name}", provider.key_url),)


def _in(node, name, vtype, cb):
    port = StaticTypeInputPort(name, [vtype])
    port.subscribe(id(node), cb)
    node.inputs[name] = port
    return port


def _out(node, name, vtype):
    port = StaticTypeOutputPort(name, [vtype])
    node.outputs[name] = port
    return port


def _text(value: str) -> list:
    return [np.array([value])]


class _LLMNode(BaseNode):
    """Shared wiring: trigger, runner, text/json/busy/error/usage outputs."""

    def _init_llm(self):
        self._outs = {
            "text": _out(self, "text", STR),
            "json": _out(self, "json", STR),
            "busy": _out(self, "busy", BOOL),
            "error": _out(self, "error", STR),
            "usage": _out(self, "usage", STR),
        }
        _in(self, "trigger", ANY, self._on_trigger)
        self._runner: RequestRunner | None = None

    # -- lifecycle -------------------------------------------------------------

    def activate(self):
        path = None
        graph = self.parent_graph
        persist = graph.persist if graph is not None else None
        if persist is not None:
            path = persist.data_dir_path / "genai-nodes-usage.json"
        self._runner = RequestRunner(
            provider_for=self._provider,
            on_result=self._deliver_result,
            on_error=self._deliver_error,
            on_busy=self._deliver_busy,
            on_commit=self._flush,
            settings=lambda: self.properties,
            counter=DailyCounter(path, self.name, today_utc),
        )
        self._runner.start(asyncio.get_running_loop())

    def finalize(self):
        if self._runner is not None:
            self._runner.stop()

    # -- requests --------------------------------------------------------------

    def _provider(self) -> Provider:
        name = str(self.properties["provider"])
        if name not in PROVIDERS:
            raise ValueError(f"provider must be one of {sorted(PROVIDERS)}")
        return PROVIDERS[name]

    def _build_request(self) -> Request | None:  # overridden
        raise NotImplementedError

    def _on_trigger(self, values: list[PortValue]):
        if self._runner is None:
            self._deliver_error("not running (node is not active)")
            return
        provider = self._provider()
        key_env = str(self.properties["api_key_env"]) or provider.default_key_env
        # Read now, not kept: a key set while running applies to the next request
        api_key = environment.get(key_env)
        if not api_key:
            self._deliver_error(f"{key_env} is not set (editor: Environment settings, or .env)")
            return
        req = self._build_request()
        if req is None:
            return
        req.api_key = api_key
        req.model = str(self.properties["model"]) or provider.default_model
        req.system = str(self.properties["system"])
        req.timeout = float(self.properties["timeout_s"])
        req.max_tokens = int(self.properties["max_tokens"])
        self._runner.submit(req)

    # -- results (the runner commits them: see RequestRunner) -----------------------

    def _flush(self):
        if self.parent_graph is not None:
            self.parent_graph.flush_outputs()

    def _deliver_result(self, req: Request, response: Response):
        self._outs["text"].dispatch(_text(response.text))
        if response.data is not None:
            self._outs["json"].dispatch(_text(json.dumps(response.data, ensure_ascii=False)))
            self._dispatch_fields(response.data)
        self._outs["usage"].dispatch(_text(json.dumps(response.usage, default=str)))

    def _dispatch_fields(self, data):  # overridden by LLMInterpret
        pass

    def _deliver_error(self, message: str):
        logger.warning("%s: %s", self.name, message)
        self._outs["error"].dispatch(_text(message))

    def _deliver_busy(self, busy: bool):
        self._outs["busy"].dispatch([np.array([busy])])

    def _set_common(self, properties: dict[str, PropertyDataType]):
        for key, desc in COMMON_PROPERTIES.items():
            if key in properties:
                kind = type(desc["default_value"])
                self.properties[key] = kind(properties[key])
        if "provider" in properties:
            self._provider()  # validate


class LLMDescribeImageNode(_LLMNode):
    """Ask an LLM what an image shows."""

    @classmethod
    def describe(cls, properties=None):
        return NodeDoc(
            description=(
                "Sends the latest image (JPEG/PNG bytes, e.g. from Imencode) "
                "with a prompt to a cloud LLM and outputs its description. "
                "With a JSON Schema in 'schema', also outputs structured JSON." + _PROVIDER_NOTE
            ),
            category="AI",
            inputs={
                "image": "Encoded image bytes u8[?] (the latest one is sent)",
                "trigger": "Any value: send a request now (subject to pacing)",
            },
            outputs={
                "text": "The model's reply",
                "json": "Reply parsed as JSON (when 'schema' is set)",
                "busy": "True while a request is running",
                "error": "Last error (missing key, rate limit, daily cap, ...)",
                "usage": "Token usage of the last request, as JSON",
            },
            env=_env_docs(properties),
        )

    PROPERTIES = {
        **COMMON_PROPERTIES,
        "prompt": {
            "description": "Question about the image",
            "data_type": "str",
            "default_value": "Describe what this image shows in one short sentence.",
        },
        "schema": {"description": "Optional JSON Schema for structured output", "default_value": ""},
        "image_mime": {"description": "MIME type of the image bytes", "default_value": "image/jpeg"},
        "send_images": {
            "description": "Allow sending images to the provider (turn off for live cameras without consent)",
            "data_type": "bool",
            "default_value": True,
        },
    }

    def __init__(self, name: str):
        super().__init__(name, "LLMDescribeImage")
        _in(self, "image", BYTES, self._on_image)
        self._init_llm()
        self._image: bytes | None = None

    def _on_image(self, values: list[PortValue]):
        self._image = values[0][-1].tobytes()

    def _build_request(self) -> Request | None:
        if not self.properties["send_images"]:
            self._deliver_error("sending images is turned off (send_images)")
            return None
        if self._image is None:
            self._deliver_error("no image received yet")
            return None
        schema = _parse_schema(self.properties["schema"])
        return Request(
            prompt=str(self.properties["prompt"]),
            api_key="",
            model="",
            image=self._image,
            image_mime=str(self.properties["image_mime"]),
            schema=schema,
        )

    def set_properties(self, properties: dict[str, PropertyDataType]):
        self._set_common(properties)
        for key in ("prompt", "schema", "image_mime"):
            if key in properties:
                self.properties[key] = str(properties[key])
        if "send_images" in properties:
            self.properties["send_images"] = bool(properties["send_images"])
        if self.parent_graph is not None:
            self.parent_graph.notify_node_updated(self)


class LLMInterpretNode(_LLMNode):
    """Ask an LLM to interpret numeric features (optionally with an image)."""

    @classmethod
    def describe(cls, properties=None):
        fields = _schema_fields(_parse_schema((properties or {}).get("schema", "")))
        outputs = {
            "text": "The model's reply",
            "json": "Reply parsed as JSON",
            "busy": "True while a request is running",
            "error": "Last error (missing key, rate limit, daily cap, ...)",
            "usage": "Token usage of the last request, as JSON",
        }
        outputs.update({f"out_{k}": f"Schema field '{k}'" for k in fields})
        return NodeDoc(
            description=(
                "Sends named feature values (and an optional context text and "
                "image) to a cloud LLM for interpretation. Each top-level "
                "number/string/boolean field of 'schema' becomes an output "
                "port out_<field>." + _PROVIDER_NOTE
            ),
            category="AI",
            inputs={
                "features": "Feature values (any shape; flattened and named by 'labels')",
                "context": "Extra context text for the prompt",
                "image": "Optional encoded image bytes u8[?]",
                "trigger": "Any value: send a request now (subject to pacing)",
            },
            outputs=outputs,
            env=_env_docs(properties),
        )

    PROPERTIES = {
        **COMMON_PROPERTIES,
        "prompt": {
            "description": "Instruction; the named features and context are appended",
            "data_type": "str",
            "default_value": "Interpret these measurements.",
        },
        "labels": {"description": "Comma-separated names for the feature values", "default_value": ""},
        "schema": {"description": "JSON Schema of the reply (object)", "default_value": ""},
        "send_images": {
            "description": "Include the latest image in requests",
            "data_type": "bool",
            "default_value": False,
        },
    }

    def __init__(self, name: str):
        super().__init__(name, "LLMInterpret")
        _in(self, "features", ANY, self._on_features)
        _in(self, "context", STR, self._on_context)
        _in(self, "image", BYTES, self._on_image)
        self._init_llm()
        self._features: np.ndarray | None = None
        self._context = ""
        self._image: bytes | None = None
        self._field_ports: dict[str, str] = {}  # field -> port name
        self._apply_schema_ports()

    def _on_features(self, values: list[PortValue]):
        x = np.asarray(values[0])
        self._features = (x[-1] if x.ndim > 1 else x).flatten()  # a copy: kept past the dispatch

    def _on_context(self, values: list[PortValue]):
        self._context = str(values[0][-1])

    def _on_image(self, values: list[PortValue]):
        self._image = values[0][-1].tobytes()

    def _feature_text(self) -> str:
        if self._features is None:
            return ""
        labels = [s.strip() for s in str(self.properties["labels"]).split(",") if s.strip()]
        lines = []
        for i, value in enumerate(self._features):
            name = labels[i] if i < len(labels) else f"value_{i}"
            try:
                lines.append(f"- {name}: {float(value):.4g}")
            except (TypeError, ValueError):
                lines.append(f"- {name}: {value}")
        return "\n".join(lines)

    def _build_request(self) -> Request | None:
        if self._features is None and not self._context:
            self._deliver_error("no features or context received yet")
            return None
        parts = [str(self.properties["prompt"])]
        features = self._feature_text()
        if features:
            parts.append("Measurements:\n" + features)
        if self._context:
            parts.append("Context: " + self._context)
        image = self._image if self.properties["send_images"] else None
        return Request(
            prompt="\n\n".join(parts),
            api_key="",
            model="",
            image=image,
            schema=_parse_schema(self.properties["schema"]),
        )

    def _dispatch_fields(self, data):
        if not isinstance(data, dict):
            return
        for field, port_name in self._field_ports.items():
            if field not in data:
                continue
            value = data[field]
            vtype = self.outputs[port_name].value_type_list[0]
            try:
                if vtype.data_type == DataType.Float32:
                    arr = np.array([float(value)], dtype=np.float32)
                elif vtype.data_type == DataType.Bool:
                    arr = np.array([bool(value)])
                else:
                    arr = np.array([str(value)])
            except (TypeError, ValueError):
                continue
            self.outputs[port_name].dispatch([arr])

    def _apply_schema_ports(self):
        fields = _schema_fields(_parse_schema(self.properties["schema"]))
        wanted = {f"out_{k}": (k, _SCHEMA_PORT_TYPES[t]) for k, t in fields.items()}
        for field, port_name in list(self._field_ports.items()):
            if port_name not in wanted or wanted[port_name][1] != self.outputs[port_name].value_type_list[0]:
                if self.parent_graph is not None:
                    self.parent_graph.disconnect_all_from_node_output(self.name, port_name)
                del self.outputs[port_name]
                del self._field_ports[field]
        for port_name, (field, vtype) in wanted.items():
            if port_name not in self.outputs:
                _out(self, port_name, vtype)
            self._field_ports[field] = port_name

    def set_properties(self, properties: dict[str, PropertyDataType]):
        self._set_common(properties)
        for key in ("prompt", "labels"):
            if key in properties:
                self.properties[key] = str(properties[key])
        if "send_images" in properties:
            self.properties["send_images"] = bool(properties["send_images"])
        if "schema" in properties:
            _parse_schema(str(properties["schema"]), strict=True)
            self.properties["schema"] = str(properties["schema"])
            self._apply_schema_ports()
        if self.parent_graph is not None:
            self.parent_graph.notify_node_updated(self)


def _parse_schema(text, strict: bool = False) -> dict | None:
    text = str(text or "").strip()
    if not text:
        return None
    try:
        schema = json.loads(text)
    except ValueError as e:
        if strict:
            raise ValueError(f"schema is not valid JSON: {e}") from e
        return None
    if not isinstance(schema, dict):
        if strict:
            raise ValueError("schema must be a JSON object")
        return None
    return schema


def _schema_fields(schema: dict | None) -> dict[str, str]:
    """Top-level fields with a scalar type -> JSON type name."""
    if not schema:
        return {}
    fields = {}
    for name, spec in (schema.get("properties") or {}).items():
        kind = spec.get("type") if isinstance(spec, dict) else None
        if kind in _SCHEMA_PORT_TYPES and name.isidentifier():
            fields[name] = kind
    return fields
