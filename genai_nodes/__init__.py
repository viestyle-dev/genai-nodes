"""genai-nodes: nodes that call generative AI models.

The nodes ask LLMs (Gemini, Claude, models on OpenRouter) to describe images
or interpret features. Requests start only from a trigger, are paced and are
capped per day.

``register_portable_nodes`` registers the same nodes as ``register_nodes``:
they only need HTTPS. On mobile, keep API keys off the device and point the
host at a relay server instead.
"""

from nonoha_core.node_registry import NodeRegistry


def register_portable_nodes(registry: NodeRegistry):
    from .nodes import LLMDescribeImageNode, LLMInterpretNode

    registry.register_node("LLMDescribeImage", LLMDescribeImageNode)
    registry.register_node("LLMInterpret", LLMInterpretNode)


def register_nodes(registry: NodeRegistry):
    """Plugin-host entry point."""
    register_portable_nodes(registry)
