"""Unit tests for safe object serialization in showcase inspector."""

import json
from dataclasses import dataclass
from enum import Enum
from examples.showcase.components.object_inspector import safe_serialize_json


class SampleEnum(Enum):
    ALPHA = "alpha"
    BETA = "beta"


@dataclass
class SampleDataclass:
    name: str
    value: int
    tags: set[str]


class CustomObject:
    def __init__(self, key: str):
        self.key = key
        self.sub = None


def test_safe_serialize_basic():
    data = {
        "string": "hello",
        "number": 123,
        "float": 45.67,
        "boolean": True,
        "null": None,
        "list": [1, 2, 3],
    }
    serialized = safe_serialize_json(data)
    loaded = json.loads(serialized)
    assert loaded["string"] == "hello"
    assert loaded["number"] == 123
    assert loaded["float"] == 45.67
    assert loaded["list"] == [1, 2, 3]


def test_safe_serialize_dataclass_and_enum():
    dc = SampleDataclass(name="test_dc", value=999, tags={"sec", "opt"})
    serialized = safe_serialize_json(dc)
    loaded = json.loads(serialized)
    assert loaded["name"] == "test_dc"
    assert loaded["value"] == 999
    assert isinstance(loaded["tags"], list)
    assert set(loaded["tags"]) == {"sec", "opt"}


def test_safe_serialize_circular_references():
    a = CustomObject("node_a")
    b = CustomObject("node_b")
    a.sub = b
    b.sub = a  # Circular reference

    # Must serialize without infinite loop or recursion error
    serialized = safe_serialize_json(a)
    assert "node_a" in serialized
    assert "node_b" in serialized
    loaded = json.loads(serialized)
    assert loaded["key"] == "node_a"
