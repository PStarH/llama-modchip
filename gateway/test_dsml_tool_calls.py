"""Regression tests for DeepSeek DSML tool calls.

The compact-tools prompt in sse_gateway.py (the `MODCHIP_COMPACT_TOOLS` / dsv4 tools injection) tells the model to
invoke tools as
`<｜DSML｜tool_calls><｜DSML｜invoke name="TOOL_NAME"><｜DSML｜parameter name="PARAM" string="true|false">VAL</parameter></invoke></tool_calls>`
(opening tags carry the U+FF5C DSML bars, closing tags are plain), but the normalizer and try_parse_tool_call only
knew `<tool_call>`, `<function=>` and JSON. The calls were streamed to the client as assistant text.

Run with: python -m pytest -q gateway/test_dsml_tool_calls.py
"""

import importlib.util
import json
import os
import unittest
from pathlib import Path

GATEWAY_PATH = Path(__file__).resolve().parent / "sse_gateway.py"


def _load_gateway():
    # Keep the import side-effect free: the module opens /dev/shm when HARP_CACHE is enabled, which a unit test
    # must not do. Restore the caller's environment.
    previous = os.environ.get("HARP_CACHE")
    os.environ["HARP_CACHE"] = "0"
    try:
        spec = importlib.util.spec_from_file_location("sse_gateway_dsml_under_test", GATEWAY_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        if previous is None:
            os.environ.pop("HARP_CACHE", None)
        else:
            os.environ["HARP_CACHE"] = previous
    return module


gw = _load_gateway()

DSML = "｜DSML｜"
DSML_OPEN = f"<{DSML}tool_calls>"
DSML_CLOSE = "</tool_calls>"
SCHEMAS = {
    "Read": {"file_path": {"type": "string"}, "offset": {"type": "integer"}},
    "Bash": {"command": {"type": "string"}, "run_in_background": {"type": "boolean"}},
}


def _invoke(name, params):
    parts = []
    for key, value in params:
        string_attr = "true" if isinstance(value, str) else "false"
        parts.append(f'<{DSML}parameter name="{key}" string="{string_attr}">{value}</parameter>')
    return f'<{DSML}invoke name="{name}">' + "".join(parts) + "</invoke>"


def _block(*invokes):
    return DSML_OPEN + "".join(invokes) + DSML_CLOSE


def _event(type_, **payload):
    payload["type"] = type_
    return f"event: {type_}\ndata: {json.dumps(payload)}\n\n"


def _stream(parts):
    out = [
        _event(
            "message_start",
            message={
                "id": "msg",
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": "test",
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        ),
        _event("content_block_start", index=0, content_block={"type": "text", "text": ""}),
    ]
    for part in parts:
        out.append(_event("content_block_delta", index=0, delta={"type": "text_delta", "text": part}))
    out.append(_event("content_block_stop", index=0))
    out.append(
        _event(
            "message_delta",
            delta={"stop_reason": "end_turn", "stop_sequence": None},
            usage={"output_tokens": 1},
        )
    )
    out.append(_event("message_stop"))
    return "".join(out)


def _events(rendered):
    events = []
    for block in rendered.split("\n\n"):
        for line in block.split("\n"):
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
    return events


class ParserTest(unittest.TestCase):
    def test_the_compact_tools_prompt_advertises_the_syntax_the_parser_reads(self):
        source = GATEWAY_PATH.read_text(encoding="utf-8")
        self.assertIn(f"invoke via {DSML_OPEN}", source)
        self.assertIn(f'<{DSML}invoke name=', source)
        self.assertIn(f'<{DSML}parameter name=', source)

    def test_reads_the_advertised_format(self):
        text = _block(_invoke("Read", [("file_path", "/etc/hosts"), ("offset", 3)]))
        self.assertEqual(
            gw.try_parse_tool_call(text, SCHEMAS),
            {"name": "Read", "input": {"file_path": "/etc/hosts", "offset": 3}},
        )

    def test_parameter_types_follow_the_string_attribute(self):
        # string="false" is JSON (the schema types it too); string="true" stays a string even when a schema would
        # otherwise infer a boolean.
        text = _block(_invoke("Bash", [("command", "true"), ("run_in_background", False)]))
        self.assertEqual(
            gw.try_parse_tool_call(text, SCHEMAS),
            {"name": "Bash", "input": {"command": "true", "run_in_background": False}},
        )

    def test_unknown_tool_is_rejected_when_schemas_are_given(self):
        text = _block(_invoke("Nope", [("x", "1")]))
        self.assertIsNone(gw.try_parse_tool_call(text, SCHEMAS))

    def test_plain_text_is_not_a_tool_call(self):
        self.assertIsNone(gw.try_parse_tool_call("just an answer", SCHEMAS))

    def test_non_streaming_text_becomes_tool_use_blocks(self):
        block = _block(
            _invoke("Read", [("file_path", "/a")]),
            _invoke("Bash", [("command", "ls")]),
        )
        resp = gw.sanitize_non_streaming_response({"content": [{"type": "text", "text": block}]}, SCHEMAS)
        uses = [b for b in resp["content"] if b.get("type") == "tool_use"]
        self.assertEqual(
            [(u["name"], u["input"]) for u in uses],
            [("Read", {"file_path": "/a"}), ("Bash", {"command": "ls"})],
        )
        self.assertEqual(resp["stop_reason"], "tool_use")


class DsmlStreamingTest(unittest.TestCase):
    def _run(self, parts):
        stream_filter = gw.ThinkBlockFilter(mode="med", schemas=SCHEMAS)
        rendered = stream_filter.feed(_stream(parts)) + stream_filter.flush()
        return _events(rendered)

    def _tool_uses(self, events):
        starts = [
            e
            for e in events
            if e["type"] == "content_block_start" and e.get("content_block", {}).get("type") == "tool_use"
        ]
        deltas = [
            e
            for e in events
            if e["type"] == "content_block_delta" and e.get("delta", {}).get("type") == "input_json_delta"
        ]
        self.assertEqual(len(starts), len(deltas), events)
        return [
            (start["content_block"]["name"], json.loads(delta["delta"]["partial_json"]))
            for start, delta in zip(starts, deltas)
        ]

    def _assert_single_read(self, events):
        self.assertEqual(
            self._tool_uses(events),
            [("Read", {"file_path": "/etc/hosts"})],
            events,
        )
        message_delta = next(e for e in events if e["type"] == "message_delta")
        self.assertEqual(message_delta["delta"]["stop_reason"], "tool_use")

    def test_dsml_block_in_one_delta(self):
        self._assert_single_read(self._run([_block(_invoke("Read", [("file_path", "/etc/hosts")]))]))

    def test_dsml_block_split_at_every_position(self):
        text = _block(_invoke("Read", [("file_path", "/etc/hosts")]))
        for split in range(1, len(text)):
            with self.subTest(split=split):
                self._assert_single_read(self._run([text[:split], text[split:]]))

    def test_text_before_and_after_the_block_is_kept(self):
        text = "Checking.\n" + _block(_invoke("Read", [("file_path", "/etc/hosts")])) + "\nDone."
        for split in (1, 7, 19, len(text) - 1):
            with self.subTest(split=split):
                events = self._run([text[:split], text[split:]])
                self._assert_single_read(events)
                visible = "".join(
                    e["delta"]["text"]
                    for e in events
                    if e["type"] == "content_block_delta" and e.get("delta", {}).get("type") == "text_delta"
                )
                self.assertIn("Checking.", visible)
                self.assertIn("Done.", visible)
                self.assertNotIn(f"<{DSML}", visible)

    def test_two_invokes_emit_two_tool_uses(self):
        text = _block(
            _invoke("Read", [("file_path", "/a")]),
            _invoke("Bash", [("command", "ls"), ("run_in_background", False)]),
        )
        events = self._run([text])
        self.assertEqual(
            self._tool_uses(events),
            [("Read", {"file_path": "/a"}), ("Bash", {"command": "ls", "run_in_background": False})],
            events,
        )
        message_delta = next(e for e in events if e["type"] == "message_delta")
        self.assertEqual(message_delta["delta"]["stop_reason"], "tool_use")


if __name__ == "__main__":
    unittest.main()
