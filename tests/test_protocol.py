from mcc.llm.ollama import extract_inline_calls


def test_extracts_name_arguments_form():
    calls, text = extract_inline_calls(
        'Reading it.\n```json\n{"name":"read_file","arguments":{"path":"a.py"}}\n```'
    )
    assert len(calls) == 1
    assert calls[0].name == "read_file"
    assert calls[0].args == {"path": "a.py"}
    assert "```" not in text


def test_extracts_tool_args_variant():
    calls, _ = extract_inline_calls('```\n{"tool":"grep","args":{"pattern":"def"}}\n```')
    assert calls[0].name == "grep" and calls[0].args["pattern"] == "def"


def test_flat_args_fallback():
    calls, _ = extract_inline_calls('```json\n{"name":"list_dir","path":"src"}\n```')
    assert calls[0].args == {"path": "src"}


def test_plain_prose_yields_nothing():
    calls, text = extract_inline_calls("I think the bug is in main.py")
    assert calls == [] and "main.py" in text


def test_malformed_json_ignored():
    calls, _ = extract_inline_calls('```json\n{"name": broken\n```')
    assert calls == []
