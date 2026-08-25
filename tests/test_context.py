from mcc.context import estimate_tokens, needs_compaction, split_for_compaction


def test_system_message_never_folded():
    msgs = [{"role": "system", "content": "s"}] + [
        {"role": "user", "content": f"m{i}"} for i in range(12)
    ]
    fold, keep = split_for_compaction(msgs, keep_recent=4)
    assert all(m["role"] != "system" for m in fold)
    assert keep[0]["role"] == "system"


def test_short_history_not_split():
    msgs = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
    fold, keep = split_for_compaction(msgs)
    assert fold == [] and keep == msgs


def test_split_does_not_orphan_tool_results():
    msgs = [{"role": "assistant", "content": "call"}] + [
        {"role": "tool", "content": f"r{i}"} for i in range(3)
    ] + [{"role": "user", "content": f"u{i}"} for i in range(6)]
    fold, keep = split_for_compaction(msgs, keep_recent=8)
    assert keep[0]["role"] != "tool"


def test_compaction_threshold():
    small = [{"role": "user", "content": "x"}]
    assert not needs_compaction(small, 16384, 0.7)
    big = [{"role": "user", "content": "x" * 100_000}]
    assert needs_compaction(big, 16384, 0.7)


def test_token_estimate_scales():
    assert estimate_tokens("a" * 360) > 90
