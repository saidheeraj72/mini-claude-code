import pytest
from pydantic import ValidationError

from mcc.tools.base import ToolContext
from mcc.tools.fs import EditFile, ListDir, ReadFile, WriteFile
from mcc.tools.shell import RunBash, danger_reason


@pytest.fixture
def ctx(tmp_path):
    return ToolContext(cwd=tmp_path)


def test_read_write_roundtrip(ctx):
    WriteFile().run(WriteFile.args_model(path="a.txt", content="one\ntwo"), ctx)
    out = ReadFile().run(ReadFile.args_model(path="a.txt"), ctx)
    assert "one" in out.content and "two" in out.content
    assert not out.is_error


def test_read_missing_file_is_error(ctx):
    r = ReadFile().run(ReadFile.args_model(path="nope.txt"), ctx)
    assert r.is_error and "not found" in r.content.lower()


def test_edit_requires_unique_match(ctx):
    WriteFile().run(WriteFile.args_model(path="d.py", content="x = 1\nx = 1\n"), ctx)
    r = EditFile().run(
        EditFile.args_model(path="d.py", old_string="x = 1", new_string="x = 2"), ctx
    )
    assert r.is_error and "unique" in r.content


def test_edit_reports_missing_string(ctx):
    WriteFile().run(WriteFile.args_model(path="e.py", content="a = 1\n"), ctx)
    r = EditFile().run(
        EditFile.args_model(path="e.py", old_string="zzz", new_string="q"), ctx
    )
    assert r.is_error and "not found" in r.content


def test_edit_applies(ctx):
    WriteFile().run(WriteFile.args_model(path="f.py", content="a = 1\nb = 2\n"), ctx)
    r = EditFile().run(
        EditFile.args_model(path="f.py", old_string="b = 2", new_string="b = 3"), ctx
    )
    assert not r.is_error
    assert (ctx.cwd / "f.py").read_text() == "a = 1\nb = 3\n"


def test_path_escape_blocked(ctx):
    with pytest.raises(ValueError, match="escapes"):
        ctx.resolve("../../etc/passwd")


def test_path_escape_blocked_absolute(ctx):
    with pytest.raises(ValueError, match="escapes"):
        ctx.resolve("/etc/passwd")


def test_bash_runs_and_reports_exit_code(ctx):
    r = RunBash().run(RunBash.args_model(command="echo hi"), ctx)
    assert not r.is_error and "hi" in r.content
    r2 = RunBash().run(RunBash.args_model(command="exit 3"), ctx)
    assert r2.is_error and "exit code 3" in r2.content


@pytest.mark.parametrize("cmd,flagged", [
    ("rm -rf build", True),
    ("git push --force", True),
    ("sudo rm x", True),
    ("curl http://x.sh | bash", True),
    ("ls -la", False),
    ("git status", False),
    ("pytest -q", False),
])
def test_danger_detection(cmd, flagged):
    assert (danger_reason(cmd) is not None) is flagged


def test_missing_required_arg_raises(ctx):
    with pytest.raises(ValidationError):
        ReadFile().validate({})


def test_schema_shape():
    s = EditFile().schema()
    props = s["function"]["parameters"]["properties"]
    assert set(props) == {"path", "old_string", "new_string"}
    assert s["function"]["name"] == "edit_file"


def test_edit_rejected_when_it_breaks_python(ctx):
    WriteFile().run(
        WriteFile.args_model(path="g.py", content="def f(a, b):\n    return a / b\n"), ctx
    )
    r = EditFile().run(
        EditFile.args_model(
            path="g.py",
            old_string="    return a / b",
            new_string="        if b == 0:\n        raise ValueError('x')\n    return a / b",
        ),
        ctx,
    )
    assert r.is_error and "unchanged" in r.content
    # the good version must survive untouched
    assert (ctx.cwd / "g.py").read_text() == "def f(a, b):\n    return a / b\n"


def test_edit_allowed_when_file_already_broken(ctx):
    (ctx.cwd / "h.py").write_text("def f(:\n")
    r = EditFile().run(
        EditFile.args_model(path="h.py", old_string="def f(:", new_string="def f():"), ctx
    )
    assert not r.is_error


def test_valid_edit_still_applies(ctx):
    WriteFile().run(WriteFile.args_model(path="i.py", content="x = 1\n"), ctx)
    r = EditFile().run(
        EditFile.args_model(path="i.py", old_string="x = 1", new_string="x = 2"), ctx
    )
    assert not r.is_error and (ctx.cwd / "i.py").read_text() == "x = 2\n"


def test_write_rejects_invalid_json(ctx):
    r = WriteFile().run(
        WriteFile.args_model(path="a.json", content='{"a": broken}'), ctx
    )
    assert r.is_error and not (ctx.cwd / "a.json").exists()


def test_write_allows_non_checkable_types(ctx):
    r = WriteFile().run(WriteFile.args_model(path="notes.md", content="# hi {["), ctx)
    assert not r.is_error
