from mcc.permissions import _generalise
from mcc.tools.fs import EditFile
from mcc.tools.shell import RunBash


def test_generalise_bash_keeps_subcommand():
    assert _generalise(RunBash(), {"command": "git status -sb"}) == "git status*"


def test_generalise_bash_single_word():
    assert _generalise(RunBash(), {"command": "pytest"}) == "pytest*"


def test_generalise_bash_flag_not_treated_as_subcommand():
    assert _generalise(RunBash(), {"command": "ls -la"}) == "ls*"


def test_generalise_path_to_extension():
    assert _generalise(EditFile(), {"path": "src/main.py"}) == "*.py"
