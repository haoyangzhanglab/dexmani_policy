from pathlib import Path
from unittest.mock import Mock

import pytest

from dexmani_policy.common import io_util
from dexmani_policy.common.checkpoint import Checkpoint, load_checkpoint, save_checkpoint


def test_atomic_publish_does_not_unlink_after_replace(tmp_path, monkeypatch):
    target = tmp_path / "result"
    target.write_text("old")
    unlink = Path.unlink
    calls = []

    def track_unlink(path, *args, **kwargs):
        calls.append(path)
        return unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", track_unlink)
    with io_util.atomic_path(target) as temporary:
        assert temporary.parent == target.parent
        temporary.write_text("new")
    assert target.read_text() == "new"
    assert not temporary.exists()
    assert calls == []


@pytest.mark.parametrize("source", ["writer", "replace"])
def test_failed_atomic_publish_keeps_previous_file(tmp_path, monkeypatch, source):
    target = tmp_path / "result"
    target.write_text("old")

    def fail(*args):
        raise OSError("publish failed")

    if source == "replace":
        monkeypatch.setattr(io_util.os, "replace", fail)
    with pytest.raises(OSError):
        with io_util.atomic_path(target) as temporary:
            temporary.write_text("new")
            if source == "writer":
                fail()
    assert target.read_text() == "old"
    assert not temporary.exists()


def test_keyboard_interrupt_cleans_up(tmp_path):
    target = tmp_path / "result"
    target.write_text("old")
    interrupt = KeyboardInterrupt("stop writer")
    with pytest.raises(KeyboardInterrupt) as caught:
        with io_util.atomic_path(target) as temporary:
            raise interrupt
    assert caught.value is interrupt
    assert not temporary.exists()
    assert target.read_text() == "old"


def test_atomic_publish_preserves_writer_error_on_cleanup_failure(tmp_path, monkeypatch, caplog):
    target = tmp_path / "result"
    target.write_text("old")
    failure = ValueError("writer failed")
    with pytest.raises(ValueError) as caught, io_util.atomic_path(target):
        monkeypatch.setattr(Path, "unlink", Mock(side_effect=OSError("unlink failed")))
        raise failure
    assert caught.value is failure
    assert "unlink failed" in caplog.text
    assert target.read_text() == "old"


def test_checkpoint_and_latest_symlink_publish(tmp_path, monkeypatch):
    latest = tmp_path / "latest.pt"
    first = save_checkpoint(tmp_path / "step_1.pt", Checkpoint(1, {}, {}, {}, {}))
    latest.symlink_to(first.name)
    second = save_checkpoint(tmp_path / "step_2.pt", Checkpoint(2, {}, {}, {}, {}))
    unlink = Path.unlink
    calls = []

    def track_unlink(path, *args, **kwargs):
        calls.append(path)
        return unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", track_unlink)
    with io_util.atomic_path(latest) as temporary:
        temporary.unlink()
        temporary.symlink_to(second.name)
    assert calls == [temporary]
    assert latest.is_symlink() and latest.readlink() == Path(second.name)
    assert load_checkpoint(latest).global_step == 2
    assert first.exists() and second.exists()
    assert not temporary.exists() and not temporary.is_symlink()
