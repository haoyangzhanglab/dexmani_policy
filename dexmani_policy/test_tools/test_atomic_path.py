import logging
from pathlib import Path

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
@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_publish_failure_preserves_exception_and_traceback(tmp_path, monkeypatch, caplog, source, cleanup_fails):
    target = tmp_path / "result"
    target.write_text("old")
    primary = ValueError("writer failed") if source == "writer" else OSError("replace failed")
    secondary = PermissionError("cleanup failed")
    original_traceback = None
    calls = []
    unlink = Path.unlink

    def fail(*args):
        nonlocal original_traceback
        try:
            raise primary
        except BaseException:
            original_traceback = primary.__traceback__
            raise

    def cleanup(path, *args, **kwargs):
        if path == temporary:
            calls.append(path)
            if cleanup_fails:
                raise secondary
        return unlink(path, *args, **kwargs)

    if source == "replace":
        monkeypatch.setattr(io_util.os, "replace", fail)
    monkeypatch.setattr(Path, "unlink", cleanup)
    with caplog.at_level(logging.WARNING, logger=io_util.__name__):
        with pytest.raises(type(primary)) as caught:
            with io_util.atomic_path(target) as temporary:
                temporary.write_text("new")
                if source == "writer":
                    fail()
    assert caught.value is primary
    traceback = caught.value.__traceback__
    while traceback is not None and traceback is not original_traceback:
        traceback = traceback.tb_next
    assert traceback is original_traceback
    assert target.read_text() == "old"
    assert calls == [temporary]
    assert temporary.exists() == cleanup_fails
    if cleanup_fails:
        record, = [record for record in caplog.records if record.name == io_util.__name__]
        assert str(temporary) in record.getMessage()
        assert record.exc_info[1] is secondary
        assert "PermissionError: cleanup failed" in caplog.text
    else:
        assert not caplog.records


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
