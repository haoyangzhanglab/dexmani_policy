import shutil
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace

import hydra
import pytest
import zarr
from omegaconf import OmegaConf

from dexmani_policy.common import config as common_config
from dexmani_policy.common.checkpoint import Checkpoint, load_checkpoint, save_checkpoint
from dexmani_policy.training import resume, trainer as policy_training
from dexmani_policy.training.data import build_dataset


@pytest.mark.parametrize("path_kind", ["relative", "absolute", "tilde", "saved_zarr"])
def test_cli_and_api_choose_caller_files(tmp_path, small_dp3_config, make_demo, monkeypatch, path_kind):
    caller, project = tmp_path / "caller", tmp_path / "project"
    for directory, marker, step in ((caller, "caller", 1), (project, "project", 2)):
        directory.mkdir()
        shutil.copytree(make_demo(marker), directory / "data.zarr")
        zarr.open_group(str(directory / "data.zarr"), mode="a").attrs["marker"] = marker
        cfg = small_dp3_config(directory / "run", f"zarr_path={directory / 'data.zarr'}")
        save_checkpoint(directory / "run/checkpoints/latest.pt", Checkpoint(
            step, {"marker": marker}, {}, {}, OmegaConf.to_container(cfg, resolve=True),
        ))
    monkeypatch.setattr(common_config, "PROJECT_ROOT", project)
    monkeypatch.chdir(caller)
    if path_kind == "tilde":
        expanduser = Path.expanduser

        def controlled_home(path):
            return caller.joinpath(*path.parts[1:]) if path.parts[:1] == ("~",) else expanduser(path)

        monkeypatch.setattr(Path, "expanduser", controlled_home)
    run_arg = str(caller / "run") if path_kind == "absolute" else "~/run" if path_kind == "tilde" else "run"
    zarr_arg = str(caller / "data.zarr") if path_kind == "absolute" else "~/data.zarr" if path_kind == "tilde" else "data.zarr"
    if path_kind == "saved_zarr":
        zarr_arg = None

    constructed = []

    def inspect_build(cfg, *, checkpoint):
        dataset, _ = build_dataset(cfg)
        result = SimpleNamespace(
            directory=cfg.workspace.output_dir, data_path=dataset.episode_store.path,
            data_marker=dataset.episode_store.attrs["marker"],
            checkpoint_marker=checkpoint.model_state["marker"], step=checkpoint.global_step,
        )
        constructed.append(result)
        return result

    monkeypatch.setattr(policy_training, "build_trainer", inspect_build)
    monkeypatch.setattr(policy_training, "run_trainer", lambda result, **kwargs: result.step)
    api = resume.prepare_resume(run_arg, zarr_path=zarr_arg)
    assert Path.cwd() == caller
    prepare_resume = resume.prepare_resume

    def inspect_prepare(run, *, zarr_path, **kwargs):
        assert Path.cwd() == project
        assert run == str(caller / "run")
        assert zarr_path == (None if path_kind == "saved_zarr" else str(caller / "data.zarr"))
        return prepare_resume(run, zarr_path=zarr_path, **kwargs)

    monkeypatch.setattr(resume, "prepare_resume", inspect_prepare)
    args = ["--run", run_arg]
    if zarr_arg is not None:
        args += ["--zarr-path", zarr_arg]
    resume.main(args)
    cli = constructed[-1]
    assert cli == api
    assert cli.checkpoint_marker == cli.data_marker == "caller"
    assert cli.step == 1 and cli.directory == str(caller / "run")
    assert cli.data_path == str(caller / "data.zarr")
    assert Path.cwd() == project


@pytest.mark.parametrize("missing", ["run", "zarr"])
def test_missing_paths_fail_at_existing_io_boundary(tmp_path, small_dp3_checkpoint, monkeypatch, missing):
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setattr(common_config, "PROJECT_ROOT", project)
    monkeypatch.chdir(tmp_path)
    run_arg = "missing_run" if missing == "run" else "dp3_run"
    zarr_arg = "missing.zarr" if missing == "zarr" else "demo.zarr"
    prepare_resume = resume.prepare_resume
    entered = []

    def inspect_prepare(run, *, zarr_path, **kwargs):
        entered.append(True)
        assert Path.cwd() == project
        assert run == str(tmp_path / run_arg)
        assert zarr_path == str(tmp_path / zarr_arg)
        return prepare_resume(run, zarr_path=zarr_path, **kwargs)

    monkeypatch.setattr(resume, "prepare_resume", inspect_prepare)
    error_type = FileNotFoundError if missing == "run" else hydra.errors.InstantiationException
    with pytest.raises(error_type) as caught:
        resume.main(["--run", run_arg, "--zarr-path", zarr_arg])
    assert entered == [True]
    if missing == "run":
        assert Path(caught.value.filename) == tmp_path / "missing_run/checkpoints/latest.pt"
    else:
        assert isinstance(caught.value.__cause__, zarr.errors.GroupNotFoundError)


def test_cli_really_restores_and_continues_current_dp3(tmp_path, small_dp3_checkpoint, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setattr(common_config, "PROJECT_ROOT", project)
    monkeypatch.chdir(tmp_path)
    with closing(resume.prepare_resume("dp3_run", zarr_path="demo.zarr")) as restored:
        assert Path.cwd() == tmp_path
        assert restored.global_step == 1
        assert restored.logger.output_dir == tmp_path / "dp3_run"
        assert restored.train_loader.dataset.episode_store.path == str(tmp_path / "demo.zarr")
    run_trainer = policy_training.run_trainer
    steps = []

    def inspect_run(restored, **kwargs):
        assert Path.cwd() == project
        assert restored.global_step == 1
        assert restored.logger.output_dir == tmp_path / "dp3_run"
        assert restored.cfg.dataset.zarr_path == str(tmp_path / "demo.zarr")
        steps.append(run_trainer(restored, **kwargs))

    monkeypatch.setattr(policy_training, "run_trainer", inspect_run)
    resume.main(["--run", "dp3_run", "--zarr-path", "demo.zarr", "--max-updates", "1"])
    assert steps == [2]
    checkpoint = load_checkpoint(small_dp3_checkpoint)
    assert checkpoint.global_step == 2
    assert checkpoint.effective_config["workspace"]["output_dir"] == str(tmp_path / "dp3_run")
    assert checkpoint.effective_config["dataset"]["zarr_path"] == str(tmp_path / "demo.zarr")
