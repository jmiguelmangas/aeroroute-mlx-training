import json
from pathlib import Path

from aeroroute_mlx_training.cli import main


def test_baseline_cli_writes_reproducible_report(
    tmp_path: Path, monkeypatch
) -> None:
    source = Path(__file__).parents[2] / "samples" / "prompt-only-baseline.json"
    output = tmp_path / "report.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "aeroroute-mlx-training",
            "evaluate-baseline",
            "--input",
            str(source),
            "--output",
            str(output),
        ],
    )

    main()

    report = json.loads(output.read_text())
    assert report["numeric_fidelity"] == 1.0


def test_preflight_cli_returns_nonzero_when_artifacts_are_missing(
    tmp_path: Path, monkeypatch
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({}))
    monkeypatch.setattr(
        "sys.argv",
        [
            "aeroroute-mlx-training",
            "preflight-smoke",
            "--manifest",
            str(manifest),
        ],
    )

    try:
        main()
    except SystemExit as error:
        assert error.code == 2
    else:
        raise AssertionError("preflight must reject absent artefacts")
