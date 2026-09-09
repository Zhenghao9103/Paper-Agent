from scripts.post_install_check import run_checks


def test_post_install_fails_when_any_required_probe_fails() -> None:
    results = run_checks(
        checks={"router": lambda: True, "chroma": lambda: False}
    )
    assert results.ok is False
    assert results.failed == ("chroma",)


def test_post_install_records_probe_exceptions_as_failures() -> None:
    def broken() -> bool:
        raise RuntimeError("not ready")

    results = run_checks(checks={"mineru": broken})
    assert results.ok is False
    assert results.failed == ("mineru",)
    assert results.details["mineru"] == "not ready"
