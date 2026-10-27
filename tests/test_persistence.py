from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from handwriting.api import create_app
from handwriting.config import Settings
from handwriting.store import Store
from tests.conftest import upload_sheet, wait_job


def test_fonts_survive_restart_and_only_one_worker_can_claim_data_directory(tmp_path, sheet):
    settings = Settings(data_dir=tmp_path)
    with TestClient(create_app(settings)) as client:
        job = upload_sheet(client, sheet)
        assert job["status"] == "ready"
        with pytest.raises(RuntimeError, match="already in use"):
            with TestClient(create_app(settings)):
                pass
    with TestClient(create_app(settings)) as restarted:
        assert restarted.get(f"/api/fonts/{job['id']}").json()["status"] == "ready"
        assert restarted.get(job["artifacts"]["font.woff2"]).status_code == 200


def test_interrupted_generation_is_recovered_from_sqlite(tmp_path, sheet):
    settings = Settings(data_dir=tmp_path)
    with TestClient(create_app(settings)) as client:
        job = upload_sheet(client, sheet)
    store = Store(settings.database)
    store.edit(job["id"], sheet[1], "auto")
    store.queue_generation(job["id"])
    claimed = store.claim()
    assert claimed["id"] == job["id"]
    assert store.get(job["id"])["status"] == "processing"
    # This is the durable state left by a process exit during conversion.
    with TestClient(create_app(settings)) as restarted:
        recovered = wait_job(restarted, job["id"])
        assert recovered["status"] == "ready", recovered
        assert recovered["revision"] == 2
        assert restarted.get(recovered["artifacts"]["font.ttf"]).status_code == 200


def test_claims_are_atomic_and_busy_mappings_cannot_change(tmp_path):
    store = Store(tmp_path / "jobs.db")
    store.initialize()
    store.create("one", "Font", {}, 5)
    with ThreadPoolExecutor(max_workers=2) as pool:
        claims = list(pool.map(lambda _: store.claim(), range(2)))
    assert sum(job is not None for job in claims) == 1
    with pytest.raises(ValueError, match="current operation"):
        store.edit("one", [], "auto")


def test_polling_closes_connections_without_garbage_collection(tmp_path):
    import gc
    import os

    store = Store(tmp_path / "jobs.db")
    store.initialize()
    store.create("one", "Font", {}, 5)
    gc.collect()
    gc.disable()
    try:
        before = len(os.listdir("/dev/fd"))
        for _ in range(50):
            assert store.get("one")["name"] == "Font"
        assert len(os.listdir("/dev/fd")) <= before + 2
    finally:
        gc.enable()
        gc.collect()


def test_generation_admission_obeys_queue_capacity(tmp_path):
    store = Store(tmp_path / "jobs.db")
    store.initialize()
    store.create("review", "Font", {}, 5)
    store.finish("review", "review_required", "generate", {"glyphs": [{}]})
    for index in range(20):
        store.create(f"pending-{index}", "Another", {}, 30)
    with pytest.raises(ValueError, match="queue is full"):
        store.queue_generation("review")
    assert store.get("review")["status"] == "review_required"


@pytest.mark.parametrize("cleanup_blocked", [False, True])
def test_publication_failure_records_failed_job_without_partial_downloads(
    client, sheet, monkeypatch, caplog, cleanup_blocked
):
    from pathlib import Path

    rename = Path.rename

    def fail_publication(path, destination):
        if path.name.startswith(".build-"):
            raise PermissionError("simulated filesystem failure")
        return rename(path, destination)

    monkeypatch.setattr(Path, "rename", fail_publication)
    if cleanup_blocked:
        import shutil

        def deny_cleanup(path):
            raise PermissionError("simulated directory permission failure")

        monkeypatch.setattr(shutil, "rmtree", deny_cleanup)
    job = upload_sheet(client, sheet)
    assert job["status"] == "failed"
    assert job["artifacts"] == {}
    assert "Review the crops" in job["error"]
    assert client.get(f"/api/fonts/{job['id']}/versions/1/font.ttf").status_code == 409
    client.app.state.worker.stop()
    leftovers = list(client.app.state.store.path.parent.glob("*/.build-*"))
    assert bool(leftovers) == cleanup_blocked
    if cleanup_blocked:
        assert "Staging cleanup failed: PermissionError" in caplog.text
