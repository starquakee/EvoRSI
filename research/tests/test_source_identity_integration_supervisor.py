"""Real API-produced evidence must be accepted by the real staging verifier."""
import json
from pathlib import Path
from test_api_admission_gate import client, spies, _submit_payload, API_HEADERS
from test_dispatcher_worker_control import td

def test_inline_api_evidence_matches_staging_verifier(client, spies):
    connection, redis = spies
    response = client.post("/api/v1/jobs", json=_submit_payload(), headers=API_HEADERS)
    assert response.status_code == 200
    job = json.loads(redis.pushes[0][1])
    reason, detail = td.verify_staged_source_identity(job, Path(job["code_file_path"]))
    assert reason is None, detail
    assert job["source_gate"]["entrypoint_sha256"]
