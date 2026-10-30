import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[2]/"deploy/gcp"))
from observe import observe


def test_no_release_is_degraded_without_raw_exception(tmp_path):
    result=observe(tmp_path/"missing-config",tmp_path/"missing-nginx",tmp_path/"control")
    assert result["state"]=="degraded"
    assert set(result)=={"component","state","tlsExpiring","latencyMs","timestamp"}


def test_drain_observation_never_reads_secret_or_calls_network(tmp_path):
    (tmp_path/"drain").touch()
    result=observe(tmp_path/"missing-config",tmp_path/"missing-nginx",tmp_path)
    assert result["state"]=="maintenance" and result["tlsExpiring"] is None
