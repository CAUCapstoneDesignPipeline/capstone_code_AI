from pathlib import Path
import importlib.util
import pytest

spec = importlib.util.spec_from_file_location("publish_preflight", Path(__file__).parents[2] / "deploy/gcp/publish_preflight.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def identity():
    return {"GITHUB_REPOSITORY": "CAUCapstoneDesignPipeline/capstone_code_AI", "GITHUB_REPOSITORY_ID": "1397258740",
            "GITHUB_REPOSITORY_OWNER_ID": "324880339", "GITHUB_REF": "refs/heads/develop",
            "GITHUB_EVENT_NAME": "workflow_dispatch", "GITHUB_SHA": "a" * 40,
            "GITHUB_WORKFLOW_REF": "CAUCapstoneDesignPipeline/capstone_code_AI/.github/workflows/gcp-publish-image.yml@refs/heads/develop",
            "AR_REPOSITORY": "capstone-ai", "GCP_PROJECT_ID": "capsnote", "GCP_REGION": "asia-northeast3",
            "GCP_IMAGE_SERVICE_ACCOUNT": "capsnote-ai-publisher@capsnote.iam.gserviceaccount.com",
            "GCP_WIF_PROVIDER": "projects/169516793433/locations/global/workloadIdentityPools/github/providers/ai"}


def test_valid_develop_publisher():
    module.validate(identity())


@pytest.mark.parametrize("key,value", [("GITHUB_REPOSITORY", "fork/capstone_code_AI"),
    ("GITHUB_REPOSITORY_ID", "999"), ("GITHUB_REPOSITORY_OWNER_ID", "999"),
    ("GITHUB_REF", "refs/heads/main"), ("GITHUB_REF", "refs/pull/1/merge"),
    ("GITHUB_EVENT_NAME", "pull_request"), ("GITHUB_WORKFLOW_REF", "wrong/workflow.yml@refs/heads/develop"),
    ("AR_REPOSITORY", "capstone-be"), ("GITHUB_SHA", "develop"), ("GCP_WIF_PROVIDER", ""),
    ("GCP_REGION", "us-central1"), ("GCP_IMAGE_SERVICE_ACCOUNT", "user@example.com")])
def test_denies_identity_or_destination_change(key, value):
    env = identity()
    env[key] = value
    with pytest.raises(ValueError):
        module.validate(env)
