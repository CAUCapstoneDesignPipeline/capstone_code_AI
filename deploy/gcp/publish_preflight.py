"""Fail closed before requesting a WIF token. IAM must independently enforce claims."""
import os
import re


def validate(env):
    exact = {"GITHUB_REPOSITORY": "CAUCapstoneDesignPipeline/capstone_code_AI",
             "GITHUB_REPOSITORY_ID": "1397258740", "GITHUB_REPOSITORY_OWNER_ID": "324880339",
             "GITHUB_REF": "refs/heads/develop", "GITHUB_EVENT_NAME": "workflow_dispatch",
             "GITHUB_WORKFLOW_REF": "CAUCapstoneDesignPipeline/capstone_code_AI/.github/workflows/gcp-publish-image.yml@refs/heads/develop",
             "AR_REPOSITORY": "capstone-ai"}
    for key, expected in exact.items():
        if env.get(key) != expected:
            raise ValueError(f"invalid {key}")
    patterns = {"GITHUB_SHA": r"[0-9a-f]{40}", "GCP_PROJECT_ID": r"[a-z][a-z0-9-]{4,28}[a-z0-9]",
                "GCP_REGION": r"asia-northeast3", "GCP_IMAGE_SERVICE_ACCOUNT": r"[a-z][a-z0-9-]+@" + re.escape(env.get("GCP_PROJECT_ID", "")) + r"\.iam\.gserviceaccount\.com",
                "GCP_WIF_PROVIDER": r"projects/[0-9]+/locations/global/workloadIdentityPools/[a-z0-9-]+/providers/[a-z0-9-]+"}
    for key, pattern in patterns.items():
        if not re.fullmatch(pattern, env.get(key, "")):
            raise ValueError(f"invalid {key}")


if __name__ == "__main__":
    validate(os.environ)
