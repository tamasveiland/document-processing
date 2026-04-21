#!/usr/bin/env python3
"""
One-time setup: configure model deployment defaults for the Content
Understanding resource so that prebuilt analyzers (e.g. prebuilt-documentSearch)
can resolve the required language and embedding models.

This script must be run once per Content Understanding resource before using
any analyzer that requires language or embedding models.

Architecture note
-----------------
    Model deployments (GPT, embedding) live in the AI Foundry account.
    The update_defaults call must therefore be routed through the AI Foundry
    project endpoint (AZURE_AI_PROJECT_ENDPOINT), not through the standalone
    Content Understanding endpoint — which has no deployments of its own.

Usage
-----
    # List available deployments in your Azure AI Foundry account:
    python setup_model_defaults.py --list-deployments

    # Configure defaults using the deployment names shown by --list-deployments:
    python setup_model_defaults.py --completion-model <deployment-name> --embedding-model <deployment-name>

    # Example:
    python setup_model_defaults.py --completion-model gpt-5.2 --embedding-model text-embedding-3-large

Auth
----
    Uses DefaultAzureCredential (az login / managed identity — no keys in code).
    Reads AZURE_AI_PROJECT_ENDPOINT and AZURE_OPENAI_ENDPOINT from .env.
"""

from __future__ import annotations

import argparse
import os
import sys

from azure.ai.contentunderstanding import ContentUnderstandingClient
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

# Logical role names expected by Content Understanding — do not change these keys.
_COMPLETION_ROLE = "gpt-5.4-nano"        # logical role key, not the deployment name
_EMBEDDING_ROLE  = "text-embedding-3-large"  # logical role key, not the deployment name


def _list_deployments(openai_endpoint: str, credential: DefaultAzureCredential) -> None:
    """Print all model deployments available in the Azure OpenAI resource."""
    try:
        from azure.mgmt.cognitiveservices import CognitiveServicesManagementClient  # type: ignore
    except ImportError:
        print(
            "azure-mgmt-cognitiveservices is not installed.\n"
            "Run:  pip install azure-mgmt-cognitiveservices\n\n"
            "Alternatively, view deployments in the Azure portal or AI Foundry portal.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Derive subscription and resource from the endpoint / env vars
    subscription_id = os.environ.get("AZURE_SUBSCRIPTION_ID", "")
    resource_group  = os.environ.get("AZURE_RESOURCE_GROUP", "")
    account_name    = os.environ.get("AZURE_AI_ACCOUNT_NAME", "")

    if not all([subscription_id, resource_group, account_name]):
        print(
            "To list deployments, set AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP, "
            "and AZURE_AI_ACCOUNT_NAME in your .env file.",
            file=sys.stderr,
        )
        sys.exit(1)

    mgmt = CognitiveServicesManagementClient(credential, subscription_id)
    print(f"\nDeployments in account '{account_name}' ({resource_group}):\n")
    found = False
    for dep in mgmt.deployments.list(resource_group, account_name):
        model = dep.properties.model if dep.properties else None
        model_info = f"  model : {model.name} {model.version}" if model else ""
        print(f"  name  : {dep.name}{model_info}")
        print()
        found = True
    if not found:
        print("  (no deployments found)")
    print(
        "Use one of these names as --completion-model / --embedding-model "
        "when running this script without --list-deployments."
    )


def _configure_defaults(
    ai_account_endpoint: str,
    credential: DefaultAzureCredential,
    completion_deployment: str,
    embedding_deployment: str,
) -> None:
    """Push model deployment defaults to the AI account's Content Understanding endpoint."""
    model_deployments = {
        _COMPLETION_ROLE: completion_deployment,
        _EMBEDDING_ROLE:  embedding_deployment,
    }

    # update_defaults must be called against the AI account's cognitiveservices
    # endpoint (where deployments live), not the project endpoint or the standalone
    # CU resource endpoint.  Standard Cognitive Services auth applies here.
    client = ContentUnderstandingClient(
        endpoint=ai_account_endpoint,
        credential=credential,
    )
    client.update_defaults({"modelDeployments": model_deployments})

    print("\nModel defaults configured successfully:")
    for role, deployment in model_deployments.items():
        print(f"  {role:35s} -> {deployment}")
    print(
        "\nYou can now run process_document.py. "
        "This setup only needs to be done once per resource."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Configure model deployment defaults for Azure Content Understanding.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--list-deployments",
        action="store_true",
        help="List available deployments in the Azure OpenAI / AI Foundry resource and exit.",
    )
    parser.add_argument(
        "--completion-model",
        default=None,
        metavar="DEPLOYMENT_NAME",
        help="Exact deployment name to use as the completion (GPT) model.",
    )
    parser.add_argument(
        "--embedding-model",
        default="text-embedding-3-large",
        metavar="DEPLOYMENT_NAME",
        help="Exact deployment name to use as the embedding model "
             "(default: text-embedding-3-large).",
    )
    args = parser.parse_args()

    load_dotenv()
    credential = DefaultAzureCredential()

    # update_defaults must target the AI account that owns the GPT/embedding
    # deployments.  Construct the cognitiveservices endpoint from the account name.
    ai_account_name = os.environ.get("AZURE_AI_ACCOUNT_NAME", "")
    if not ai_account_name:
        print(
            "Error: AZURE_AI_ACCOUNT_NAME is not set.\n"
            "Copy .env.sample to .env and fill in your values.",
            file=sys.stderr,
        )
        sys.exit(1)
    ai_account_endpoint = f"https://{ai_account_name}.cognitiveservices.azure.com/"

    openai_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "")

    if args.list_deployments:
        _list_deployments(openai_endpoint, credential)
        return

    if not args.completion_model:
        parser.error(
            "--completion-model is required.\n"
            "Run with --list-deployments first to see available deployment names."
        )

    print(f"AI account endpoint : {ai_account_endpoint}")
    print("Configuring model deployment defaults ...")
    _configure_defaults(ai_account_endpoint, credential, args.completion_model, args.embedding_model)


if __name__ == "__main__":
    main()
