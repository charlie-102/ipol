"""Validation status tracking for IPOL methods."""
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Validation status file location
STATUS_FILE = Path(__file__).parent.parent / "validation_status.json"


def load_status() -> Dict:
    """Load validation status from file."""
    if STATUS_FILE.exists():
        try:
            status = json.loads(STATUS_FILE.read_text())
            # Ensure all keys exist
            if "skipped_cuda" not in status:
                status["skipped_cuda"] = {}
            return status
        except json.JSONDecodeError:
            return {"passed": {}, "failed": {}, "skipped_cuda": {}}
    return {"passed": {}, "failed": {}, "skipped_cuda": {}}


def save_status(status: Dict) -> None:
    """Save validation status to file."""
    STATUS_FILE.write_text(json.dumps(status, indent=2))


def mark_passed(method_name: str, output_info: str = "") -> None:
    """Mark a method as passing validation."""
    status = load_status()

    # Remove from failed if present
    if method_name in status["failed"]:
        del status["failed"][method_name]

    status["passed"][method_name] = {
        "timestamp": datetime.now().isoformat(),
        "output": output_info
    }
    save_status(status)


def mark_failed(method_name: str, error: str, notes: str = "") -> None:
    """Mark a method as failing validation with error details."""
    status = load_status()

    # Remove from passed if present
    if method_name in status["passed"]:
        del status["passed"][method_name]

    status["failed"][method_name] = {
        "timestamp": datetime.now().isoformat(),
        "error": error,
        "notes": notes
    }
    save_status(status)


def mark_skipped_cuda(method_name: str) -> None:
    """Mark a method as skipped due to CUDA requirement."""
    status = load_status()

    # Remove from failed if present (it's not a real failure)
    if method_name in status["failed"]:
        del status["failed"][method_name]

    status["skipped_cuda"][method_name] = {
        "timestamp": datetime.now().isoformat(),
        "reason": "Requires NVIDIA CUDA GPU"
    }
    save_status(status)


def mark_skipped_docker(method_name: str) -> None:
    """Mark a method as skipped due to Docker requirement."""
    status = load_status()
    if "skipped_docker" not in status:
        status["skipped_docker"] = {}

    # Remove from failed if present (it's not a real failure)
    if method_name in status["failed"]:
        del status["failed"][method_name]

    status["skipped_docker"][method_name] = {
        "timestamp": datetime.now().isoformat(),
        "reason": "Requires IPOL Docker infrastructure"
    }
    save_status(status)


def get_passed_methods() -> List[str]:
    """Get list of methods that passed validation."""
    status = load_status()
    return list(status.get("passed", {}).keys())


def get_failed_methods() -> Dict[str, Dict]:
    """Get dict of methods that failed validation with their error info."""
    status = load_status()
    return status.get("failed", {})


def is_validated(method_name: str) -> bool:
    """Check if a method has passed validation."""
    status = load_status()
    return method_name in status.get("passed", {})


def get_validation_summary() -> Dict:
    """Get summary of validation status.

    Returns:
        Dict with passed, failed, skipped_cuda, skipped_docker
    """
    status = load_status()
    return {
        "passed": status.get("passed", {}),
        "failed": status.get("failed", {}),
        "skipped_cuda": status.get("skipped_cuda", {}),
        "skipped_docker": status.get("skipped_docker", {}),
    }


def print_validation_report() -> None:
    """Print a formatted validation report."""
    summary = get_validation_summary()
    passed = summary["passed"]
    failed = summary["failed"]
    cuda = summary["skipped_cuda"]
    docker = summary["skipped_docker"]

    print("\n" + "=" * 50)
    print("IPOL Method Validation Report")
    print("=" * 50)

    print(f"\nPASSED ({len(passed)}):")
    if passed:
        for name in sorted(passed.keys()):
            print(f"  ✓ {name}")
    else:
        print("  (none)")

    print(f"\nSKIPPED - CUDA REQUIRED ({len(cuda)}):")
    if cuda:
        for name in sorted(cuda.keys()):
            print(f"  ⊘ {name} (requires NVIDIA GPU)")
    else:
        print("  (none)")

    print(f"\nSKIPPED - DOCKER REQUIRED ({len(docker)}):")
    if docker:
        for name in sorted(docker.keys()):
            print(f"  ⊘ {name} (requires IPOL Docker)")
    else:
        print("  (none)")

    print(f"\nFAILED ({len(failed)}):")
    if failed:
        for name, info in sorted(failed.items()):
            print(f"  ✗ {name}")
            print(f"      Error: {info.get('error', 'Unknown')}")
            if info.get('notes'):
                print(f"      Notes: {info['notes']}")
    else:
        print("  (none)")

    print("\n" + "=" * 50)
