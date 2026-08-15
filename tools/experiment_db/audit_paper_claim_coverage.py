#!/usr/bin/env python3
"""Check that every planned paper claim is covered by canonical routes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
CLAIMS = ROOT / "experiments/manifests/paper_claim_manifest.yaml"
ROUTES = ROOT / "experiments/manifests/paper_experiment_route_matrix.yaml"

READY_STATUSES = {
    "COMPLETED_VERIFIED",
    "COMPLETED_FIXED_CHECKPOINT",
    "COMPLETED_POINT_ESTIMATE",
    "COMPLETED_EVIDENCE_ONLY",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission-ready", action="store_true")
    args = parser.parse_args()
    claims = yaml.safe_load(CLAIMS.read_text())
    route_payload = yaml.safe_load(ROUTES.read_text())
    routes = {row["ledger_id"]: row for row in route_payload["experiments"]}
    historical = set(claims.get("historical_only_routes", []))
    errors: list[str] = []
    blocked: list[dict] = []
    covered: set[str] = set()

    if claims.get("authoritative") is not True:
        errors.append("claim manifest is not authoritative")
    for claim in claims["claims"]:
        for route_id in claim["required_routes"]:
            covered.add(route_id)
            route = routes.get(route_id)
            if route is None:
                errors.append(f"{claim['claim_id']}: missing route {route_id}")
                continue
            if claim.get("required_for_submission") and route_id in historical:
                errors.append(
                    f"{claim['claim_id']}: historical route used for a canonical claim: {route_id}"
                )
            if (
                args.submission_ready
                and claim.get("required_for_submission")
                and route["status"] not in READY_STATUSES
            ):
                blocked.append(
                    {
                        "claim_id": claim["claim_id"],
                        "route_id": route_id,
                        "status": route["status"],
                    }
                )

    for route_id in historical:
        if route_id not in routes:
            errors.append(f"historical route missing from route matrix: {route_id}")
    result = {
        "status": "FAIL" if errors or (args.submission_ready and blocked) else "PASS",
        "claims": len(claims["claims"]),
        "covered_routes": len(covered),
        "route_groups": len(routes),
        "submission_readiness_checked": args.submission_ready,
        "submission_ready": (
            not blocked and not errors if args.submission_ready else None
        ),
        "blocked_count": len(blocked),
        "blocked": blocked,
        "errors": errors,
    }
    print(json.dumps(result, indent=2))
    return int(bool(errors or (args.submission_ready and blocked)))


if __name__ == "__main__":
    raise SystemExit(main())
