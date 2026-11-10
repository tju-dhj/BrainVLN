#!/usr/bin/env python3
"""Read-only audit of collected R2R-CE RGB-D instruction/action episodes."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from spikingnav.vln.data.audit import audit_dataset, write_audit_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Root containing scene/episode directories")
    parser.add_argument("--max-episodes", type=int, default=20, help="0 audits every episode")
    parser.add_argument(
        "--r2r-dataset",
        default="storage/datasets/R2R_VLNCE_v1-3_preprocessed",
        help="Official dataset root, used only to verify stored train actions",
    )
    parser.add_argument("--json-out", help="Report path under storage/brain_vln/audits")
    args = parser.parse_args()
    if args.max_episodes < 0:
        parser.error("--max-episodes must be >= 0")
    dataset_root = Path(args.r2r_dataset)
    report = audit_dataset(
        Path(args.data),
        max_episodes=args.max_episodes,
        dataset_root=dataset_root if dataset_root.is_dir() else None,
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = Path(args.json_out) if args.json_out else Path("storage/brain_vln/audits") / f"audit_{stamp}.json"
    write_audit_report(report, output)
    print(
        "audit: scenes={scene_count} episode_dirs={episode_directory_count} ready={ready_episode_count} "
        "checked={audited_episode_count} passed={passed} failed={failed} "
        "official_train_gt={official_train_gt_available}".format(**report)
    )
    print(f"report: {output}")
    for result in report["results"]:
        for issue in result["errors"]:
            print(f"{issue['category']}: {result['episode']}: {issue['detail']}")
    return 1 if report["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
