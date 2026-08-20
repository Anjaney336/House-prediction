from __future__ import annotations

import argparse
import json

from src.platform.persistence import purge_expired_leads
from src.platform.retraining import due_for_retraining
from src.platform.service import purge_expired_customer_datasets, train_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrain active PricePredict models whose cadence has elapsed.")
    parser.add_argument("--tenant", required=True)
    parser.add_argument("--cadence-days", type=int, default=90)
    parser.add_argument("--execute", action="store_true", help="Execute jobs; without this flag the command is a dry run.")
    args = parser.parse_args()
    due = due_for_retraining(args.tenant, args.cadence_days)
    results = []
    for item in due:
        results.append(train_dataset(item["dataset_id"], args.tenant, item["target"]) if args.execute else item)
    purged_leads = purge_expired_leads() if args.execute else 0
    purged_datasets = purge_expired_customer_datasets() if args.execute else 0
    print(json.dumps({"dry_run": not args.execute, "due": results, "expired_leads_purged": purged_leads, "expired_customer_datasets_purged": purged_datasets}, indent=2))


if __name__ == "__main__":
    main()
