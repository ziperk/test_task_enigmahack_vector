import argparse
import csv
from pathlib import Path
from typing import NamedTuple


ROOT = Path(__file__).resolve().parent
REQUIRED_COLUMNS = {"id", "intent_id", "decision"}
DECISIONS = {"self", "clarify", "handoff"}


class Prediction(NamedTuple):
    intent_id: int
    decision: str
    reason: str


def read_submission(path: Path) -> dict[str, Prediction]:
    predictions = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, strict=True)
        columns = reader.fieldnames or []
        missing = REQUIRED_COLUMNS.difference(columns)
        if missing:
            raise ValueError(f"{path.name}: отсутствуют столбцы {', '.join(sorted(missing))}")
        if len(columns) != len(set(columns)):
            raise ValueError(f"{path.name}: повторяющиеся заголовки столбцов")

        for row in reader:
            location = f"{path.name}, строка {reader.line_num}"
            if None in row or None in row.values():
                raise ValueError(f"{location}: неверное количество полей")
            message_id = row["id"].strip()
            if not message_id:
                raise ValueError(f"{location}: пустой ID")
            if message_id in predictions:
                raise ValueError(f"{location}: повторяется ID {message_id}")

            try:
                intent_id = int(row["intent_id"])
            except ValueError as error:
                raise ValueError(f"{location}: intent_id должен быть целым числом") from error
            if intent_id not in range(16):
                raise ValueError(f"{location}: intent_id должен быть от 0 до 15")
            decision = row["decision"].strip()
            if decision not in DECISIONS:
                raise ValueError(f"{location}: недопустимое decision {decision!r}")
            predictions[message_id] = Prediction(intent_id, decision, row.get("reason", "").strip())

    return predictions


def main() -> int:
    parser = argparse.ArgumentParser(description="Сравнить две разметки МОТИВ по ID")
    parser.add_argument("--baseline", type=Path, default=ROOT / "submission.csv")
    parser.add_argument("--candidate", type=Path, default=ROOT / "gpt-6.2-submission.csv")
    args = parser.parse_args()

    try:
        baseline = read_submission(args.baseline)
        candidate = read_submission(args.candidate)
    except (csv.Error, OSError, ValueError) as error:
        parser.error(str(error))

    common_ids = sorted(baseline.keys() & candidate.keys())
    print(f"1. {args.baseline.name} ({len(baseline)} строк)")
    print(f"2. {args.candidate.name} ({len(candidate)} строк)")
    print(f"Общих ID: {len(common_ids)}")
    for name, ids in (
        (args.baseline.name, baseline.keys() - candidate.keys()),
        (args.candidate.name, candidate.keys() - baseline.keys()),
    ):
        if ids:
            print(f"Только в {name}: {', '.join(sorted(ids))}")
    if not common_ids:
        return 0

    intent_matches = 0
    decision_matches = 0
    differences = []
    for message_id in common_ids:
        first = baseline[message_id]
        second = candidate[message_id]
        same_intent = first.intent_id == second.intent_id
        same_decision = first.decision == second.decision
        intent_matches += same_intent
        decision_matches += same_decision
        if not (same_intent and same_decision):
            differences.append((message_id, first, second))

    total = len(common_ids)
    for label, matches in (
        ("Совпадает intent_id", intent_matches),
        ("Совпадает decision", decision_matches),
        ("Совпадают оба поля", total - len(differences)),
    ):
        print(f"{label}: {matches}/{total} ({matches / total:.1%})")

    print(f"\nСтрок с различиями: {len(differences)}")
    for message_id, first, second in differences:
        print(f"{message_id}: ({first.intent_id}, {first.decision}) → ({second.intent_id}, {second.decision})")
        if second.reason:
            print(f"  {second.reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
