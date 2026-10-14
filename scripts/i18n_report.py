"""Print, per language, how many WebUI translation keys are missing.

The reference is webui/i18n/en.json. Usage:

    python scripts/i18n_report.py            # summary table
    python scripts/i18n_report.py --keys     # also list every missing key
    python scripts/i18n_report.py --write-allowlist
        # regenerate test/i18n_allowlist.json from the current state

test/test_i18n_keys.py fails when a language lacks a key that is neither in
en.json's allowlist entry for that language nor translated. Shrink the allowlist
as translations land; never grow it to hide a new untranslated string.
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
I18N_DIR = ROOT / "webui" / "i18n"
ALLOWLIST = ROOT / "test" / "i18n_allowlist.json"
REFERENCE = "en"


def load_keys(path: Path) -> set:
    data = json.loads(path.read_text(encoding="utf-8"))
    return set(data.get("Translation", {}))


def missing_by_language() -> dict:
    reference = load_keys(I18N_DIR / f"{REFERENCE}.json")
    result = {}
    for path in sorted(I18N_DIR.glob("*.json")):
        lang = path.stem
        if lang == REFERENCE:
            continue
        result[lang] = sorted(reference - load_keys(path))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--keys", action="store_true", help="list missing keys")
    parser.add_argument(
        "--write-allowlist",
        action="store_true",
        help="rewrite test/i18n_allowlist.json with the current missing keys",
    )
    args = parser.parse_args()

    missing = missing_by_language()
    total = len(load_keys(I18N_DIR / f"{REFERENCE}.json"))
    print(f"reference: {REFERENCE}.json ({total} keys)")
    for lang, keys in missing.items():
        print(f"{lang:>4}: {len(keys):4d} missing")
        if args.keys:
            for key in keys:
                print(f"        - {key}")
    if args.write_allowlist:
        payload = {lang: keys for lang, keys in missing.items() if keys}
        ALLOWLIST.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {ALLOWLIST.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
