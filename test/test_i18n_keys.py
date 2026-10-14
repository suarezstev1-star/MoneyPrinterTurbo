"""Guard against WebUI translations drifting from the English reference.

Every webui/i18n/<lang>.json must contain every key of en.json, except keys
listed for that language in test/i18n_allowlist.json (known gaps). The
allowlist should only ever shrink.
"""

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
I18N_DIR = ROOT / "webui" / "i18n"
ALLOWLIST = Path(__file__).resolve().parent / "i18n_allowlist.json"


def _keys(path: Path) -> set:
    return set(json.loads(path.read_text(encoding="utf-8")).get("Translation", {}))


class I18nKeyParityTests(unittest.TestCase):
    def test_translations_have_all_english_keys(self):
        allowlist = json.loads(ALLOWLIST.read_text(encoding="utf-8"))
        reference = _keys(I18N_DIR / "en.json")
        failures = []
        for path in sorted(I18N_DIR.glob("*.json")):
            if path.stem == "en":
                continue
            missing = reference - _keys(path) - set(allowlist.get(path.stem, []))
            if missing:
                failures.append(f"{path.name}: {len(missing)} missing, e.g. {sorted(missing)[:5]}")
        self.assertEqual(
            failures,
            [],
            "Translations lack keys present in en.json. Translate them "
            "(preferred); run `python scripts/i18n_report.py --keys` for details.",
        )

    def test_allowlist_only_names_known_languages_and_keys(self):
        allowlist = json.loads(ALLOWLIST.read_text(encoding="utf-8"))
        reference = _keys(I18N_DIR / "en.json")
        languages = {p.stem for p in I18N_DIR.glob("*.json")}
        self.assertLessEqual(set(allowlist), languages - {"en"})
        for lang, keys in allowlist.items():
            self.assertLessEqual(
                set(keys), reference, f"{lang}: allowlist has keys not in en.json"
            )


if __name__ == "__main__":
    unittest.main()
