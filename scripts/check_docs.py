"""Validate active Markdown links without loading archived history or generated artifacts."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"\[[^\]]*\]\(([^)\s#]+)(?:#[^)]*)?\)")


def main():
    bad = []
    for path in ROOT.rglob("*.md"):
        rel = path.relative_to(ROOT)
        if any(p.startswith(".") for p in rel.parts) or rel.parts[0] in {"archive", "runs", "notebooks", "artifacts"}:
            continue
        for target in LINK.findall(path.read_text()):
            if target.startswith(("http:", "https:", "mailto:")):
                continue
            if not (path.parent / target).exists():
                bad.append(f"{rel}: {target}")
    print("\n".join(bad) if bad else "Active Markdown links PASS; archive excluded")
    return bool(bad)


if __name__ == "__main__":
    raise SystemExit(main())
