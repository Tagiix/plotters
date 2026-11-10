"""``python -m tests.visual [slug-substring]`` builds gallery/index.html (or only.html)."""

import sys

from .build import CARDS, build

path = build(only=sys.argv[1] if len(sys.argv) > 1 else None)
errors = path.read_text(encoding="utf-8").count('class="err"')
print(f"wrote {path} ({len(CARDS)} cards registered, {errors} errors)")
sys.exit(1 if errors else 0)
