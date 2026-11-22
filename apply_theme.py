"""Switch the app's color theme for good.

Usage:
    python apply_theme.py                 shows the available themes
    python apply_theme.py forest_night    applies one

It sets DEFAULT_THEME in themes.py and writes a matching .streamlit/config.toml, so the native
Streamlit widgets (inputs, tabs, tables) use the same colors as the rest of the app.
Restart Streamlit afterwards. When you have chosen, set THEME_PICKER = False in ui.py to hide the
live theme picker from visitors.
"""

import re
import sys
from pathlib import Path

import themes

ROOT = Path(__file__).resolve().parent


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in themes.THEMES:
        print("Available themes:")
        for name, t in themes.THEMES.items():
            print(f"  {name:18s} {t['mode']:5s} {t['note']}")
        print("\nUsage: python apply_theme.py <theme name>")
        return
    name = sys.argv[1]

    config_dir = ROOT / ".streamlit"
    config_dir.mkdir(exist_ok=True)
    (config_dir / "config.toml").write_text(themes.native_config(name))

    path = ROOT / "themes.py"
    text = path.read_text()
    new_text, n = re.subn(r'^DEFAULT_THEME = ".*"$', f'DEFAULT_THEME = "{name}"', text, count=1, flags=re.M)
    if n != 1:
        print("Could not find the DEFAULT_THEME line in themes.py. Set it by hand.")
        return
    path.write_text(new_text)
    print(f"Theme set to {name}. Restart Streamlit to see it everywhere.")


if __name__ == "__main__":
    main()
