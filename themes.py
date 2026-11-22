"""Color themes for the Agriculture Smart System app.

Each theme needs only four colors (background, surface, ink, accent) plus its mode. Everything else
(borders, muted text, bar tracks, chips and so on) is derived from them by tokens().

To switch theme for good, run:  python apply_theme.py <theme name>
That sets DEFAULT_THEME below and writes a matching .streamlit/config.toml, so the native Streamlit
widgets (inputs, tabs, tables) use the same colors as the custom parts of the app.
"""

DEFAULT_THEME = "meadow"

THEMES = {
    # ==== light
    "meadow": {
        "label": "Meadow", "mode": "light",
        "note": "Fresh leaf green on a clean white",
        "bg": "#F3F8EC", "surface": "#FFFFFF", "ink": "#1A2E1E", "accent": "#2E7D32",
    },
    "irrigation": {
        "label": "Irrigation", "mode": "light",
        "note": "Water teal with crisp blue-gray",
        "bg": "#EEF6F8", "surface": "#FFFFFF", "ink": "#14303A", "accent": "#0E7C86",
    },
    "harvest": {
        "label": "Harvest", "mode": "light",
        "note": "Ripe wheat amber on neutral light gray",
        "bg": "#F5F6F3", "surface": "#FFFFFF", "ink": "#2A2B24", "accent": "#A85A08",
    },
    "orchard": {
        "label": "Orchard", "mode": "light",
        "note": "Apple red with soft leaf-gray",
        "bg": "#F6F8F4", "surface": "#FFFFFF", "ink": "#1F2A22", "accent": "#B8322A",
    },
    # ==== dark
    "forest_night": {
        "label": "Forest Night", "mode": "dark",
        "note": "Deep canopy green with bright leaf accents",
        "bg": "#0E1A13", "surface": "#162A1E", "ink": "#E4F0E6", "accent": "#6CCB7B", "accent_ink": "#06210F",
    },
    "night_irrigation": {
        "label": "Night Irrigation", "mode": "dark",
        "note": "Dark water blue with cyan accents",
        "bg": "#0B171E", "surface": "#12252F", "ink": "#E2F0F5", "accent": "#4FC3D9", "accent_ink": "#04222A",
    },
    "deep_soil": {
        "label": "Deep Soil", "mode": "dark",
        "note": "Rich soil brown with golden wheat accents",
        "bg": "#16120E", "surface": "#221B15", "ink": "#F0E8DA", "accent": "#E3A72F", "accent_ink": "#241800",
    },
    "dusk_orchard": {
        "label": "Dusk Orchard", "mode": "dark",
        "note": "Plum charcoal with warm apple accents",
        "bg": "#171315", "surface": "#231C1F", "ink": "#F3E9EA", "accent": "#EF7B6E", "accent_ink": "#2A0A06",
    },
}


def _rgb(hex_color):
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def mix(a, b, t):
    """Blend color a toward color b by fraction t (0 keeps a, 1 gives b)."""
    ra, rb = _rgb(a), _rgb(b)
    return "#%02X%02X%02X" % tuple(round(x * (1 - t) + y * t) for x, y in zip(ra, rb))


def luminance(hex_color):
    def lin(c):
        c = c / 255
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (lin(v) for v in _rgb(hex_color))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a, b):
    la, lb = sorted((luminance(a), luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def tokens(name):
    """All colors used by the app for one theme."""
    t = THEMES[name]
    dark = t["mode"] == "dark"
    bg, surface, ink, accent = t["bg"], t["surface"], t["ink"], t["accent"]
    k = {
        "name": name, "label": t["label"], "mode": t["mode"],
        "bg": bg, "surface": surface, "ink": ink, "accent": accent,
        "accent_ink": t.get("accent_ink", "#FFFFFF"),
        "text": mix(ink, bg, 0.18),
        "muted": mix(ink, bg, 0.36 if dark else 0.30),
        "border": mix(surface, ink, 0.14),
        "rule": mix(bg, ink, 0.16),
        "sidebar": mix(bg, ink, 0.05),
        "track": mix(surface, ink, 0.10),
        "alt": mix(surface, accent, 0.45),
        "soft": mix(surface, accent, 0.16),
        "chip_bg": mix(surface, ink, 0.04),
        "row_line": mix(surface, ink, 0.08),
    }
    if dark:
        k.update(ok_bg="#1F3B27", ok_fg="#8FDB9C", warn_bg="#3F3213", warn_fg="#F0C25E",
                 bad_bg="#43211B", bad_fg="#F4A08C", good="#6CCB7B", mid="#E3B33F", bad="#E5684F")
    else:
        k.update(ok_bg="#E1F0DA", ok_fg="#25601F", warn_bg="#FAEBC6", warn_fg="#7A4E06",
                 bad_bg="#F8DAD2", bad_fg="#8B2A14", good="#3C8E3F", mid="#D39A2F", bad="#C4472E")
    k["neutral_bg"] = mix(surface, ink, 0.09)
    k["neutral_fg"] = k["text"]
    return k


def native_config(name):
    """Text for .streamlit/config.toml so native Streamlit widgets match the theme."""
    k = tokens(name)
    return (
        "[theme]\n"
        f'base = "{k["mode"]}"\n'
        f'primaryColor = "{k["accent"]}"\n'
        f'backgroundColor = "{k["bg"]}"\n'
        f'secondaryBackgroundColor = "{k["sidebar"]}"\n'
        f'textColor = "{k["ink"]}"\n'
        'font = "sans serif"\n\n'
        "[browser]\n"
        "gatherUsageStats = false\n"
    )
