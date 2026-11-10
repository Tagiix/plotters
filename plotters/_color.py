"""Colour maths used to build and validate the palette.

Private module: nothing here draws. It converts between hex / sRGB / OKLCH, simulates
colour-vision deficiencies and measures perceptual distance (CIEDE2000), which is what the
palette tests rely on.
"""

from __future__ import annotations

import math
from typing import Literal

RGB = tuple[float, float, float]
Deficiency = Literal["protan", "deutan", "tritan"]

# Machado, Oliveira & Fernandes (2009), severity 1.0, applied to linear RGB.
_CVD_MATRICES: dict[str, tuple[RGB, RGB, RGB]] = {
    "protan": (
        (0.152286, 1.052583, -0.204868),
        (0.114503, 0.786281, 0.099216),
        (-0.003882, -0.048116, 1.051998),
    ),
    "deutan": (
        (0.367322, 0.860646, -0.227968),
        (0.280085, 0.672501, 0.047413),
        (-0.011820, 0.042940, 0.968881),
    ),
    "tritan": (
        (1.255528, -0.076749, -0.178779),
        (-0.078411, 0.930809, 0.147602),
        (0.004733, 0.691367, 0.303900),
    ),
}

DEFICIENCIES: tuple[Deficiency, ...] = ("protan", "deutan", "tritan")


def hex_to_rgb(value: str) -> RGB:
    """Parse ``'#rrggbb'`` into sRGB floats in [0, 1]."""
    v = value.lstrip("#")
    return (int(v[0:2], 16) / 255, int(v[2:4], 16) / 255, int(v[4:6], 16) / 255)


def rgb_to_hex(rgb: RGB) -> str:
    """Format sRGB floats in [0, 1] as ``'#rrggbb'`` (clipped)."""
    r, g, b = (round(min(1.0, max(0.0, c)) * 255) for c in rgb)
    return f"#{r:02x}{g:02x}{b:02x}"


def _to_linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _to_srgb(c: float) -> float:
    c = min(1.0, max(0.0, c))
    return 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055


def relative_luminance(value: str) -> float:
    """WCAG relative luminance of a hex colour (0 = black, 1 = white)."""
    r, g, b = (_to_linear(c) for c in hex_to_rgb(value))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast_ratio(a: str, b: str) -> float:
    """WCAG contrast ratio between two hex colours (1 to 21)."""
    la, lb = relative_luminance(a), relative_luminance(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def simulate(value: str, deficiency: Deficiency) -> str:
    """Return ``value`` as seen with a colour-vision deficiency, as hex."""
    lin = [_to_linear(c) for c in hex_to_rgb(value)]
    m = _CVD_MATRICES[deficiency]
    out = tuple(_to_srgb(sum(row[i] * lin[i] for i in range(3))) for row in m)
    return rgb_to_hex(out)  # type: ignore[arg-type]


def hex_to_oklch(value: str) -> tuple[float, float, float]:
    """Convert hex to OKLCH ``(L, C, h_degrees)``."""
    r, g, b = (_to_linear(c) for c in hex_to_rgb(value))
    l_ = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m_ = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s_ = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    l_, m_, s_ = (math.copysign(abs(x) ** (1 / 3), x) for x in (l_, m_, s_))
    lab_l = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
    lab_a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
    lab_b = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_
    return lab_l, math.hypot(lab_a, lab_b), math.degrees(math.atan2(lab_b, lab_a)) % 360


def oklch_to_hex(lightness: float, chroma: float, hue: float) -> str:
    """Convert OKLCH to hex, reducing chroma until the colour fits in sRGB."""
    for scale in [1 - 0.02 * i for i in range(51)]:
        c = chroma * scale
        a = c * math.cos(math.radians(hue))
        b = c * math.sin(math.radians(hue))
        l_ = (lightness + 0.3963377774 * a + 0.2158037573 * b) ** 3
        m_ = (lightness - 0.1055613458 * a - 0.0638541728 * b) ** 3
        s_ = (lightness - 0.0894841775 * a - 1.2914855480 * b) ** 3
        r = 4.0767416621 * l_ - 3.3077115913 * m_ + 0.2309699292 * s_
        g = -1.2684380046 * l_ + 2.6097574011 * m_ - 0.3413193965 * s_
        bl = -0.0041960863 * l_ - 0.7034186147 * m_ + 1.7076147010 * s_
        if all(-1e-4 <= x <= 1 + 1e-4 for x in (r, g, bl)):
            return rgb_to_hex((_to_srgb(r), _to_srgb(g), _to_srgb(bl)))
    return rgb_to_hex((_to_srgb(lightness),) * 3)  # type: ignore[arg-type]


def _to_lab(value: str) -> tuple[float, float, float]:
    r, g, b = (_to_linear(c) for c in hex_to_rgb(value))
    x = (0.4124564 * r + 0.3575761 * g + 0.1804375 * b) / 0.95047
    y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    z = (0.0193339 * r + 0.1191920 * g + 0.9503041 * b) / 1.08883

    def f(t: float) -> float:
        return t ** (1 / 3) if t > 216 / 24389 else (24389 / 27 * t + 16) / 116

    fx, fy, fz = f(x), f(y), f(z)
    return 116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)


def delta_e(a: str, b: str) -> float:
    """CIEDE2000 colour difference between two hex colours (about 1 = just noticeable)."""
    l1, a1, b1 = _to_lab(a)
    l2, a2, b2 = _to_lab(b)
    c1, c2 = math.hypot(a1, b1), math.hypot(a2, b2)
    c_bar7 = ((c1 + c2) / 2) ** 7
    g = 0.5 * (1 - math.sqrt(c_bar7 / (c_bar7 + 25**7)))
    a1p, a2p = (1 + g) * a1, (1 + g) * a2
    c1p, c2p = math.hypot(a1p, b1), math.hypot(a2p, b2)
    h1p = math.degrees(math.atan2(b1, a1p)) % 360
    h2p = math.degrees(math.atan2(b2, a2p)) % 360

    dl, dc = l2 - l1, c2p - c1p
    if c1p * c2p == 0:
        dh = 0.0
    else:
        dh = h2p - h1p
        dh += -360 if dh > 180 else 360 if dh < -180 else 0
    dhp = 2 * math.sqrt(c1p * c2p) * math.sin(math.radians(dh / 2))

    lp, cp = (l1 + l2) / 2, (c1p + c2p) / 2
    if c1p * c2p == 0:
        hp = h1p + h2p
    elif abs(h1p - h2p) <= 180:
        hp = (h1p + h2p) / 2
    else:
        hp = (h1p + h2p + (360 if h1p + h2p < 360 else -360)) / 2
    t = (
        1
        - 0.17 * math.cos(math.radians(hp - 30))
        + 0.24 * math.cos(math.radians(2 * hp))
        + 0.32 * math.cos(math.radians(3 * hp + 6))
        - 0.20 * math.cos(math.radians(4 * hp - 63))
    )
    sl = 1 + 0.015 * (lp - 50) ** 2 / math.sqrt(20 + (lp - 50) ** 2)
    sc = 1 + 0.045 * cp
    sh = 1 + 0.015 * cp * t
    d_theta = 30 * math.exp(-(((hp - 275) / 25) ** 2))
    rc = 2 * math.sqrt(cp**7 / (cp**7 + 25**7))
    rt = -math.sin(math.radians(2 * d_theta)) * rc
    return math.sqrt(
        (dl / sl) ** 2 + (dc / sc) ** 2 + (dhp / sh) ** 2 + rt * (dc / sc) * (dhp / sh)
    )


def min_adjacent_delta_e(colors: list[str], deficiency: Deficiency | None = None) -> float:
    """Smallest ΔE between neighbouring colours, optionally as seen with a deficiency."""
    seen = colors if deficiency is None else [simulate(c, deficiency) for c in colors]
    return min(delta_e(seen[i], seen[i + 1]) for i in range(len(seen) - 1))


def min_pair_delta_e(colors: list[str], deficiency: Deficiency | None = None) -> float:
    """Smallest ΔE over every pair of colours, optionally as seen with a deficiency."""
    seen = colors if deficiency is None else [simulate(c, deficiency) for c in colors]
    return min(delta_e(seen[i], seen[j]) for i in range(len(seen)) for j in range(i + 1, len(seen)))


def with_alpha(value: str, alpha: float) -> str:
    """``'#rrggbb'`` as an ``rgba(...)`` string with the given opacity."""
    r, g, b = (round(c * 255) for c in hex_to_rgb(value))
    return f"rgba({r}, {g}, {b}, {alpha})"
