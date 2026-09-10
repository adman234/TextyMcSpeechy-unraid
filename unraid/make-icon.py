#!/usr/bin/env python3
"""Generate unraid/textymcspeechy.png -- the container icon shown in the Unraid UI.

Same approach as make-icon.py: no third-party dependencies, supersampled
distance fields, PNG written by hand, so it can be regenerated anywhere.
"""
import math
import struct
import zlib
from pathlib import Path

SIZE = 256
SS = 3  # supersampling factor
BG = (0x1C, 0x2A, 0x33)
BG_EDGE = (0x11, 0x1A, 0x20)
FG = (0x5E, 0xD3, 0xB0)
RADIUS = 58


def rounded_rect_sdf(x, y, w, h, r):
    """Signed distance to a rounded rectangle centred on (w/2, h/2)."""
    qx = abs(x - w / 2) - (w / 2 - r)
    qy = abs(y - h / 2) - (h / 2 - r)
    return math.hypot(max(qx, 0), max(qy, 0)) + min(max(qx, qy), 0) - r


def segment_sdf(px, py, ax, ay, bx, by):
    """Signed distance to a line segment (round caps come from thresholding)."""
    vx, vy = bx - ax, by - ay
    wx, wy = px - ax, py - ay
    denom = vx * vx + vy * vy
    t = 0.0 if denom == 0 else max(0.0, min(1.0, (wx * vx + wy * vy) / denom))
    return math.hypot(wx - t * vx, wy - t * vy)


def mix(c0, c1, t):
    return tuple(round(a + (b - a) * t) for a, b in zip(c0, c1))


def main():
    # A speech waveform -- what goes in, and what the trained voice puts out.
    stroke = 17.0
    cy = SIZE / 2
    # (x, half-height) for each bar, symmetric about the centre line.
    bars = [(52, 20), (86, 46), (120, 72), (154, 46), (188, 20)]
    strokes = [(x, cy - h, x, cy + h) for x, h in bars]

    rows = []
    n = SS * SS
    for py in range(SIZE):
        row = bytearray([0])  # PNG filter type 0
        for px in range(SIZE):
            r_acc = g_acc = b_acc = a_acc = 0
            for sy in range(SS):
                for sx in range(SS):
                    x = px + (sx + 0.5) / SS
                    y = py + (sy + 0.5) / SS

                    outside = rounded_rect_sdf(x, y, SIZE, SIZE, RADIUS)
                    if outside > 0.75:
                        continue
                    cover = min(1.0, max(0.0, 0.5 - outside))

                    # Subtle vertical gradient so the tile doesn't read flat.
                    base = mix(BG, BG_EDGE, y / SIZE)

                    d = min(segment_sdf(x, y, *s) for s in strokes)
                    ink = min(1.0, max(0.0, (stroke / 2 - d) + 0.5))
                    col = mix(base, FG, ink)

                    r_acc += col[0]
                    g_acc += col[1]
                    b_acc += col[2]
                    a_acc += cover * 255
            row += bytes((r_acc // n, g_acc // n, b_acc // n, round(a_acc / n)))
        rows.append(bytes(row))

    raw = b"".join(rows)

    def chunk(tag, data):
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", SIZE, SIZE, 8, 6, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, 9))
    png += chunk(b"IEND", b"")

    out = Path(__file__).with_name("textymcspeechy.png")
    out.write_bytes(png)
    print(f"wrote {out} ({len(png)} bytes)")


if __name__ == "__main__":
    main()
