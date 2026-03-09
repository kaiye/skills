import random
import sys

from PIL import Image


def _kmeans_2(points: list[tuple[int, int, int]], iters: int = 30) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    # Simple k=2 k-means in RGB space with farthest-point init.
    c1 = points[random.randrange(len(points))]
    c2 = max(points, key=lambda p: (p[0] - c1[0]) ** 2 + (p[1] - c1[1]) ** 2 + (p[2] - c1[2]) ** 2)

    for _ in range(iters):
        g1: list[tuple[int, int, int]] = []
        g2: list[tuple[int, int, int]] = []

        for p in points:
            d1 = (p[0] - c1[0]) ** 2 + (p[1] - c1[1]) ** 2 + (p[2] - c1[2]) ** 2
            d2 = (p[0] - c2[0]) ** 2 + (p[1] - c2[1]) ** 2 + (p[2] - c2[2]) ** 2
            (g1 if d1 <= d2 else g2).append(p)

        if not g1 or not g2:
            break

        new_c1 = tuple(sum(x[i] for x in g1) // len(g1) for i in range(3))
        new_c2 = tuple(sum(x[i] for x in g2) // len(g2) for i in range(3))
        if new_c1 == c1 and new_c2 == c2:
            break
        c1, c2 = new_c1, new_c2

    return c1, c2


def _rgb_hex(c: tuple[int, int, int]) -> str:
    return f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}"


def _sample_border_rgb(img: Image.Image) -> list[tuple[int, int, int]]:
    w, h = img.size
    px = img.load()

    step = max(1, min(w, h) // 256)
    samples: list[tuple[int, int, int]] = []

    for x in range(0, w, step):
        r, g, b, a = px[x, 0]
        if a > 10:
            samples.append((r, g, b))
        r, g, b, a = px[x, h - 1]
        if a > 10:
            samples.append((r, g, b))

    for y in range(0, h, step):
        r, g, b, a = px[0, y]
        if a > 10:
            samples.append((r, g, b))
        r, g, b, a = px[w - 1, y]
        if a > 10:
            samples.append((r, g, b))

    return samples


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python3 bg2colors.py input.png", file=sys.stderr)
        return 2

    img = Image.open(sys.argv[1]).convert("RGBA")
    samples = _sample_border_rgb(img)
    if len(samples) < 10:
        print("error: not enough border samples", file=sys.stderr)
        return 1

    c1, c2 = _kmeans_2(samples)
    print(_rgb_hex(c1))
    print(_rgb_hex(c2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
