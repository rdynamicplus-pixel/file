#!/usr/bin/env python3
"""R DYNAMIC PLUS - 40s cinematic commercial renderer (frames -> ffmpeg)."""
import sys, os, math, subprocess
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
from multiprocessing import Pool

SCR = os.path.dirname(os.path.abspath(__file__))
UP = "/root/.claude/uploads/3db9658d-21c8-587f-8ed6-b820f03ee960/"
FONT_SERIF = SCR + "/fonts/Cinzel%5Bwght%5D.ttf"
FONT_SANS = SCR + "/fonts/Montserrat%5Bwght%5D.ttf"
FPS = 30
LANG = os.environ.get("AD_LANG", "en")
AR = LANG == "ar"
def X(en, ar): return ar if AR else en
FONT_AMIRI_B = SCR + "/fonts/Amiri-Bold.ttf"
FONT_AMIRI_R = SCR + "/fonts/Amiri-Regular.ttf"
FONT_CAIRO = SCR + "/fonts/Cairo%5Bslnt%2Cwght%5D.ttf"
def is_ar(t): return any("\u0600" <= c <= "\u06ff" for c in t)
DUR = 40.0

IMGS = {
    "urus": ("ab0a22c0-image.jpg", None, (0.47, 0.60)),
    "mclaren": ("9f268812-image.jpg", None, (0.50, 0.68)),
    "gt3rs": ("d37e5853-image.jpg", None, (0.47, 0.62)),
    "huracan": ("ad295f6a-image.jpg", None, (0.50, 0.74)),
    "m4": ("f5e95a5d-image.png", (0, 560, 1320, 2305), (0.48, 0.60)),
    "escalade": ("f99b24ec-image.jpg", None, (0.50, 0.62)),
    "gmc": ("8f9d21d4-image.jpg", None, (0.50, 0.62)),
    "macan": ("9692ecce-image.jpg", None, (0.50, 0.50)),
    "patrol": ("64ee9382-image.jpg", None, (0.47, 0.60)),
    "bmw4": ("d22d5644-image.jpg", None, (0.53, 0.60)),
}

GOLD_TOP = np.array([0.98, 0.90, 0.64], np.float32)
GOLD_MID = np.array([0.85, 0.66, 0.30], np.float32)
GOLD_BOT = np.array([0.55, 0.39, 0.13], np.float32)
WHITE = np.array([0.96, 0.95, 0.92], np.float32)
BLACK = np.array([0.02, 0.02, 0.02], np.float32)

def clamp(x, a=0.0, b=1.0): return max(a, min(b, x))
def ease(x):  # easeOutCubic
    x = clamp(x); return 1 - (1 - x) ** 3
def ease_io(x):
    x = clamp(x); return x * x * (3 - 2 * x)
def win(t, a, b, fi=0.5, fo=0.5):
    """1 inside [a,b] with fades."""
    if t < a or t > b: return 0.0
    v = 1.0
    if fi > 0: v = min(v, ease_io((t - a) / fi))
    if fo > 0: v = min(v, ease_io((b - t) / fo))
    return v

# ---------------------------------------------------------------- images
_img_cache = {}
def load_src(key):
    if key in _img_cache: return _img_cache[key]
    fn, crop, focal = IMGS[key]
    im = Image.open(UP + fn).convert("RGB")
    if crop: im = im.crop(crop)
    a = np.asarray(im).astype(np.float32) / 255.0
    # grade: gentle contrast S-curve, slight warmth, richer blacks
    a = np.clip(a, 0, 1)
    a = a * a * (3 - 2 * a) * 0.35 + a * 0.65
    a[..., 0] *= 1.03; a[..., 2] *= 0.97
    a = np.clip(a, 0, 1)
    _img_cache[key] = (a, focal)
    return _img_cache[key]

_plate_cache = {}
def plate_assets(key, W, H, box, mode, base_zoom):
    ck = (key, W, H, box, mode, base_zoom)
    if ck in _plate_cache: return _plate_cache[ck]
    a, focal = load_src(key)
    sh, sw = a.shape[:2]
    bx, by, bw, bh = box
    if mode == "fit": s = min(bw / sw, bh / sh)
    else: s = max(bw / sw, bh / sh)
    s *= base_zoom
    # pre-scale sharp image to ~1.12x the max size needed
    pre = min(1.0, s * 1.12)
    small = cv2.resize(a, (max(2, int(sw * pre)), max(2, int(sh * pre))), interpolation=cv2.INTER_AREA)
    # feather mask in source space
    ph, pw = small.shape[:2]
    m = np.zeros((ph, pw), np.float32)
    f = int(min(ph, pw) * 0.06) + 2
    m[f:ph - f, f:pw - f] = 1.0
    m = cv2.GaussianBlur(m, (0, 0), f * 0.6)
    # blurred background (cover whole frame)
    sb = max(W / sw, H / sh) * 1.15
    bg = cv2.resize(a, (int(sw * sb / 8) + 1, int(sh * sb / 8) + 1), interpolation=cv2.INTER_AREA)
    bg = cv2.GaussianBlur(bg, (0, 0), 6)
    bg = cv2.resize(bg, (int(sw * sb), int(sh * sb)), interpolation=cv2.INTER_LINEAR)
    bg = bg * 0.38
    res = dict(img=small, mask=m, pre=pre, s=s, bg=bg, focal=focal, sw=sw, sh=sh)
    _plate_cache[ck] = res
    return res

def render_plate(W, H, key, box, mode, base_zoom, z, pan=(0, 0)):
    """box in pixels; z = extra zoom factor (animated); pan in px."""
    A = plate_assets(key, W, H, box, mode, base_zoom)
    bx, by, bw, bh = box
    # background
    bg = A["bg"]; bh_, bw_ = bg.shape[:2]
    zb = 1.0 + (z - 1.0) * 0.5
    M = np.float32([[zb, 0, W / 2 - bw_ / 2 * zb], [0, zb, H / 2 - bh_ / 2 * zb]])
    out = cv2.warpAffine(bg, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    # sharp image: focal point maps to its z=1 position, zoom around it
    fx, fy = A["focal"]; sw, sh = A["sw"], A["sh"]
    s = A["s"]
    cx, cy = bx + bw / 2, by + bh / 2
    # z=1 placement: image centered in box, but if covering, align focal toward center
    if mode == "fit":
        Dfx = cx + (fx - 0.5) * sw * s
        Dfy = cy + (fy - 0.5) * sh * s
    else:
        Dfx = cx + (fx - 0.5) * sw * s
        Dfy = cy + (fy - 0.5) * sh * s
        # clamp so image still covers box
        w_img, h_img = sw * s, sh * s
        left = Dfx - fx * w_img
        left = min(bx, max(bx + bw - w_img, left)); Dfx = left + fx * w_img
        top = Dfy - fy * h_img
        top = min(by, max(by + bh - h_img, top)); Dfy = top + fy * h_img
    k = s * z / A["pre"]
    Fx, Fy = fx * A["img"].shape[1], fy * A["img"].shape[0]
    M = np.float32([[k, 0, Dfx + pan[0] - Fx * k], [0, k, Dfy + pan[1] - Fy * k]])
    sharp = cv2.warpAffine(A["img"], M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    if mode == "fit":
        m = cv2.warpAffine(A["mask"], M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        out = out * (1 - m[..., None]) + sharp * m[..., None]
    else:
        m = np.zeros((H, W), np.float32); m[int(by):int(by + bh), int(bx):int(bx + bw)] = 1
        out = out * (1 - m[..., None]) + sharp * m[..., None]
    return out

# ---------------------------------------------------------------- text
_font_cache = {}
def font(kind, size, weight):
    k = (kind, size, weight)
    if k not in _font_cache:
        f = ImageFont.truetype(FONT_SERIF if kind == "serif" else FONT_SANS, size)
        f.set_variation_by_name(weight)
        _font_cache[k] = f
    return _font_cache[k]

_text_cache = {}
def text_layer(text, kind, size, weight, tracking=0.0):
    k = (text, kind, size, weight, tracking)
    if k in _text_cache: return _text_cache[k]
    if is_ar(text):
        if kind == "serif":
            sz = int(size * 1.12)
            f = ImageFont.truetype(FONT_AMIRI_B if weight in ("Bold", "Black") else FONT_AMIRI_R, sz)
        else:
            sz = int(size * 1.15)
            f = ImageFont.truetype(FONT_CAIRO, sz)
            wmap = {"Light": "Light", "Regular": "Regular", "Medium": "Medium", "SemiBold": "SemiBold", "Bold": "Bold"}
            try: f.set_variation_by_name(wmap.get(weight, "Regular"))
            except Exception: pass
        pad = int(sz * 0.8)
        tmp = Image.new("L", (10, 10)); d0 = ImageDraw.Draw(tmp)
        bb = d0.textbbox((0, 0), text, font=f, direction="rtl", language="ar")
        w, h = bb[2] - bb[0] + 2 * pad, bb[3] - bb[1] + 2 * pad
        im = Image.new("L", (w, h), 0); d = ImageDraw.Draw(im)
        d.text((pad - bb[0], pad - bb[1]), text, font=f, fill=255, direction="rtl", language="ar")
        m = np.asarray(im).astype(np.float32) / 255.0
        ys, xs = np.nonzero(m > 0.01)
        m = m[max(0, ys.min() - pad // 3): ys.max() + pad // 3, max(0, xs.min() - pad // 3): xs.max() + pad // 3]
        sh = cv2.GaussianBlur(m, (0, 0), max(1.5, size * 0.12))
        L = dict(m=m, sh=sh, w=m.shape[1], h=m.shape[0], rtl=True)
        _text_cache[k] = L
        return L
    f = font(kind, size, weight)
    tr = tracking * size
    widths = [f.getlength(c) for c in text]
    total = sum(widths) + tr * max(0, len(text) - 1)
    asc, desc = f.getmetrics()
    pad = int(size * 0.5)
    w, h = int(total) + 2 * pad, asc + desc + 2 * pad
    im = Image.new("L", (w, h), 0); d = ImageDraw.Draw(im)
    x = pad
    for c, cw in zip(text, widths):
        d.text((x, pad), c, font=f, fill=255); x += cw + tr
    m = np.asarray(im).astype(np.float32) / 255.0
    ys, xs = np.nonzero(m > 0.01)
    m = m[max(0, ys.min() - pad // 2): ys.max() + pad // 2, max(0, xs.min() - pad // 2): xs.max() + pad // 2]
    sh = cv2.GaussianBlur(m, (0, 0), max(1.5, size * 0.12))
    L = dict(m=m, sh=sh, w=m.shape[1], h=m.shape[0])
    _text_cache[k] = L
    return L

def gold_fill(h, w, shine=None):
    y = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    top = np.clip(1 - y * 2, 0, 1)[..., None]; bot = np.clip(y * 2 - 1, 0, 1)[..., None]
    mid = 1 - top - bot
    col = top * GOLD_TOP + mid * GOLD_MID + bot * GOLD_BOT
    col = np.broadcast_to(col, (h, w, 3)).copy()
    if shine is not None:
        xx = np.arange(w, dtype=np.float32)[None, :] + np.arange(h, dtype=np.float32)[:, None] * 0.5
        pos = shine * (w + h) - h * 0.5
        band = np.exp(-((xx - pos) / (max(w, h) * 0.06)) ** 2)[..., None]
        col = col * (1 - band * 0.8) + np.array([1.0, 0.97, 0.86], np.float32) * band * 0.8
    return col

def blit(canvas, L, x, y, alpha=1.0, color="white", wipe=None, shadow=0.55, shine=None, anchor="c"):
    """Composite text layer. x,y = anchor position. color 'gold'|'white'|'black'|rgb array."""
    if alpha <= 0.002: return
    H, W = canvas.shape[:2]
    w, h = L["w"], L["h"]
    if anchor == "c": x0, y0 = int(x - w / 2), int(y - h / 2)
    elif anchor == "l": x0, y0 = int(x), int(y - h / 2)
    else: x0, y0 = int(x - w), int(y - h / 2)
    m = L["m"]; sm = L["sh"]
    if wipe is not None and wipe < 1:
        feather = max(8, w * 0.25)
        xs = np.arange(w, dtype=np.float32)
        ramp = np.clip((wipe * (w + feather) - xs) / feather, 0, 1)[None, :]
        if L.get("rtl"): ramp = ramp[:, ::-1]
        m = m * ramp; sm = sm * ramp
    cx0, cy0 = max(0, x0), max(0, y0); cx1, cy1 = min(W, x0 + w), min(H, y0 + h)
    if cx1 <= cx0 or cy1 <= cy0: return
    sl = (slice(cy0 - y0, cy1 - y0), slice(cx0 - x0, cx1 - x0))
    reg = canvas[cy0:cy1, cx0:cx1]
    if shadow > 0:
        a = (sm[sl] * shadow * alpha)[..., None]
        reg *= (1 - a)
    if isinstance(color, str):
        if color == "gold": col = gold_fill(h, w, shine)[sl]
        elif color == "white": col = WHITE
        else: col = BLACK
    else: col = color
    a = (m[sl] * alpha)[..., None]
    reg[:] = reg * (1 - a) + col * a

def hline(canvas, cx, y, length, thick, alpha, progress=1.0, color=GOLD_MID):
    if alpha <= 0 or progress <= 0: return
    H, W = canvas.shape[:2]
    L = length * progress
    x0, x1 = int(cx - L / 2), int(cx + L / 2)
    y0, y1 = int(y - thick / 2), int(y + thick / 2) + 1
    x0, x1 = max(0, x0), min(W, x1)
    if x1 <= x0: return
    xs = np.linspace(-1, 1, x1 - x0, dtype=np.float32)
    fade = (1 - np.abs(xs) ** 3)[None, :, None]
    reg = canvas[y0:y1, x0:x1]
    a = alpha * fade
    reg[:] = reg * (1 - a) + color * a

_rect_cache = {}
def round_rect_mask(w, h, r, border=0):
    k = (w, h, r, border)
    if k in _rect_cache: return _rect_cache[k]
    im = Image.new("L", (w * 2, h * 2), 0); d = ImageDraw.Draw(im)
    if border:
        d.rounded_rectangle([0, 0, w * 2 - 1, h * 2 - 1], r * 2, outline=255, width=border * 2)
    else:
        d.rounded_rectangle([0, 0, w * 2 - 1, h * 2 - 1], r * 2, fill=255)
    m = np.asarray(im.resize((w, h), Image.LANCZOS)).astype(np.float32) / 255
    _rect_cache[k] = m
    return m

def panel(canvas, cx, cy, w, h, alpha, r=18, fill_a=0.62, border_a=0.85, fill=BLACK, border=GOLD_MID, bw=2):
    if alpha <= 0: return
    H, W = canvas.shape[:2]
    w, h = int(w), int(h)
    x0, y0 = int(cx - w / 2), int(cy - h / 2)
    reg = canvas[y0:y0 + h, x0:x0 + w]
    m = round_rect_mask(w, h, r)[..., None] * fill_a * alpha
    reg[:] = reg * (1 - m) + fill * m
    if border_a > 0:
        b = round_rect_mask(w, h, r, bw)[..., None] * border_a * alpha
        reg[:] = reg * (1 - b) + border * b

def diamond(canvas, x, y, r, alpha):
    if alpha <= 0: return
    x0, y0 = int(x - r - 2), int(y - r - 2); n = int(2 * r + 5)
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    d = np.abs(xx - (x - x0)) + np.abs(yy - (y - y0))
    m = np.clip(r - d + 0.5, 0, 1)
    inner = np.clip((r * 0.45) - d + 0.5, 0, 1)
    m = np.clip(m - inner * 0.9, 0, 1)[..., None] * alpha
    reg = canvas[y0:y0 + n, x0:x0 + n]
    reg[:] = reg * (1 - m) + GOLD_TOP * m

# ---------------------------------------------------------------- logo
_logo = None
def logo_rgba():
    global _logo
    if _logo is None:
        im = np.asarray(Image.open(UP + "6c4bf4ca-image.jpg").convert("RGB").crop((250, 225, 1010, 800))).astype(np.float32) / 255
        lum = im.max(axis=2)
        a = np.clip((lum - 0.06) / 0.5, 0, 1)
        col = np.where(a[..., None] > 0.01, im / np.maximum(a[..., None], 1e-3), 0)
        col = np.clip(col, 0, 1)
        _logo = (col, a)
    return _logo

_logo_sz = {}
def blit_logo(canvas, cx, cy, width, alpha, shine=None, glow=0.0):
    if alpha <= 0: return
    col, a = logo_rgba()
    k = int(width)
    if k not in _logo_sz:
        h = int(a.shape[0] * width / a.shape[1])
        c2 = cv2.resize(col, (k, h), interpolation=cv2.INTER_AREA)
        a2 = cv2.resize(a, (k, h), interpolation=cv2.INTER_AREA)
        g = cv2.GaussianBlur(a2, (0, 0), width * 0.05)
        _logo_sz[k] = (c2, a2, g)
    c2, a2, g = _logo_sz[k]
    h, w = a2.shape
    x0, y0 = int(cx - w / 2), int(cy - h / 2)
    reg = canvas[y0:y0 + h, x0:x0 + w]
    if glow > 0:
        reg += (g * glow)[..., None] * GOLD_MID * 0.6
    c = c2
    if shine is not None:
        xx = np.arange(w, dtype=np.float32)[None, :] + np.arange(h, dtype=np.float32)[:, None] * 0.6
        pos = shine * (w + h * 0.6) - h * 0.3
        band = np.exp(-((xx - pos) / (w * 0.07)) ** 2)[..., None]
        c = np.clip(c + band * 0.6, 0, 1)
    am = (a2 * alpha)[..., None]
    reg[:] = reg * (1 - am) + c * am

# ---------------------------------------------------------------- fx
_rng = np.random.default_rng(7)
PARTS = _rng.random((70, 5)).astype(np.float32)  # x, y, size, speed, phase
_dot_cache = {}
def particles(canvas, t, amount, s):
    if amount <= 0: return
    H, W = canvas.shape[:2]
    for px, py, ps, sp, ph in PARTS:
        r = int((3 + ps * 14) * s)
        if r not in _dot_cache:
            n = 4 * r + 1; yy, xx = np.mgrid[0:n, 0:n].astype(np.float32) - 2 * r
            _dot_cache[r] = np.exp(-(xx ** 2 + yy ** 2) / (2 * (r * 0.55) ** 2))
        d = _dot_cache[r]; n = d.shape[0]
        x = int(px * W + math.sin(t * 0.6 + ph * 6) * 30 * s)
        y = int((py * H - t * (20 + sp * 50) * s) % (H + 200) - 100)
        tw = 0.5 + 0.5 * math.sin(t * (1 + sp * 2) + ph * 10)
        a = amount * (0.15 + 0.35 * tw) * (0.3 + 0.7 * (1 - ps))
        x0, y0 = x - n // 2, y - n // 2
        if x0 < 0 or y0 < 0 or x0 + n > W or y0 + n > H: continue
        canvas[y0:y0 + n, x0:x0 + n] += (d * a)[..., None] * GOLD_TOP

def streak(canvas, y, xpos, strength, s):
    """anamorphic gold light streak centered at xpos."""
    if strength <= 0: return
    H, W = canvas.shape[:2]
    yy = np.arange(H, dtype=np.float32)[:, None]
    xx = np.arange(W, dtype=np.float32)[None, :]
    v = np.exp(-((yy - y) / (5 * s)) ** 2) * np.exp(-((xx - xpos) / (W * 0.35)) ** 2)
    core = np.exp(-((yy - y) / (40 * s)) ** 2) * np.exp(-((xx - xpos) / (W * 0.08)) ** 2) * 0.5
    canvas += ((v + core) * strength)[..., None] * np.array([1.0, 0.8, 0.45], np.float32)

_vig = {}
def vignette(W, H):
    if (W, H) not in _vig:
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        d = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
        _vig[(W, H)] = np.clip(1 - 0.45 * np.clip(d - 0.55, 0, 1) ** 1.5, 0, 1)[..., None].astype(np.float32)
    return _vig[(W, H)]

_grad = {}
def vgrad(W, H, kind):
    k = (W, H, kind)
    if k not in _grad:
        y = np.linspace(0, 1, H, dtype=np.float32)
        if kind == "top": g = np.clip(1 - y / 0.35, 0, 1) ** 1.5
        elif kind == "bottom": g = np.clip((y - 0.55) / 0.45, 0, 1) ** 1.3
        elif kind == "left":
            x = np.linspace(0, 1, W, dtype=np.float32); g = np.clip(1 - x / 0.62, 0, 1) ** 1.2
            _grad[k] = np.broadcast_to(g[None, :, None], (H, W, 1)).astype(np.float32); return _grad[k]
        _grad[k] = g[:, None, None].astype(np.float32)
    return _grad[k]

def darken(canvas, g, amt):
    canvas *= (1 - g * amt)

def flash(canvas, t, cuts, s):
    for c in cuts:
        d = t - c
        if -0.25 < d < 0.4:
            k = math.exp(-(d / 0.12) ** 2)
            H, W = canvas.shape[:2]
            streak(canvas, H * 0.5, W * (0.5 + d * 1.5), 0.9 * k, s * 3)
            canvas += k * 0.10 * np.array([1.0, 0.85, 0.55], np.float32)

# ---------------------------------------------------------------- scenes
def build(W, H):
    P = H > W
    s = W / 1080 if P else H / 1080

    def T(text, kind="sans", size=40, weight="Regular", tr=0.0):
        return text_layer(text, kind, int(size * s), weight, tr)

    FULL = (0, 0, W, H)
    # (start, end, key, box, mode, base_zoom, z0, z1, pan)
    if P:
        band = (0, int(H * 0.55 - W * 0.42), W, int(W * 0.84))
        shots = [
            (4.0, 5.6, "huracan", (0, int(H * 0.55) - 400, W, 800), "fit", 1.45, 1.10, 1.02, (0, 0)),
            (5.6, 7.2, "gt3rs", (0, int(H * 0.55) - 400, W, 800), "fit", 1.45, 1.02, 1.10, (0, 0)),
            (7.2, 8.8, "mclaren", (0, int(H * 0.08), W, int(H * 0.92)), "fit", 1.0, 1.0, 1.08, (0, 0)),
            (8.8, 10.4, "m4", (0, int(H * 0.08), W, int(H * 0.92)), "fit", 1.0, 1.08, 1.0, (0, 0)),
            (10.4, 12.0, "patrol", (0, int(H * 0.08), W, int(H * 0.92)), "fit", 1.0, 1.0, 1.07, (0, 0)),
            (12.0, 16.0, "escalade", (0, int(H * 0.28 - H * 0.4), W, int(H * 0.8)), "fit", 1.0, 1.0, 1.06, (0, 0)),
            (16.0, 20.0, "gmc", (0, int(H * 0.28 - H * 0.4), W, int(H * 0.8)), "fit", 1.0, 1.06, 1.0, (0, 0)),
            (20.0, 27.0, "macan", (0, int(-H * 0.12), W, int(H * 1.0)), "fit", 1.0, 1.0, 1.06, (0, 0)),
            (27.0, 30.4, "urus", (0, int(H * 0.04), W, int(H * 0.8)), "fit", 1.0, 1.0, 1.08, (0, 0)),
            (30.4, 33.0, "huracan", (0, int(H * 0.45) - 400, W, 800), "fit", 1.45, 1.0, 1.08, (0, 0)),
            (33.0, 40.0, "bmw4", FULL, "cover", 1.0, 1.12, 1.0, (0, 0)),
        ]
    else:
        right = (int(W * 0.5), 0, int(W * 0.5), H)
        shots = [
            (4.0, 5.6, "huracan", FULL, "cover", 1.0, 1.10, 1.02, (0, 0)),
            (5.6, 7.2, "gt3rs", FULL, "cover", 1.0, 1.02, 1.10, (0, 0)),
            (7.2, 8.8, "mclaren", FULL, "fit", 1.0, 1.0, 1.08, (0, 0)),
            (8.8, 10.4, "m4", FULL, "fit", 1.0, 1.08, 1.0, (0, 0)),
            (10.4, 12.0, "patrol", FULL, "fit", 1.0, 1.0, 1.07, (0, 0)),
            (12.0, 16.0, "escalade", (int(W * 0.52), 0, int(W * 0.48), H), "fit", 1.0, 1.0, 1.06, (0, 0)),
            (16.0, 20.0, "gmc", (int(W * 0.52), 0, int(W * 0.48), H), "fit", 1.0, 1.06, 1.0, (0, 0)),
            (20.0, 27.0, "macan", (0, 0, int(W * 0.4), H), "fit", 1.0, 1.0, 1.06, (0, 0)),
            (27.0, 30.4, "urus", (int(W * 0.50), 0, int(W * 0.50), H), "fit", 1.0, 1.0, 1.08, (0, 0)),
            (30.4, 33.0, "huracan", FULL, "cover", 1.0, 1.0, 1.08, (0, 0)),
            (33.0, 40.0, "bmw4", FULL, "cover", 1.0, 1.12, 1.0, (0, 0)),
        ]
    XF = 0.35  # crossfade
    cuts = [4.0, 12.0, 20.0, 27.0, 33.0]
    soft_cuts = [5.6, 7.2, 8.8, 10.4, 16.0, 30.4]

    labels = {"huracan": "LAMBORGHINI HURACAN", "gt3rs": "PORSCHE 911 GT3 RS", "mclaren": "McLAREN",
              "m4": "BMW M4 CONVERTIBLE", "patrol": "NISSAN PATROL"}

    services = X(["Luxury & Sports Car Rental", "Airport Pickup & Delivery", "Hotel Delivery Across Dubai",
                "24/7 Customer Support", "Flexible Rental Options", "Premium Customer Experience"],
                ["تأجير السيارات الفاخرة والرياضية", "الاستلام والتوصيل من المطار", "التوصيل إلى الفنادق في جميع أنحاء دبي",
                 "خدمة عملاء على مدار الساعة 24/7", "خيارات إيجار مرنة", "تجربة عملاء مميزة"])
    phones = ["+971 52 414 5956", "+971 52 885 1019", "+971 58 994 5983", "+971 52 522 8428"]

    def frame(t):
        cv = np.zeros((H, W, 3), np.float32)
        # ---------- plates
        for (a, b, key, box, mode, bz, z0, z1, pan) in shots:
            if t < a - XF / 2 or t > b + XF / 2: continue
            wgt = 1.0
            if a > 4.0: wgt = min(wgt, ease_io((t - (a - XF / 2)) / XF))
            if b < 40.0: wgt = min(wgt, ease_io(((b + XF / 2) - t) / XF))
            if a == 4.0: wgt = min(wgt, ease_io((t - 3.7) / 0.6))
            if wgt <= 0: continue
            u = clamp((t - a) / (b - a))
            z = z0 + (z1 - z0) * ease_io(u)
            img = render_plate(W, H, key, box, mode, bz, z, pan)
            cv = cv * (1 - wgt) + img * wgt

        # ---------- grading per scene
        if t >= 3.7:
            darken(cv, vgrad(W, H, "top"), 0.72)
            darken(cv, vgrad(W, H, "bottom"), 0.6)
        if 12 <= t < 20 + XF:
            darken(cv, vgrad(W, H, "left") if not P else vgrad(W, H, "bottom"), 0.55 if not P else 0.75)
        if 20 - XF <= t < 27 + XF:
            k = win(t, 19.8, 27.2, 0.4, 0.4)
            if P: darken(cv, vgrad(W, H, "bottom"), 0.5 * k)
            cv *= 1 - 0.2 * k
        if 27 <= t < 33 + XF and not P:
            darken(cv, vgrad(W, H, "left"), 0.6)
        if t >= 33 - XF:
            k = ease_io((t - 32.8) / 0.8)
            cv *= 1 - 0.78 * k

        cv *= vignette(W, H)

        # ================= SCENE 1 (0-4) =================
        if t < 4.3:
            out = 1 - ease_io((t - 3.6) / 0.5)
            particles(cv, t, 0.9 * clamp(t / 1.0) * out, s)
            streak(cv, H * (0.40 if P else 0.38), W * (-0.3 + 1.6 * ease_io((t - 0.2) / 1.6)), 0.9 * win(t, 0.2, 1.9, 0.4, 0.5), s)
            la = ease((t - 0.8) / 1.2) * out
            lw = (560 if P else 440) * s
            ly = H * (0.36 if P else 0.30)
            blit_logo(cv, W / 2, ly, lw * (0.92 + 0.08 * ease((t - 0.8) / 2.0)), la,
                      shine=clamp((t - 1.9) / 1.0) if 1.9 < t < 2.9 else None, glow=0.5 * la)
            lh = lw * 575 / 760
            y1 = ly + lh / 2 + 70 * s
            wa = ease((t - 1.5) / 0.9) * out
            blit(cv, T(X("R DYNAMIC PLUS", "آر دايناميك بلس"), "serif", 92 if P else 80, "Bold", 0.08), W / 2, y1 + (1 - wa) * 20 * s, wa,
                 "gold", wipe=ease((t - 1.5) / 1.0), shine=clamp((t - 2.2) / 1.0))
            y2 = y1 + 78 * s
            hline(cv, W / 2, y2, 640 * s, 2 * s, 0.9 * out, ease((t - 1.9) / 0.8))
            blit(cv, T(X("CAR RENTAL  ·  DUBAI", "تأجير السيارات  ·  دبي"), "sans", 28, "Medium", 0.45), W / 2, y2 + 42 * s, ease((t - 2.1) / 0.7) * out, "white")
            ta = ease((t - 2.5) / 0.8) * out
            blit(cv, T(X("EXPERIENCE DUBAI IN STYLE", "عِش دبي بأناقة"), "sans", 46 if P else 40, "Light", 0.22), W / 2,
                 y2 + (170 if P else 140) * s + (1 - ta) * 25 * s, ta, "white", wipe=ease((t - 2.5) / 1.0))

        # ================= SCENE 2 (4-12) =================
        if 4.0 <= t < 12.3:
            out = 1 - ease_io((t - 11.8) / 0.4)
            words = X(["YOUR JOURNEY.", "YOUR STYLE.", "YOUR CAR."], ["رحلتك.", "أسلوبك.", "سيارتك."])
            starts = [4.5, 6.1, 7.7]
            if P:
                for i, (wd, st) in enumerate(zip(words, starts)):
                    a = ease((t - st) / 0.7) * out
                    blit(cv, T(wd, "serif", 78, "Bold", 0.06), W / 2, H * 0.085 + i * 100 * s + (1 - a) * 30 * s, a,
                         "gold" if i == 2 else "white", wipe=ease((t - st) / 0.9), shine=clamp((t - st - 0.5) / 1.2) if i == 2 else None)
                hline(cv, W / 2, H * 0.085 + 2.62 * 100 * s, 300 * s, 2 * s, out, ease((t - 8.2) / 0.8))
            else:
                parts = [T(w_, "serif", 66, "Bold", 0.05) for w_ in words]
                gap = 40 * s
                tot = sum(p["w"] for p in parts) + 2 * gap
                x = W / 2 + tot / 2 if AR else W / 2 - tot / 2
                for i, (p, st) in enumerate(zip(parts, starts)):
                    a = ease((t - st) / 0.7) * out
                    blit(cv, p, x, H * 0.84 + (1 - a) * 25 * s, a, "gold" if i == 2 else "white", wipe=ease((t - st) / 0.9),
                         anchor="r" if AR else "l", shine=clamp((t - st - 0.5) / 1.2) if i == 2 else None)
                    x += -(p["w"] + gap) if AR else (p["w"] + gap)
                hline(cv, W / 2, H * 0.84 + 62 * s, 520 * s, 2 * s, out, ease((t - 8.2) / 0.8))
            # car labels
            for (a_, b_, key, *_r) in shots:
                if key in labels and a_ < 12 and a_ - 0.1 <= t < b_:
                    la = win(t, a_ + 0.15, b_ - 0.05, 0.35, 0.25)
                    lx, ly = (70 * s, H * 0.93) if P else (90 * s, 80 * s)
                    hline(cv, lx + 25 * s, ly, 50 * s, 3 * s, la, 1.0, GOLD_TOP)
                    blit(cv, T(labels[key], "sans", 26, "SemiBold", 0.3), lx + 64 * s, ly, la, "white", anchor="l", shadow=0.8)

        # ================= SCENE 3 (12-20) =================
        if 12.0 <= t < 20.3:
            out = 1 - ease_io((t - 19.8) / 0.4)
            if P:
                ty = H * 0.635
                a = ease((t - 12.3) / 0.8) * out
                blit(cv, T(X("WHAT WE OFFER", "خدماتنا"), "serif", 76, "Bold", 0.1), W / 2, ty + (1 - a) * 25 * s, a, "gold",
                     wipe=ease((t - 12.3) / 1.0), shine=clamp((t - 13.0) / 1.2))
                hline(cv, W / 2, ty + 62 * s, 420 * s, 2 * s, out, ease((t - 12.6) / 0.8))
                pa = ease((t - 12.8) / 0.6) * out
                pcy = ty + 62 * s + 40 * s + 6 * 72 * s / 2 + 30 * s
                panel(cv, W / 2, pcy, 900 * s, 6 * 72 * s + 60 * s, pa, r=int(20 * s), fill_a=0.55, border_a=0.7, bw=max(1, int(2 * s)))
                y0 = pcy - 6 * 72 * s / 2 + 36 * s
                for i, sv in enumerate(services):
                    st = 13.1 + i * 0.4
                    a = ease((t - st) / 0.6) * out
                    yy = y0 + i * 72 * s
                    if AR:
                        diamond(cv, W / 2 + 390 * s, yy, 13 * s, a)
                        blit(cv, T(sv, "sans", 38, "Medium", 0.02), W / 2 + 350 * s - (1 - a) * 30 * s, yy, a, "white", anchor="r")
                    else:
                        diamond(cv, W / 2 - 390 * s, yy, 13 * s, a)
                        blit(cv, T(sv, "sans", 38, "Medium", 0.02), W / 2 - 350 * s + (1 - a) * 30 * s, yy, a, "white", anchor="l")
            else:
                lx = W * 0.47 if AR else 140 * s
                dr = -1 if AR else 1
                a = ease((t - 12.3) / 0.8) * out
                blit(cv, T(X("WHAT WE OFFER", "خدماتنا"), "serif", 76, "Bold", 0.1), lx, H * 0.2 + (1 - a) * 25 * s, a, "gold",
                     wipe=ease((t - 12.3) / 1.0), anchor="r" if AR else "l", shine=clamp((t - 13.0) / 1.2))
                hline(cv, lx + dr * 210 * s, H * 0.2 + 62 * s, 420 * s, 2 * s, out, ease((t - 12.6) / 0.8))
                for i, sv in enumerate(services):
                    st = 13.1 + i * 0.4
                    a = ease((t - st) / 0.6) * out
                    yy = H * 0.2 + 150 * s + i * 92 * s
                    diamond(cv, lx + dr * 14 * s, yy, 14 * s, a)
                    blit(cv, T(sv, "sans", 42, "Medium", 0.02), lx + dr * (56 * s + (1 - a) * 30 * s), yy, a, "white", anchor="r" if AR else "l")

        # ================= SCENE 4 (20-27) =================
        if 20.0 <= t < 27.3:
            out = 1 - ease_io((t - 26.8) / 0.4)
            items = X([("ESTABLISHED SINCE", "2012", None), ("24/7", None, "CUSTOMER SUPPORT"),
                     ("FREE", None, "DELIVERY IN DUBAI"), ("WIDE RANGE", None, "OF LUXURY CARS")],
                     [("منذ عام", "2012", None), ("24/7", None, "خدمة العملاء"),
                     ("مجاناً", None, "التوصيل داخل دبي"), ("تشكيلة واسعة", None, "من السيارات الفاخرة")])
            if P:
                cw, ch = 480 * s, 250 * s
                gx, gy = W / 2, H * 0.73
                centers = [(gx - cw / 2 - 10 * s, gy - ch / 2 - 10 * s), (gx + cw / 2 + 10 * s, gy - ch / 2 - 10 * s),
                           (gx - cw / 2 - 10 * s, gy + ch / 2 + 10 * s), (gx + cw / 2 + 10 * s, gy + ch / 2 + 10 * s)]
            else:
                cw, ch = 520 * s, 300 * s
                gx, gy = W * 0.70, H * 0.5
                centers = [(gx - cw / 2 - 12 * s, gy - ch / 2 - 12 * s), (gx + cw / 2 + 12 * s, gy - ch / 2 - 12 * s),
                           (gx - cw / 2 - 12 * s, gy + ch / 2 + 12 * s), (gx + cw / 2 + 12 * s, gy + ch / 2 + 12 * s)]
            for i, ((top, num, bot), (cx, cy)) in enumerate(zip(items, centers)):
                st = 20.4 + i * 0.75
                a = ease((t - st) / 0.6) * out
                panel(cv, cx, cy + (1 - a) * 30 * s, cw, ch, a, r=int(18 * s), fill_a=0.6, border_a=0.8, bw=max(1, int(2 * s)))
                yy = cy + (1 - a) * 30 * s
                big = 92 if P else 100
                if i == 0:
                    yr = int(round(1990 + 22 * ease((t - st) / 1.3)))
                    blit(cv, T(top, "sans", 26 if not AR else 34, "SemiBold", 0.25), cx, yy - 62 * s, a, "white")
                    blit(cv, T(str(yr), "serif", big, "Bold", 0.04), cx, yy + 28 * s, a, "gold", shine=clamp((t - st - 1.3) / 1.0))
                else:
                    blit(cv, T(top, "serif", big if len(top) < (7 if AR else 6) else int(big * 0.62), "Bold", 0.04), cx, yy - 22 * s, a, "gold",
                         wipe=ease((t - st) / 0.8), shine=clamp((t - st - 0.8) / 1.0))
                    blit(cv, T(bot, "sans", 26 if not AR else 32, "SemiBold", 0.25), cx, yy + 72 * s, a, "white")

        # ================= SCENE 5 (27-33) =================
        if 27.0 <= t < 33.3:
            out = 1 - ease_io((t - 32.8) / 0.4)
            a1 = ease((t - 27.5) / 0.9) * out
            a2 = ease((t - 28.3) / 1.0) * out
            if P:
                by = H * 0.83
                blit(cv, T(X("MAKE EVERY MOMENT IN DUBAI", "اجعل كل لحظة في دبي"), "sans", 42 if not AR else 50, "Medium", 0.18), W / 2, by + (1 - a1) * 25 * s, a1, "white",
                     wipe=ease((t - 27.5) / 1.0), shadow=0.8)
                blit(cv, T(X("UNFORGETTABLE", "لا تنسى"), "serif", 96 if not AR else 110, "Bold", 0.06), W / 2, by + 100 * s + (1 - a2) * 25 * s, a2, "gold",
                     wipe=ease((t - 28.3) / 1.1), shine=clamp((t - 29.4) / 1.4), shadow=0.8)
                hline(cv, W / 2, by + 175 * s, 520 * s, 2 * s, out, ease((t - 29.0) / 0.8))
            else:
                lx = W * 0.44 if AR else 130 * s
                an = "r" if AR else "l"; dr = -1 if AR else 1
                blit(cv, T(X("MAKE EVERY MOMENT", "اجعل كل لحظة"), "sans", 50 if not AR else 58, "Medium", 0.16), lx, H * 0.40 + (1 - a1) * 25 * s, a1, "white",
                     wipe=ease((t - 27.5) / 1.0), anchor=an, shadow=0.8)
                blit(cv, T(X("IN DUBAI", "في دبي"), "sans", 50 if not AR else 58, "Medium", 0.16), lx, H * 0.40 + (70 if not AR else 85) * s + (1 - a1) * 25 * s, a1, "white",
                     wipe=ease((t - 27.7) / 1.0), anchor=an, shadow=0.8)
                blit(cv, T(X("UNFORGETTABLE", "لا تنسى"), "serif", 74 if not AR else 100, "Bold", 0.06), lx, H * 0.40 + (170 if not AR else 200) * s + (1 - a2) * 25 * s, a2, "gold",
                     wipe=ease((t - 28.3) / 1.1), anchor=an, shine=clamp((t - 29.4) / 1.4), shadow=0.8)
                hline(cv, lx + dr * 200 * s, H * 0.40 + (240 if not AR else 285) * s, 400 * s, 2 * s, out, ease((t - 29.0) / 0.8))

        # ================= SCENE 6 (33-40) =================
        if t >= 33.0:
            particles(cv, t, 0.5 * ease((t - 33.2) / 1.0), s)
            def A(st, d=0.6): return ease((t - st) / d)
            if P:
                cx = W / 2
                blit_logo(cv, cx, 300 * s, 330 * s, A(33.2, 0.9), shine=clamp((t - 34.0) / 1.0) if 34 < t < 35 else None, glow=0.4 * A(33.2))
                blit(cv, T(X("R DYNAMIC PLUS", "آر دايناميك بلس"), "serif", 92, "Bold", 0.07), cx, 520 * s, A(33.5), "gold", wipe=A(33.5, 0.9),
                     shine=clamp((t - 36.5) / 1.2))
                hline(cv, cx - 300 * s, 610 * s, 130 * s, 2 * s, A(33.8))
                hline(cv, cx + 300 * s, 610 * s, 130 * s, 2 * s, A(33.8))
                blit(cv, T(X("CAR RENTAL", "لتأجير السيارات"), "serif", 42, "Regular", 0.3), cx, 610 * s, A(33.8), "gold")
                blit(cv, T(X("LUXURY CAR RENTAL IN DUBAI", "تأجير السيارات الفاخرة في دبي"), "sans", 34, "Medium", 0.18), cx, 700 * s, A(34.1), "white")
                pa = A(34.4)
                panel(cv, cx, 815 * s, 820 * s, 100 * s, pa, r=int(50 * s), fill_a=1.0, border_a=0.0, fill=GOLD_MID)
                blit(cv, T(X("BOOK YOUR DREAM CAR TODAY", "احجز سيارة أحلامك اليوم"), "sans", 36 if not AR else 40, "Bold", 0.08), cx, 815 * s, pa, "black", shadow=0)
                blit(cv, T(X("WHATSAPP", "واتساب"), "sans", 28 if not AR else 34, "SemiBold", 0.45), cx, 950 * s, A(34.8), "gold")
                for i, ph in enumerate(phones):
                    a = A(35.0 + i * 0.25)
                    blit(cv, T(ph, "sans", 54, "SemiBold", 0.04), cx + (1 - a) * 30 * s, (1025 + i * 82) * s, a, "white", shadow=0.8)
                hline(cv, cx, 1370 * s, 700 * s, 2 * s, A(36.0), A(36.0, 0.8))
                blit(cv, T(X("LOCATION", "الموقع"), "sans", 26 if not AR else 32, "SemiBold", 0.45), cx, 1425 * s, A(36.2), "gold")
                blit(cv, T(X("Al Garhoud – Airport Street, Dubai, UAE", "القرهود – شارع المطار، دبي، الإمارات"), "sans", 34 if not AR else 36, "Medium", 0.02), cx, 1475 * s, A(36.3), "white")
                blit(cv, T(X("WEBSITE", "الموقع الإلكتروني"), "sans", 26 if not AR else 32, "SemiBold", 0.45), cx, 1560 * s, A(36.5), "gold")
                blit(cv, T("www.rdynamicpluscarrental.com", "sans", 40, "SemiBold", 0.02), cx, 1612 * s, A(36.6), "white")
            else:
                cx = W * 0.27
                blit_logo(cv, cx, 260 * s, 330 * s, A(33.2, 0.9), shine=clamp((t - 34.0) / 1.0) if 34 < t < 35 else None, glow=0.4 * A(33.2))
                blit(cv, T(X("R DYNAMIC PLUS", "آر دايناميك بلس"), "serif", 84, "Bold", 0.07), cx, 470 * s, A(33.5), "gold", wipe=A(33.5, 0.9),
                     shine=clamp((t - 36.5) / 1.2))
                hline(cv, cx - 285 * s, 555 * s, 120 * s, 2 * s, A(33.8))
                hline(cv, cx + 285 * s, 555 * s, 120 * s, 2 * s, A(33.8))
                blit(cv, T(X("CAR RENTAL", "لتأجير السيارات"), "serif", 40, "Regular", 0.3), cx, 555 * s, A(33.8), "gold")
                blit(cv, T(X("LUXURY CAR RENTAL IN DUBAI", "تأجير السيارات الفاخرة في دبي"), "sans", 32, "Medium", 0.18), cx, 640 * s, A(34.1), "white")
                pa = A(34.4)
                panel(cv, cx, 760 * s, 760 * s, 96 * s, pa, r=int(48 * s), fill_a=1.0, border_a=0.0, fill=GOLD_MID)
                blit(cv, T(X("BOOK YOUR DREAM CAR TODAY", "احجز سيارة أحلامك اليوم"), "sans", 34 if not AR else 38, "Bold", 0.08), cx, 760 * s, pa, "black", shadow=0)
                # divider
                dv = A(34.6, 0.8)
                x = int(W * 0.535); y0, y1 = int(H * 0.5 - 380 * s * dv), int(H * 0.5 + 380 * s * dv)
                if y1 > y0: cv[y0:y1, x - 1:x + 1] = cv[y0:y1, x - 1:x + 1] * 0.2 + GOLD_MID * 0.8
                rx = W * 0.76
                blit(cv, T(X("WHATSAPP", "واتساب"), "sans", 28 if not AR else 34, "SemiBold", 0.45), rx, 175 * s, A(34.8), "gold")
                for i, ph in enumerate(phones):
                    a = A(35.0 + i * 0.25)
                    blit(cv, T(ph, "sans", 52, "SemiBold", 0.04), rx + (1 - a) * 30 * s, (250 + i * 78) * s, a, "white", shadow=0.8)
                hline(cv, rx, 590 * s, 560 * s, 2 * s, A(36.0), A(36.0, 0.8))
                blit(cv, T(X("LOCATION", "الموقع"), "sans", 24 if not AR else 30, "SemiBold", 0.45), rx, 650 * s, A(36.2), "gold")
                blit(cv, T(X("Al Garhoud – Airport Street, Dubai, UAE", "القرهود – شارع المطار، دبي، الإمارات"), "sans", 31 if not AR else 32, "Medium", 0.02), rx, 700 * s, A(36.3), "white")
                blit(cv, T(X("WEBSITE", "الموقع الإلكتروني"), "sans", 24 if not AR else 30, "SemiBold", 0.45), rx, 790 * s, A(36.5), "gold")
                blit(cv, T("www.rdynamicpluscarrental.com", "sans", 37, "SemiBold", 0.02), rx, 840 * s, A(36.6), "white")

        # ---------- transitions
        flash(cv, t, cuts, s)
        for c in soft_cuts:
            d = t - c
            if -0.2 < d < 0.3:
                k = math.exp(-(d / 0.1) ** 2)
                streak(cv, H * 0.5, W * (0.2 + (d + 0.2) * 1.8), 0.35 * k, s * 2)
        return np.clip(cv, 0, 1)
    return frame

# ---------------------------------------------------------------- driver
def work(args):
    W, H, f0, f1, outp = args
    fr = build(W, H)
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                          "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "15",
                          "-pix_fmt", "yuv420p", "-profile:v", "high", outp], stdin=subprocess.PIPE)
    for i in range(f0, f1):
        img = fr(i / FPS)
        p.stdin.write((img * 255 + 0.5).astype(np.uint8).tobytes())
    p.stdin.close(); p.wait()
    return outp

if __name__ == "__main__":
    W, H = int(sys.argv[1]), int(sys.argv[2])
    outdir = sys.argv[3]
    if len(sys.argv) > 4:  # stills for review
        fr = build(W, H)
        for ts in sys.argv[4].split(","):
            img = fr(float(ts))
            Image.fromarray((img * 255).astype(np.uint8)).save(f"{outdir}/still_{W}x{H}_{ts}.jpg", quality=90)
        sys.exit()
    N = int(DUR * FPS); K = 8
    jobs = [(W, H, N * k // K, N * (k + 1) // K, f"{outdir}/chunk_{W}x{H}_{k}.mp4") for k in range(K)]
    with Pool(4) as pool:
        parts = pool.map(work, jobs)
    with open(f"{outdir}/list_{W}x{H}.txt", "w") as f:
        for pth in parts: f.write(f"file '{pth}'\n")
    print("done")
