#!/usr/bin/env python3
"""Synthesised cinematic score + sound design, 40 s, 48 kHz stereo WAV."""
import sys, numpy as np, wave
SR = 48000; D = 40.0; N = int(SR * D)
t = np.arange(N) / SR
rng = np.random.default_rng(3)
L = np.zeros(N); R = np.zeros(N)

def env(a, b, att, rel):
    e = np.zeros(N)
    i0, i1 = int(a * SR), min(N, int(b * SR))
    tt = t[i0:i1] - a
    e[i0:i1] = np.clip(tt / att, 0, 1) * np.clip((b - a - tt) / rel, 0, 1)
    return e

def lp(x, fc):  # one-pole lowpass
    a = np.exp(-2 * np.pi * fc / SR)
    from scipy.signal import lfilter
    return lfilter([1 - a], [1, -a], x)

def hz(n): return 440 * 2 ** ((n - 69) / 12)

# --- pad: warm detuned voices, progression Dm | Bb | F | C | Dm (8 s each)
chords = [[50, 57, 62, 65, 69], [46, 53, 58, 62, 65], [41, 53, 57, 60, 65], [48, 55, 60, 64, 67], [50, 57, 62, 65, 69, 74]]
for ci, ch in enumerate(chords):
    a = ci * 8.0 - 0.6; b = a + 9.4
    e = env(max(0, a), min(D, b), 1.5 if ci else 2.5, 1.6)
    if ci == 4: e = env(a, D, 1.0, 2.5)
    for n in ch:
        f = hz(n)
        for det, pan in [(-0.12, 0.3), (0.12, 0.7)]:
            ff = f * 2 ** (det / 12)
            v = sum(np.sin(2 * np.pi * ff * k * t + k) / k ** 1.6 for k in range(1, 6))
            v *= e * 0.035 * (1 + 0.15 * np.sin(2 * np.pi * 0.2 * t + n))
            L += v * (1 - pan); R += v * pan

# --- sub drone
sub = np.sin(2 * np.pi * hz(38) * t) * env(0, D, 3, 2.5) * 0.10
L += sub; R += sub

# --- pulse (kick + ticking arp) from 4 s, BPM 100
bpm = 100; beat = 60 / bpm
def kick(at, g=1.0):
    i0 = int(at * SR); n = int(0.45 * SR)
    if i0 >= N: return
    n = min(n, N - i0); tt = np.arange(n) / SR
    f = 45 + 80 * np.exp(-tt * 30)
    ph = 2 * np.pi * np.cumsum(f) / SR
    v = np.sin(ph) * np.exp(-tt * 7) * 0.55 * g
    L[i0:i0 + n] += v; R[i0:i0 + n] += v
k = 4.0
while k < 39.0:
    in_s = (4 <= k < 33)
    if in_s: kick(k, 0.6 if k < 12 else 0.8)
    k += beat
arp = [62, 65, 69, 74, 69, 65]
k = 12.0; i = 0
while k < 32.8:
    f = hz(arp[i % len(arp)] + 12)
    i0 = int(k * SR); n = int(0.25 * SR); tt = np.arange(n) / SR
    v = (np.sin(2 * np.pi * f * tt) + 0.3 * np.sin(4 * np.pi * f * tt)) * np.exp(-tt * 14) * 0.06
    pan = 0.3 + 0.4 * (i % 2)
    L[i0:i0 + n] += v * (1 - pan); R[i0:i0 + n] += v * pan
    k += beat / 2; i += 1

# --- whooshes into cuts, impacts on cuts
from scipy.signal import butter, sosfilt
def whoosh(end, dur=0.8, g=0.35):
    i0 = int((end - dur) * SR); n = int(dur * SR)
    noise = rng.standard_normal(n)
    out = np.zeros(n); seg = 8
    for j in range(seg):
        a, b = j * n // seg, (j + 1) * n // seg
        fc = 300 + 5000 * (j / seg) ** 2
        sos = butter(2, [fc * 0.6, fc * 1.4], "bandpass", fs=SR, output="sos")
        out[a:b] = sosfilt(sos, noise)[a:b]
    e = np.linspace(0, 1, n) ** 2.5
    v = out * e * g
    pan = np.linspace(0.2, 0.8, n)
    L[i0:i0 + n] += v * (1 - pan); R[i0:i0 + n] += v * pan
def impact(at, g=1.0):
    i0 = int(at * SR); n = min(int(2.5 * SR), N - i0); tt = np.arange(n) / SR
    f = 30 + 60 * np.exp(-tt * 8)
    boom = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-tt * 2.2) * 0.7
    sos = butter(2, 1800, "lowpass", fs=SR, output="sos")
    crack = sosfilt(sos, rng.standard_normal(n)) * np.exp(-tt * 9) * 0.25
    v = (boom + crack) * g
    L[i0:i0 + n] += v; R[i0:i0 + n] += v
def shimmer(at, g=0.12):
    i0 = int(at * SR); n = int(2.2 * SR); tt = np.arange(n) / SR
    v = np.zeros(n)
    for m, fr in enumerate([1568, 2093, 2637, 3136]):
        v += np.sin(2 * np.pi * fr * tt + m) * np.exp(-tt * (1.8 + m * 0.4)) * np.clip((tt - m * 0.07) * 50, 0, 1)
    v *= g
    L[i0:i0 + n] += v * 0.6; R[i0:i0 + n] += v * 0.4 + np.roll(v, 600) * 0.2

whoosh(1.0, 1.0, 0.25); impact(1.0, 0.9); shimmer(1.9)
for c in [4.0, 12.0, 20.0, 27.0]:
    whoosh(c, 0.7, 0.3); impact(c, 0.7)
for c in [5.6, 7.2, 8.8, 10.4, 16.0, 30.4]:
    whoosh(c, 0.4, 0.15)
whoosh(33.0, 1.0, 0.35); impact(33.0, 1.0); shimmer(34.0, 0.10)

# --- master: gentle fade, soft clip, normalise
fade = np.clip((D - t) / 1.8, 0, 1) ** 1.5
L *= fade; R *= fade
mx = max(np.abs(L).max(), np.abs(R).max())
L = np.tanh(L / mx * 1.4) / np.tanh(1.4) * 0.89; R = np.tanh(R / mx * 1.4) / np.tanh(1.4) * 0.89
st = (np.stack([L, R], 1) * 32767).astype(np.int16)
with wave.open(sys.argv[1], "wb") as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes(st.tobytes())
print("ok")
