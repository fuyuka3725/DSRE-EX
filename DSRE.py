import os
import sys
import traceback
from typing import Optional

import subprocess
import soundfile as sf
import tempfile
import json

import numpy as np
from scipy import signal

from PySide6 import QtCore, QtWidgets
from PySide6.QtGui import QIcon, QTextCursor, QDragEnterEvent, QDropEvent, QKeySequence, QAction

# ======== Config Dir Taget ========
def get_config_path(filename: str) -> str:
    if getattr(sys, 'frozen', False):
        base = os.path.dirname(sys.executable)
    else:
        base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, filename)

# ======== Hide CLI ========
def get_subprocess_kwargs() -> dict:
    kwargs = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = 0x08000000
    return kwargs

# ======== FFmpeg ========
def add_ffmpeg_path(relative: str) -> str:
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, relative)

add_ffmpeg_path("ffmpeg.exe")
def cmdrun(cmd, worker=None, **kw):
    stdout = kw.pop("stdout", None)
    stderr = kw.pop("stderr", None)
    check  = kw.pop("check", False)

    if kw.pop("capture_output", False):
        stdout = subprocess.PIPE
        stderr = subprocess.PIPE

    proc = subprocess.Popen(
        cmd,
        stdout=stdout,
        stderr=stderr,
        **get_subprocess_kwargs(),
        **kw
    )

    if worker is not None:
        worker._current_proc = proc

    returncode = proc.wait()

    if worker is not None:
        worker._current_proc = None

    if check and returncode != 0:
        raise subprocess.CalledProcessError(returncode, cmd)

    proc.returncode = returncode
    return proc

def lossless_headroom(data, drive=0.9, target_peak_db=-0.5):
    scaled_data = data * drive
    limited_data = np.tanh(scaled_data)
    target_peak_linear = 10 ** (target_peak_db / 20)
    data = data * target_peak_linear
    return data

def save_wav24_out(in_path, y_out, sr, out_path, worker=None, fmt="FLAC"):
    import tempfile, subprocess, numpy as np, soundfile as sf, os

    # Check shape = (n, ch)
    if y_out.ndim == 1:
        data = y_out[:, None]
    else:
        data = y_out.T if y_out.shape[0] < y_out.shape[1] else y_out

    data = data.astype(np.float32, copy=False)
    data = lossless_headroom(data)

    tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp_wav.close()
    sf.write(tmp_wav.name, data, sr, subtype="FLOAT")

    fmt = fmt.upper()
    out_path = os.path.splitext(out_path)[0] + (".m4a" if fmt == "ALAC" else ".flac")

    codec_map = {"FLAC": "flac", "ALAC": "alac"}
    sample_fmt_map = {"FLAC": "s32", "ALAC": "s32p"}

    if fmt == "ALAC":
        cmd = [
            "ffmpeg.exe", "-y",
            "-i", tmp_wav.name,
            "-i", in_path,
            "-map", "0:a",
            "-map", "1:v?",
            "-map_metadata", "1",
            "-c:a", codec_map[fmt],
            "-sample_fmt", sample_fmt_map[fmt],
            "-c:v", "copy",
            out_path
        ]
    elif fmt == "FLAC":
        cover_tmp = None
        try:
            cover_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            cover_tmp.close()
            cmdrun(
                ["ffmpeg.exe", "-y", "-i", in_path, "-an", "-c:v", "copy", cover_tmp.name],
                worker=worker, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        except Exception:
            cover_tmp = None

        if cover_tmp and os.path.exists(cover_tmp.name):
            cmd = [
                "ffmpeg.exe", "-y",
                "-i", tmp_wav.name,
                "-i", in_path,
                "-i", cover_tmp.name,
                "-map", "0:a",
                "-map", "2:v",
                "-disposition:v", "attached_pic",
                "-map_metadata", "1",
                "-c:a", codec_map[fmt],
                "-sample_fmt", sample_fmt_map[fmt],
                "-c:v", "copy",
                out_path
            ]
        else:
            cmd = [
                "ffmpeg.exe", "-y",
                "-i", tmp_wav.name,
                "-i", in_path,
                "-map", "0:a",
                "-map_metadata", "1",
                "-c:a", codec_map[fmt],
                "-sample_fmt", sample_fmt_map[fmt],
                out_path
            ]

    cmdrun(cmd, worker=worker, check=True)
    os.remove(tmp_wav.name)
    if fmt == "FLAC" and cover_tmp and os.path.exists(cover_tmp.name):
        os.remove(cover_tmp.name)

    return out_path

def load_audio(in_path: str):
    import soundfile as sf
    try:
        y, sr = sf.read(in_path, always_2d=True)
        return y.T.astype(np.float32), sr
    except Exception:
        pass

    fd, tmp_wav = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        cmdrun(
            [add_ffmpeg_path("ffmpeg.exe"), "-y", "-i", in_path,
             "-f", "wav", "-acodec", "pcm_f32le", tmp_wav],
            check=True, capture_output=True,
        )
        y, sr = sf.read(tmp_wav, always_2d=True)
        return y.T.astype(np.float32), sr
    finally:
        os.unlink(tmp_wav)

# ======== DSP：SSB ========
def freq_shift_mono(x: np.ndarray, f_shift: float, d_sr: float) -> np.ndarray:
    N_orig = len(x)
    x_f32 = x.astype(np.float32)
    poles_A = [0.4794008, 0.8762184, 0.9765975, 0.9971492]
    poles_B = [0.1617585, 0.7330289, 0.9453499, 0.9927737]
    x_A = x_f32.copy()
    x_B = x_f32.copy()
    for p in poles_A:
        b = [p, 1.0]
        a = [1.0, p]
        x_A = signal.lfilter(b, a, x_A)
    for p in poles_B:
        b = [p, 1.0]
        a = [1.0, p]
        x_B = signal.lfilter(b, a, x_B)
    t = np.arange(0, N_orig, dtype=np.float32)
    omega = 2.0 * np.pi * f_shift * d_sr
    cos_factor = np.cos(omega * t)
    sin_factor = np.sin(omega * t)
    result = (x_A * cos_factor) - (x_B * sin_factor)
    return result.astype(np.float32)

def freq_shift_multi(x: np.ndarray, f_shift: float, d_sr: float) -> np.ndarray:
    return np.asarray(
        [freq_shift_mono(x[i], f_shift, d_sr) for i in range(len(x))],
        dtype=np.float32
    )

# AutoHPF
def auto_hp_params(y: np.ndarray, sr: int):
    mono = y.mean(axis=0) if y.ndim > 1 else y

    chunk = sr // 2
    chunks = [mono[i:i+chunk] for i in range(0, len(mono)-chunk, chunk)]

    if not chunks:
        active_signal = mono
    else:
        energies = [np.mean(c**2) for c in chunks]
        threshold_e = np.percentile(energies, 70)
        active = [c for c, e in zip(chunks, energies) if e >= threshold_e]
        active_signal = np.concatenate(active) if active else mono
    freqs = np.fft.rfftfreq(len(active_signal), 1 / sr)
    mag = np.abs(np.fft.rfft(active_signal))

    peak = mag.max()
    threshold_high = peak * 10 ** (-40 / 20)

    search_limit = int(len(freqs) * 0.95)
    cutoff_idx = search_limit

    for idx in range(search_limit, 0, -1):
        if mag[idx] >= threshold_high:
            cutoff_idx = idx
            break

    cutoff_hz = freqs[cutoff_idx] if cutoff_idx > 0 else sr / 2
    post_hp = float(np.clip(cutoff_hz + 1000, 10000, 20000))

    noise_floor = np.percentile(mag, 10)
    snr_threshold = noise_floor * 10 ** (20 / 20)

    stable_hz = 1000.0
    for fq in np.arange(1000, cutoff_hz, 500):
        idx_lo = np.searchsorted(freqs, fq)
        idx_hi = np.searchsorted(freqs, fq + 500)
        band_mag = mag[idx_lo:idx_hi]
        if len(band_mag) == 0:
            continue
        if band_mag.mean() >= snr_threshold:
            stable_hz = fq

    min_pre_hp = max(5000, cutoff_hz * 0.3)
    pre_hp = float(np.clip(stable_hz * 0.75, min_pre_hp, 16000))

    return pre_hp, post_hp

# Auto Params
def auto_zansei_params(y: np.ndarray, sr: int, pre_hp: float, post_hp: float):
    mono = y.mean(axis=0) if y.ndim > 1 else y

    if len(mono) < sr * 2:
        return 7, 1.10

    freqs = np.fft.rfftfreq(len(mono), 1 / sr)
    mag = np.abs(np.fft.rfft(mono))

    total_power = np.sum(mag**2) + 1e-30

    noise_floor = np.percentile(mag, 20)
    mag_clean = np.maximum(mag - noise_floor, 0)

    cumsum = np.cumsum(mag_clean)
    total_energy = cumsum[-1]

    if total_energy < 1e-10:
        return 8, 1.10

    nyquist = sr / 2
    rolloff_95 = freqs[np.searchsorted(cumsum, total_energy * 0.95)]
    rolloff_85 = freqs[np.searchsorted(cumsum, total_energy * 0.85)]

    ratio_95 = rolloff_95 / nyquist
    ratio_85 = rolloff_85 / nyquist

    #Scale-independent high-frequency band analysis
    hf_idx_lo = np.searchsorted(freqs, nyquist * 0.65)
    hf_idx_hi = np.searchsorted(freqs, nyquist * 0.97)
    hf_slice = slice(hf_idx_lo, hf_idx_hi)
    hf_mag = mag[hf_slice]
    hf_len = len(hf_mag)

    if hf_len < 8:
        no_hf_signal = True
        flatness = 0.0
        slope = -18.0
        hf_relative_energy = 0.0
        hf_noise_ratio = 0.0
    else:
        mid_low_idx = np.searchsorted(freqs, nyquist * 0.50)
        mid_low_power = np.sum(mag[:mid_low_idx]**2) + 1e-20
        hf_power = np.sum(hf_mag**2)
        hf_power_safe = hf_power + 1e-25
        hf_relative_energy = hf_power_safe / mid_low_power

        hf_db = 20 * np.log10(np.maximum(hf_mag, 1e-12))
        if len(hf_db) >= 10:
            p80 = np.percentile(hf_db, 80)
            p20 = np.percentile(hf_db, 20)
            flatness_proxy = (p80 - p20) / 60.0
            flatness = np.clip(flatness_proxy, 0.0, 1.0)
        else:
            flatness = 0.0

        x = np.linspace(0, 1, hf_len)
        slope_per_unit = np.polyfit(x, hf_db, 1)[0]
        freq_ratio_log = np.log2((freqs[hf_idx_hi-1] / freqs[hf_idx_lo] + 1e-8))
        slope = slope_per_unit * freq_ratio_log if freq_ratio_log > 0 else -20.0

        noise_floor_hf = np.percentile(hf_mag, 20) if hf_len > 10 else 0
        hf_clean = np.maximum(hf_mag - noise_floor_hf, 0)
        hf_mean_clean = float(np.mean(hf_clean))
        hf_mean_raw = float(np.mean(hf_mag))
        hf_noise_ratio = hf_mean_clean / (hf_mean_raw + 1e-12)

        has_significant_hf = (
            (flatness > 0.15 and slope < -1.5)
            or (flatness > 0.25)
            or (slope < -3.0)
        )
        no_hf_signal = not has_significant_hf

    # Overall
    if no_hf_signal:
        if hf_noise_ratio < 0.1:
            m, decay = 8, 0.50
        elif ratio_95 > 0.5:
            m, decay = 10, 0.55
        else:
            m, decay = 12, 0.60
    elif flatness > 0.3 and slope > -2.0:
        m, decay = 4, 0.40
    elif flatness > 0.2 and slope > -3.0:
        m, decay = 10, 0.55
    elif flatness > 0.15 and slope > -4.0:
        m, decay = 12, 0.60
    else:
        m, decay = 14, 0.75

    return m, decay

def crossover_wiener(
    x: np.ndarray,
    sr: int,
    post_hp: float,
    src_nyquist: float,
    floor: float = 0.02,
    frame_ms: float = 30.0,
    noise_percentile: float = 5.0,
) -> np.ndarray:
    is_1d = x.ndim == 1
    x_2d  = x[np.newaxis, :] if is_1d else x
    out   = x_2d.copy()

    frame  = max(1, int(sr * frame_ms / 1000))
    n_fft  = frame
    freqs  = np.fft.rfftfreq(n_fft, 1.0 / sr)

    band_lo = np.searchsorted(freqs, post_hp)
    band_hi = np.searchsorted(freqs, src_nyquist)
    if band_hi <= band_lo:
        return x

    for ch in range(x_2d.shape[0]):
        sig = x_2d[ch].astype(np.float64)
        n   = len(sig)

        frames = [sig[i:i + frame] for i in range(0, n - frame, frame // 2)]
        if not frames:
            continue

        specs  = [np.fft.rfft(f * np.hanning(len(f)), n=n_fft) for f in frames]
        energies = [float(np.mean(np.abs(s[band_lo:band_hi]) ** 2)) for s in specs]

        threshold = np.percentile(energies, noise_percentile)
        noise_frames = [s for s, e in zip(specs, energies) if e <= threshold]
        if not noise_frames:
            noise_frames = [specs[int(np.argmin(energies))]]

        noise_psd = np.mean(np.abs(np.array(noise_frames)) ** 2, axis=0)  # (n_bins,)

        X_full = np.fft.rfft(sig, n=n)
        freqs_full = np.fft.rfftfreq(n, 1.0 / sr)
        band_lo_f = np.searchsorted(freqs_full, post_hp)
        band_hi_f = np.searchsorted(freqs_full, src_nyquist)

        noise_psd_full = np.interp(
            freqs_full[band_lo_f:band_hi_f],
            freqs,
            noise_psd
        )

        X_band = X_full[band_lo_f:band_hi_f]
        sig_psd = np.abs(X_band) ** 2

        # 1-Pass
        SNR1 = sig_psd / (noise_psd_full + 1e-12)
        G1   = SNR1 / (SNR1 + 1.0)
        G1   = np.maximum(G1, floor)

        # 2-Pass
        residual_psd   = np.abs(X_band * (1.0 - G1)) ** 2
        noise_psd2     = noise_psd_full + residual_psd
        SNR2           = (np.abs(X_band * G1) ** 2) / (noise_psd2 + 1e-12)
        G2             = SNR2 / (SNR2 + 1.0)
        G2             = np.maximum(G2, floor)

        G_full = np.ones(len(X_full), dtype=np.float64)
        G_full[band_lo_f:band_hi_f] = G2

        out[ch] = np.fft.irfft(X_full * G_full, n=n).astype(np.float32)

    return out[0] if is_1d else out

# ======== EC-BWE: Envelope Shaping ========
def lpc_short_term_rms(x: np.ndarray, sr: int, frame_ms: float = 4.0) -> np.ndarray:
    frame = max(1, int(sr * frame_ms / 1000))
    x2  = x.astype(np.float64) ** 2
    pad = np.pad(x2, (frame // 2, frame - frame // 2), mode='edge')
    cs  = np.cumsum(pad)
    return np.sqrt(np.maximum((cs[frame:] - cs[:-frame]) / frame, 0.0)).astype(np.float32)

def envelope_shaping(d_res: np.ndarray, x: np.ndarray,
                      sr: int, post_hp: float, src_nyquist: float,
                      hf_ratio: float, shaping_strength: float = 0.8,
                      env_frame_ms: float = 4.0) -> np.ndarray:
    nyq   = sr / 2.0
    ref_lo = np.clip(post_hp      / nyq,        1e-4, 0.999)
    ref_hi = np.clip(src_nyquist  / nyq * 0.97, ref_lo + 1e-4, 0.999)

    if ref_hi <= ref_lo:
        return d_res

    sos_ref = signal.butter(4, [ref_lo, ref_hi], 'bandpass', output='sos')

    is_1d    = d_res.ndim == 1
    x_2d     = x[np.newaxis, :]     if is_1d else x
    d_res_2d = d_res[np.newaxis, :] if is_1d else d_res
    out      = np.zeros_like(d_res_2d)

    shaping_strength = float(np.clip(0.6 + hf_ratio * 3.0, 0.6, 0.95))

    for ch in range(d_res_2d.shape[0]):
        ref_env = lpc_short_term_rms(
            signal.sosfiltfilt(sos_ref, x_2d[ch]), sr, frame_ms=env_frame_ms)
        gen_env = lpc_short_term_rms(d_res_2d[ch], sr, frame_ms=env_frame_ms)

        raw_gain = ref_env / (gen_env + 1e-7)

        sf_  = max(1, int(sr * 20.0 / 1000))
        pad  = np.pad(raw_gain, (sf_ // 2, sf_ - sf_ // 2), mode='edge')
        cs   = np.cumsum(pad.astype(np.float64))
        gain = ((cs[sf_:] - cs[:-sf_]) / sf_).astype(np.float32)
        gain = np.clip(gain, 0.0, 2.0)

        gain = gain * shaping_strength + 1.0 * (1.0 - shaping_strength)
        out[ch] = (d_res_2d[ch] * gain).astype(np.float32)

    return out[0] if is_1d else out

def zansei_impl(
    x: np.ndarray,
    sr: int,
    m: int = 0,
    decay: float = 0.00,
    src_sr: Optional[int] = None,
    progress_cb=None,
    abort_cb=None,
) -> np.ndarray:

    analysis_sr = src_sr if src_sr is not None else sr
    pre_hp, post_hp = auto_hp_params(x, analysis_sr)

    if m == 0 or decay == 0.00:
        m, decay = auto_zansei_params(x, analysis_sr, pre_hp, post_hp)

    # Pre-processing HPF
    sos = signal.butter(9, pre_hp / (sr / 2), 'highpass', output='sos')
    d_src = signal.sosfiltfilt(sos, x)
    d_src = crossover_wiener(d_src, sr, post_hp, analysis_sr / 2.0, floor=0.02)

    d_sr = 1.0 / sr
    f_dn = freq_shift_mono if (x.ndim == 1) else freq_shift_multi
    d_res = np.zeros_like(x)

    for i in range(m):
        if abort_cb and abort_cb():
            break
        shift_hz = sr * (i + 1) / (m * 2.0)
        d_res += f_dn(d_src, shift_hz, d_sr) * np.exp(-(i + 1) * decay)
        if progress_cb:
            progress_cb(i + 1, m)

    # Post-processing HPF
    sos = signal.butter(8, post_hp / (sr / 2), 'highpass', output='sos')
    d_res = signal.sosfiltfilt(sos, d_res)

    adp_power = float(np.mean(np.abs(d_res)))
    src_power = float(np.mean(np.abs(x)))

    if adp_power < 1e-10:
        return x

    sos_hf = signal.butter(4, post_hp / (sr / 2), 'highpass', output='sos')
    x_hf = signal.sosfiltfilt(sos_hf, x)
    src_hf_power = max(float(np.mean(np.abs(x_hf))), src_power * 0.01)
    hf_ratio = src_hf_power / (src_power + 1e-12)
    hf_scale = float(np.clip(hf_ratio * 20.0, 0.05, 1.0))

    # A-weighting
    freqs = np.fft.rfftfreq(x.shape[-1], 1 / sr)
    freqs[0] = 1e-10
    f2 = freqs ** 2
    f4 = freqs ** 4
    aw = (12194 ** 2 * f4) / (
        (f2 + 20.6 ** 2) *
        np.sqrt((f2 + 107.7 ** 2) * (f2 + 737.9 ** 2)) *
        (f2 + 12194 ** 2)
    )
    aw = aw / (aw.max() + 1e-10)

    suppress_mask = np.ones_like(aw)
    mid_idx_lo = np.searchsorted(freqs, 2000)
    mid_idx_hi = np.searchsorted(freqs, 8000)
    suppress_mask[mid_idx_lo:mid_idx_hi] = 1.0 - aw[mid_idx_lo:mid_idx_hi] * 0.7

    if x.ndim == 1:
        x_band_limited = np.fft.irfft(np.fft.rfft(x) * suppress_mask, n=x.shape[-1])
        d_res_masked = np.fft.irfft(np.fft.rfft(d_res) * suppress_mask, n=d_res.shape[-1])
    else:
        x_band_limited = np.zeros_like(x)
        d_res_masked = np.zeros_like(d_res)
        for ch in range(x.shape[0]):
            x_band_limited[ch] = np.fft.irfft(np.fft.rfft(x[ch]) * suppress_mask, n=x.shape[-1])
            d_res_masked[ch] = np.fft.irfft(np.fft.rfft(d_res[ch]) * suppress_mask, n=d_res.shape[-1])

    adp_power_corrected = float(np.mean(np.abs(d_res_masked)))
    src_power_corrected = float(np.mean(np.abs(x_band_limited)))

    if adp_power_corrected < 1e-10:
        return x

    adj_factor = (src_power_corrected / adp_power_corrected) * 0.10 * hf_scale
    adj_factor = min(adj_factor, 0.5)

    src_nyquist = analysis_sr / 2.0
    d_res_masked = envelope_shaping(
        d_res_masked, x, sr, post_hp, src_nyquist, hf_ratio)

    y = x + d_res_masked * adj_factor
    return y

# ======== Resampler : ARDFTSRC ========
def resample_ardftsrc(
    y: np.ndarray,
    sr_in: int,
    sr_out: int,
    bit_depth: int = 32,
    quality: int = 8192,
    bandwidth: float = 0.999,
) -> np.ndarray:

    import scipy.fft as fft
    import math

    if sr_in == sr_out:
        return y

    is_1d = y.ndim == 1
    if is_1d:
        y = y[np.newaxis, :]

    x = y.T

    gcd = math.gcd(sr_in, sr_out)
    in_nb_samples = sr_in // gcd
    out_nb_samples = sr_out // gcd

    min_samples = max(in_nb_samples, out_nb_samples)
    target_chunk_size = max(quality, sr_in // 4)
    scale_up = max(
        math.ceil(quality / min_samples),
        math.ceil(target_chunk_size / min_samples)
    )
    in_nb_samples *= scale_up
    out_nb_samples *= scale_up

    if in_nb_samples % 2 != 0:
        in_nb_samples += 1
    if out_nb_samples % 2 != 0:
        out_nb_samples += 1

    in_rdft_size = in_nb_samples * 2
    out_rdft_size = out_nb_samples * 2

    taper_size = in_rdft_size // 2 + 1

    in_offset = (in_rdft_size - in_nb_samples) // 2
    tr_nb_samples = min(in_nb_samples, out_nb_samples)
    taper_samples = round(math.ceil(tr_nb_samples * (1.0 - bandwidth)))
    scale = out_nb_samples / in_nb_samples

    size = x.shape[0]
    pad_size = size % in_nb_samples
    if pad_size > 0:
        pad_size = in_nb_samples - pad_size
        x = np.pad(x, ((0, pad_size), (0, 0)), 'constant')

    num_chunks = x.shape[0] // in_nb_samples

    # Sigmoid taper Modulation
    taper = np.zeros(taper_size, dtype=complex)
    for idx in range(taper_size):
        if idx < tr_nb_samples - taper_samples:
            taper[idx] = 1.0
        elif idx < tr_nb_samples - 1:
            n = float(idx - (tr_nb_samples - taper_samples))
            t = float(taper_samples)
            zbk = t / ((t - n) - 1.0) - t / (n + 1.0)
            v = 1.0 / (math.exp(zbk) + 1.0)
            taper[idx] = v
        else:
            taper[idx] = 0.0

    prev_chunk = np.zeros((out_nb_samples, x.shape[1]))

    fd, tmp_path = tempfile.mkstemp(suffix=".bin")
    os.close(fd)

    try:
        chunks_written = 0
        with open(tmp_path, 'wb') as f:
            for i in range(num_chunks):
                x_chunk = x[i * in_nb_samples:(i + 1) * in_nb_samples].astype(np.float64)

                chunk = np.pad(x_chunk,
                    ((in_offset, in_offset), (0, 0)), 'constant')
                chunk = fft.rfft(chunk, n=in_rdft_size, axis=0)
                chunk = chunk * taper[:, np.newaxis]

                if out_rdft_size >= in_rdft_size:
                    chunk = np.pad(chunk,
                        ((0, out_rdft_size - in_rdft_size), (0, 0)),
                        'constant')
                else:
                    chunk = chunk[:out_rdft_size // 2 + 1]

                chunk = fft.irfft(chunk, n=out_rdft_size, axis=0)

                current_chunk = chunk[:out_nb_samples] + prev_chunk
                current_chunk *= scale

                if i > 0:
                    f.write(current_chunk.astype(np.float64).tobytes())
                    chunks_written += 1

                prev_chunk = chunk[out_nb_samples:]

        total_samples = chunks_written * out_nb_samples
        with open(tmp_path, 'rb') as f:
            result = np.frombuffer(
                f.read(),
                dtype=np.float64
            ).reshape(total_samples, x.shape[1])

    finally:
        os.unlink(tmp_path)

    y_out = result.T.astype(np.float32)
    return y_out[0] if is_1d else y_out

def bde_probe_bit_depth(in_path: str) -> int:
    SF_SUBTYPE_BITS = {
        "PCM_S8":  8,  "PCM_U8":  8,
        "PCM_16": 16,
        "PCM_24": 24,
        "PCM_32": 32,
        "FLOAT":  32,
        "DOUBLE": 64,
    }
    try:
        info = sf.info(in_path)
        subtype = info.subtype.upper()
        for key, bits in SF_SUBTYPE_BITS.items():
            if subtype.startswith(key):
                return bits
    except Exception:
        pass

    try:
        cmd = [
            add_ffmpeg_path("ffprobe.exe"),
            "-v", "quiet",
            "-print_format", "json",
            "-show_streams",
            in_path,
        ]
        proc = subprocess.run(
            cmd, capture_output=True, text=True,
            **get_subprocess_kwargs()
        )
        if proc.returncode == 0:
            data = json.loads(proc.stdout)
            for stream in data.get("streams", []):
                if stream.get("codec_type") != "audio":
                    continue
                bprs = int(stream.get("bits_per_raw_sample", 0) or 0)
                if bprs >= 16:
                    return bprs
                bps = int(stream.get("bits_per_sample", 0) or 0)
                if bps >= 16:
                    return bps
                return 0
    except Exception:
        pass

    return 0

# Spectral Detail Synthesis
def bde_stft(x: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    x64 = x.astype(np.float64)
    window = np.hanning(n_fft).astype(np.float64)
    n_frames = (len(x64) - n_fft) // hop + 1
    frames = np.lib.stride_tricks.as_strided(
        x64,
        shape=(n_frames, n_fft),
        strides=(x64.strides[0] * hop, x64.strides[0])
    ).copy()
    return np.fft.rfft(frames * window, axis=1).T

def bde_istft(S: np.ndarray, n_fft: int, hop: int, length: int) -> np.ndarray:
    window   = np.hanning(n_fft).astype(np.float64)
    win_sq   = window ** 2
    n_frames = S.shape[1]
    out      = np.zeros(length + n_fft, dtype=np.float64)
    win_sum  = np.zeros_like(out)
    frames    = np.fft.irfft(S.T, n=n_fft, axis=1) * window
    positions = np.arange(n_frames) * hop
    for i, pos in enumerate(positions):
        end = pos + n_fft
        if end > len(out):
            end = len(out)
            out[pos:end]     += frames[i, :end - pos]
            win_sum[pos:end] += win_sq[:end - pos]
        else:
            out[pos:end]     += frames[i]
            win_sum[pos:end] += win_sq
    win_sum = np.maximum(win_sum, 1e-8)
    return (out / win_sum)[:length].astype(np.float32)

def bde_detect_smearing(
    mag: np.ndarray,
    freqs: np.ndarray,
    sr: int,
    cutoff_hz: float,
) -> np.ndarray:
    cutoff_idx = int(np.searchsorted(freqs, cutoff_hz))
    if cutoff_idx < 8:
        return np.zeros(len(freqs), dtype=np.float32)
    mag_band = mag[:cutoff_idx]
    mean_mag = np.mean(mag_band, axis=1) + 1e-12
    diff      = np.abs(np.diff(mean_mag, prepend=mean_mag[1]))
    diff_norm = diff / (np.max(diff) + 1e-12)
    freq_weight = np.clip((freqs[:cutoff_idx] - 1000.0) / 4000.0, 0.0, 1.0)
    smear = (1.0 - diff_norm) * freq_weight
    mask = np.zeros(len(freqs), dtype=np.float32)
    mask[:cutoff_idx] = smear.astype(np.float32)
    return mask

def bde_harmonic_profile(
    mag: np.ndarray,
    freqs: np.ndarray,
    sr: int,
) -> np.ndarray:
    if mag.ndim == 1:
        mean_mag = mag.astype(np.float64) + 1e-12
    else:
        mean_mag = np.mean(mag, axis=1).astype(np.float64) + 1e-12
    log_mag = np.log(mean_mag)
    n_cep    = (len(log_mag) - 1) * 2
    cepstrum = np.fft.irfft(log_mag, n=n_cep)
    lifter   = np.zeros(n_cep, dtype=np.float64)
    cutoff_q = max(4, n_cep // 32)
    lifter[:cutoff_q]  = 1.0
    lifter[-cutoff_q:] = 1.0
    envelope = np.exp(np.real(np.fft.rfft(lifter * cepstrum))[:len(mean_mag)])
    envelope = np.maximum(envelope, 1e-12)
    harmonic_structure  = mean_mag / envelope
    harmonic_structure /= (harmonic_structure.max() + 1e-12)
    return harmonic_structure.astype(np.float32)

def bde_subband_detail_synth(
    S: np.ndarray,
    mag: np.ndarray,
    phase: np.ndarray,
    freqs: np.ndarray,
    smear_mask: np.ndarray,
    harmonic_profile: np.ndarray,
    cutoff_idx: int,
    strength: float,
) -> np.ndarray:
    from scipy.ndimage import uniform_filter1d
    S_out = S.copy()
    if cutoff_idx < 2:
        return S_out
    mag_band   = mag[:cutoff_idx]
    phase_band = phase[:cutoff_idx]
    neighbor_amp = uniform_filter1d(
        mag_band, size=7, axis=0, mode='nearest') + 1e-12
    h_prof = harmonic_profile[:cutoff_idx, np.newaxis]
    smear  = smear_mask[:cutoff_idx, np.newaxis]
    target_amp = neighbor_amp * h_prof
    active = (smear_mask[:cutoff_idx] >= 0.05)[:, np.newaxis]
    correction = (target_amp - mag_band) * smear * strength * active
    max_corr   = mag_band * 0.5
    correction = np.clip(correction, -max_corr, max_corr)
    new_amp = np.maximum(mag_band + correction, 0.0)
    S_out[:cutoff_idx] = new_amp * np.exp(1j * phase_band)
    return S_out

def bde_spectral_detail_synth(
    y: np.ndarray,
    sr: int,
    cutoff_hz: float,
    strength: float = 0.16,
) -> np.ndarray:
    is_1d = y.ndim == 1
    y_2d  = y[np.newaxis, :] if is_1d else y.copy()
    n_ch, n_samples = y_2d.shape
    n_fft      = max(512, 2 ** int(np.ceil(np.log2(sr * 0.023))))
    hop        = n_fft // 4
    freqs      = np.fft.rfftfreq(n_fft, 1.0 / sr)
    cutoff_idx = int(np.searchsorted(freqs, cutoff_hz))
    fade_lo   = max(0, int(np.searchsorted(freqs, cutoff_hz - 500.0)))
    fade_hi   = min(len(freqs), int(np.searchsorted(freqs, cutoff_hz + 500.0)))
    fade_len  = max(1, fade_hi - fade_lo)
    fade_curve = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
    y_out = np.zeros_like(y_2d)
    for ch in range(n_ch):
        sig   = y_2d[ch].astype(np.float64)
        S     = bde_stft(sig, n_fft, hop)
        mag   = np.abs(S).astype(np.float64)
        phase = np.angle(S).astype(np.float64)
        smear  = bde_detect_smearing(mag, freqs, sr, cutoff_hz)
        h_prof = bde_harmonic_profile(mag[:cutoff_idx + 1], freqs[:cutoff_idx + 1], sr)
        h_prof_full = np.zeros(len(freqs), dtype=np.float32)
        h_prof_full[:len(h_prof)] = h_prof
        S_synth   = bde_subband_detail_synth(
            S, mag, phase, freqs,
            smear, h_prof_full,
            cutoff_idx, strength,
        )
        S_blended = S_synth.copy()
        S_blended[fade_lo:fade_hi] = (
            S_synth[fade_lo:fade_hi] * fade_curve[:, np.newaxis]
            + S[fade_lo:fade_hi]     * (1.0 - fade_curve[:, np.newaxis])
        )
        S_blended[fade_hi:] = S[fade_hi:]
        y_out[ch] = bde_istft(S_blended, n_fft, hop, n_samples)
    return y_out[0] if is_1d else y_out

# Anti-Staircase
def bde_time_domain(y: np.ndarray, sr: int) -> np.ndarray:
    from scipy.signal import butter, lfilter
    GATE_THRESH = 1e-3
    BLEND_COEFF = 0.2
    b, a = butter(2, 0.9, btype='low')
    is_1d = y.ndim == 1
    y_2d  = y[np.newaxis, :] if is_1d else y.copy()
    n_ch, n_samples = y_2d.shape
    y_out = np.empty_like(y_2d)
    for ch in range(n_ch):
        sig      = y_2d[ch].astype(np.float64)
        y_smooth = lfilter(b, a, sig)
        abs_sig  = np.abs(sig)
        env      = lfilter([0.01], [1.0, -0.99], abs_sig)
        gate     = np.clip((GATE_THRESH - env) / GATE_THRESH, 0.0, 1.0)
        y_out[ch] = (sig + (y_smooth - sig) * gate * BLEND_COEFF).astype(np.float32)
    return y_out[0] if is_1d else y_out

# ======== Language Strings ========
STRINGS = {
    "en": {
        "title":        "DSRE EX",
        "input_files":  "Audio Files List",
        "add_files":    "Add Files",
        "clear_files":  "Clear List",
        "remove_sel":   "Remove Selected",
        "output_dir":   "Output Directory",
        "select_dir":   "Select Output Directory",
        "convert":      "Convert",
        "start":        "Start",
        "cancel":       "Cancel",
        "retry":        "Retrying Failed Files",
        "params":       "Output Settings",
        "sr_label":     "Target Sample Rate:",
        "fmt_label":    "Output Format:",
        "file_prog":    "Current File Progress",
        "all_prog":     "Overall Progress",
        "log":          "Log",
        "ready":        "Ready",
        "light_mode":   "Light Mode",
        "dark_mode":    "Dark Mode",
        "about":        "About",
        "output_placeholder": "Select Folder",
        "lang_label":        "Language:",
        "file_menu":         "File(&F)",
        "process_menu":      "Process(&P)",
        "help_menu":         "Help(&H)",
        "recent_files":      "Recent Files(&R)",
        "exit":              "Exit(&X)",
        "no_recent":         "No recent files",
        "file_not_found":    "File not found",
        "file_not_found_msg": "File cannot be found.",
        "convert_error":     "Conversion error",
        "no_files_warning":  "Please select 1 or more files to convert.",
        "processing":        "Processing",
        "preparing":   "Preparing {n} files",
        "retrying":          "Retrying",
        "cancelling":        "Conversion Canceling",
        "finished":          "Working is complete.",
        "done":              "Complete",
        "error":             "Error",
        "log_loading": "Loading…: {path}",
        "log_saved":   "Conversion Completed: {path}",
        "log_author":        "Origin: Qu LeFan",
        "log_localize":      "Localization: Noir16",
        "log_mods":   "Mods: fuyuka3725",
    },
    "ja": {
        "title":        "DSRE EX",
        "input_files":  "楽曲一覧",
        "add_files":    "ファイル追加",
        "clear_files":  "リストをクリア",
        "remove_sel":   "選択項目を削除",
        "output_dir":   "出力パス",
        "select_dir":   "出力パスを選択",
        "convert":      "変換",
        "start":        "変換開始",
        "cancel":       "変換キャンセル",
        "retry":        "失敗ファイルを再試行",
        "params":       "出力設定",
        "sr_label":     "目標サンプリングレート:",
        "fmt_label":    "出力エンコード形式:",
        "file_prog":    "現在のファイル進捗",
        "all_prog":     "全体進捗",
        "log":          "ログ",
        "ready":        "準備完了",
        "light_mode":   "ライトモード",
        "dark_mode":    "ダークモード",
        "about":        "情報",
        "file_menu":    "ファイル(&F)",
        "process_menu": "処理(&P)",
        "help_menu":    "ヘルプ(&H)",
        "output_placeholder": "指定フォルダ",
        "lang_label":        "言語:",
        "recent_files":      "最近のファイル(&R)",
        "exit":              "終了(&X)",
        "no_recent":         "最近のファイルなし",
        "file_not_found":    "ファイルなし",
        "file_not_found_msg": "ファイルが見つかりませんでした。",
        "convert_error":     "変換エラー",
        "no_files_warning":  "変換するファイルを1個以上選択してください。",
        "processing":        "プロセス処理中",
        "preparing":     "{n}個のファイルを変換する準備中",
        "retrying":          "再試行中",
        "cancelling":        "変換キャンセル中",
        "finished":          "作業が完了いたしました。",
        "done":              "完了",
        "error":             "エラー",
        "log_loading": "読み込み中: {path}",
        "log_saved":   "変換完了: {path}",
        "log_author":        "原作者: 屈乐凡(Qu LeFan)",
        "log_localize":      "現地化: ノワール(Noir16)",
        "log_mods":   "改造: fuyuka3725",
    },
    "ko": {
        "title":        "DSRE EX",
        "input_files":  "음원 파일 목록",
        "add_files":    "음원 파일 추가",
        "clear_files":  "전체 항목 제거",
        "remove_sel":   "선택 항목 제거",
        "output_dir":   "출력 경로",
        "select_dir":   "출력 경로 선택",
        "convert":      "변환",
        "start":        "변환 시작",
        "cancel":       "변환 취소",
        "retry":        "실패한 파일 재시도",
        "params":       "출력 설정",
        "sr_label":     "목표 샘플링 레이트:",
        "fmt_label":    "출력 인코딩 형식:",
        "file_prog":    "현재 파일 처리 진행률",
        "all_prog":     "전체 파일 처리 진행률",
        "log":          "로그",
        "ready":        "준비완료",
        "light_mode":   "라이트 모드",
        "dark_mode":    "다크 모드",
        "about":        "정보",
        "file_menu":    "파일(&F)",
        "process_menu": "처리(&P)",
        "help_menu":    "도움말(&H)",
        "output_placeholder": "지정 폴더",
        "lang_label":        "언어:",
        "recent_files":      "최근 파일(&R)",
        "exit":              "종료(&X)",
        "no_recent":         "최근 파일 없음",
        "file_not_found":    "파일 없음",
        "file_not_found_msg": "파일을 찾을 수 없습니다",
        "convert_error":     "변환 오류",
        "no_files_warning":  "변환할 파일을 한 개 이상 선택해주세요.",
        "processing":        "작업 처리 중…",
        "preparing":     "{n}개의 파일을 변환할 준비 중",
        "retrying":          "재시도 중",
        "cancelling":        "변환 취소 중",
        "finished":          "작업이 완료되었습니다.",
        "done":              "완료",
        "error":             "오류",
        "log_loading": "불러오는 중: {path}",
        "log_saved":   "변환 완료: {path}",
        "log_author":        "원작자: 屈乐凡(Qu LeFan)",
        "log_localize":      "현지화: 느와르(Noir16)",
        "log_mods":   "개조: fuyuka3725",
    },
    "zh": {
        "title":        "DSRE EX",
        "input_files":  "歌曲列表",
        "add_files":    "添加输入文件",
        "clear_files":  "清空输入列表",
        "remove_sel":   "清空输入所选",
        "output_dir":   "输出路径",
        "select_dir":   "选择输出目录",
        "convert":      "转换",
        "start":        "开始转换",
        "cancel":       "取消转换",
        "retry":        "重试失败文件",
        "params":       "输出设置",
        "sr_label":     "目标采样率:",
        "fmt_label":    "输出编码格式:",
        "file_prog":    "当前文件进度",
        "all_prog":     "整体进度",
        "log":          "日志",
        "ready":        "准备就绪",
        "light_mode":   "浅色模式",
        "dark_mode":    "深色模式",
        "about":        "关于",
        "file_menu":    "文件(&F)",
        "process_menu": "处理(&P)",
        "help_menu":    "帮助(&H)",
        "output_placeholder": "指定文件夹",
        "lang_label":        "语言:",
        "recent_files":      "最近文件(&R)",
        "exit":              "退出(&X)",
        "no_recent":         "最近没有文件",
        "file_not_found":    "无文件",
        "file_not_found_msg": "找不到该文件。",
        "convert_error":     "没有文件",
        "no_files_warning":  "请选择一个或多个要转换的文件。",
        "processing":        "正在处理中",
        "preparing":     "转换准备转换{n}个文件",
        "retrying":          "转换重试",
        "cancelling":        "转换撤销转换",
        "finished":          "操作已完成。",
        "done":              "完成",
        "error":             "错误",
        "log_loading": "正在加载: {path}",
        "log_saved":   "转换完成: {path}",
        "log_author":        "原作者: 屈乐凡",
        "log_localize":      "本地化: Noir16",
        "log_mods":   "改装: fuyuka3725",
    },
}

# ======== Background Work Thread ========
class DSREWorker(QtCore.QThread):
    sig_log = QtCore.Signal(str)                         # log
    sig_file_progress = QtCore.Signal(int, int, str)     # current, total, filename
    sig_step_progress = QtCore.Signal(int, str)          # step progress (0~100)
    sig_overall_progress = QtCore.Signal(int, int)       # done, total
    sig_file_done = QtCore.Signal(str, str)              # in_path, out_path
    sig_error = QtCore.Signal(str, str)
    sig_finished = QtCore.Signal()

    def __init__(self, files, output_dir, params, parent=None):
        super().__init__(parent)
        self.files = files
        self.output_dir = output_dir
        self.params = params
        self._abort = False
        self._current_proc = None

    def abort(self):
        self._abort = True
        if self._current_proc and self._current_proc.poll() is None:
            self._current_proc.terminate()

    def tr(self, key: str) -> str:
        lang = self.params.get('lang', 'en')
        return STRINGS.get(lang, STRINGS["en"]).get(key, key)

    def run(self):
        total = len(self.files)
        done = 0
        self.sig_overall_progress.emit(done, total)

        for idx, in_path in enumerate(self.files, start=1):
            if self._abort:
                break

            fname = os.path.basename(in_path)
            self.sig_file_progress.emit(idx, total, fname)
            self.sig_step_progress.emit(0, fname)

            try:
                # Load
                self.sig_log.emit(self.tr("log_loading").format(path=in_path))

                # Bit_depth Check
                src_bit_depth = bde_probe_bit_depth(in_path)
                bde_bypass = src_bit_depth >= 24

                y, sr = load_audio(in_path)

                # Sort by (ch, n)
                if y.ndim == 1:
                    y = y[np.newaxis, :]

                # Anti-staircase
                if not bde_bypass:
                    y = bde_time_domain(y, sr)

                # Resample
                target_sr = int(self.params["target_sr"])
                is_upsample = target_sr > sr
                src_sr = sr
                y = resample_ardftsrc(
                    y, sr, target_sr,
                        bit_depth=32,
                        quality=8192 if is_upsample else 4096,
                        bandwidth=0.999 if is_upsample else 0.956,
                    )
                sr = target_sr

                # Processing
                if not bde_bypass:
                    pre_hp, _ = auto_hp_params(y, src_sr)
                    y = bde_spectral_detail_synth(y, sr, cutoff_hz=pre_hp)

                def step_cb(cur, m):
                    pct = int(cur * 100 / max(1, m))
                    self.sig_step_progress.emit(pct, fname)

                y_out = zansei_impl(
                    y, sr,
                    src_sr=src_sr,
                    progress_cb=step_cb,
                    abort_cb=lambda: self._abort
                )

                if self._abort:
                    break

                # Save
                os.makedirs(self.output_dir, exist_ok=True)
                base, ext = os.path.splitext(fname)
                ext = 'flac' if self.params['format'] == 'FLAC' else 'm4a'
                out_path = os.path.join(self.output_dir, f"{base}.{ext}")
                if os.path.normcase(os.path.abspath(out_path)) == \
                   os.path.normcase(os.path.abspath(in_path)):
                    out_path = os.path.join(self.output_dir, f"{base}_dsre.{ext}")
                out_path = save_wav24_out(in_path, y_out, sr, out_path, worker=self, fmt=self.params['format'])

                self.sig_log.emit(self.tr("log_saved").format(path=out_path))
                self.sig_file_done.emit(in_path, out_path)

            except Exception as e:
                err = "".join(traceback.format_exception_only(type(e), e)).strip()
                self.sig_error.emit(fname, err)

            done += 1
            self.sig_overall_progress.emit(done, total)
            self.sig_step_progress.emit(100, fname)

        self.sig_finished.emit()

# ======== GUI ========
class DragDropListWidget(QtWidgets.QListWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setDragDropMode(
            QtWidgets.QAbstractItemView.DragDropMode.DropOnly)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                file_path = url.toLocalFile()
                if file_path and (
                    self.findItems(
                        file_path, QtCore.Qt.MatchFlag.MatchExactly) == []):
                    self.addItem(file_path)
            event.acceptProposedAction()
        else:
            event.ignore()

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        # Initialize
        self.lang = "en"
        self.dark_mode = True
        self.recent_files = []
        self.max_recent_files = 10
        self.failed_files = []

        # Window
        self.setWindowTitle(self.tr("title"))
        icon_path = os.path.join(os.path.dirname(__file__), "logo.ico")
        self.setWindowIcon(QIcon(icon_path))
        self.resize(1024, 640)

        # Central
        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)

        # Start
        self.list_files = DragDropListWidget()
        self.list_files.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.list_files.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)

        self.btn_add             = QtWidgets.QPushButton(self.tr("add_files"))
        self.btn_clear           = QtWidgets.QPushButton(self.tr("clear_files"))
        self.btn_remove_selected = QtWidgets.QPushButton(self.tr("remove_sel"))
        self.btn_outdir          = QtWidgets.QPushButton(self.tr("select_dir"))
        self.le_outdir           = QtWidgets.QLineEdit()
        self.le_outdir.setPlaceholderText(self.tr("output_placeholder"))
        self.le_outdir.setText(os.path.abspath("output"))

        self.cb_sr = QtWidgets.QComboBox()
        for sr_val in [44100, 48000, 88200, 96000, 176400, 192000, 352800, 384000]:
            self.cb_sr.addItem(f"{sr_val // 1000} KHz  ({sr_val} Hz)", userData=sr_val)
        self.cb_sr.setCurrentIndex(3)

        self.pb_file = QtWidgets.QProgressBar()
        self.pb_all  = QtWidgets.QProgressBar()
        self.lbl_now = QtWidgets.QLabel(self.tr("convert"))

        self.lbl_stats = QtWidgets.QLabel(self.tr("ready"))
        self.lbl_stats.setStyleSheet("QLabel { font-size: 11px; }")
        self.lbl_eta = QtWidgets.QLabel("")
        self.lbl_eta.setStyleSheet("QLabel { font-size: 11px; }")

        self.btn_start  = QtWidgets.QPushButton(self.tr("start"))
        self.btn_cancel = QtWidgets.QPushButton(self.tr("cancel"))
        self.btn_cancel.setEnabled(False)
        self.btn_retry  = QtWidgets.QPushButton(self.tr("retry"))
        self.btn_retry.setEnabled(False)
        self.btn_retry.setStyleSheet(
            "QPushButton { background-color: #ff9800; color: white; }")

        self.btn_dark = QtWidgets.QPushButton(self.tr("light_mode"))
        self.btn_dark.clicked.connect(self.toggle_dark_mode)

        self.cb_format = QtWidgets.QComboBox()
        self.cb_format.addItems(["FLAC", "ALAC"])

        self.cb_lang = QtWidgets.QComboBox()
        self.cb_lang.addItem("English", userData="en")
        self.cb_lang.addItem("日本語", userData="ja")
        self.cb_lang.addItem("한국어", userData="ko")
        self.cb_lang.addItem("中文", userData="zh")

        self.te_log = QtWidgets.QTextEdit()
        self.te_log.setReadOnly(True)

        # Layout
        main_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)

        # Left: File list
        left_widget = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout()
        self.lbl_files = QtWidgets.QLabel(self.tr("input_files"))
        self.lbl_files.setAlignment(QtCore.Qt.AlignHCenter)
        left_layout.addWidget(self.lbl_files)
        left_layout.addWidget(self.list_files)
        left_widget.setLayout(left_layout)
        main_splitter.addWidget(left_widget)

        # Center: Transformation Operations
        middle_widget = QtWidgets.QWidget()
        middle_widget.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred,
            QtWidgets.QSizePolicy.Policy.Maximum)
        middle_layout = QtWidgets.QVBoxLayout()
        middle_layout.setAlignment(QtCore.Qt.AlignTop)
        self.lbl_ops = QtWidgets.QLabel(self.tr("convert"))
        self.lbl_ops.setAlignment(QtCore.Qt.AlignHCenter)
        middle_layout.addWidget(self.lbl_ops)

        vbtn = QtWidgets.QVBoxLayout()
        vbtn.setAlignment(QtCore.Qt.AlignTop)
        vbtn.addWidget(self.btn_add)
        vbtn.addWidget(self.btn_clear)
        vbtn.addWidget(self.btn_remove_selected)
        vbtn.addSpacing(10)
        self.lbl_outdir = QtWidgets.QLabel(self.tr("output_dir"))
        vbtn.addWidget(self.lbl_outdir)
        vbtn.addWidget(self.le_outdir)
        vbtn.addWidget(self.btn_outdir)
        vbtn.addSpacing(20)
        vbtn.addWidget(self.lbl_now)
        vbtn.addWidget(self.btn_start)
        vbtn.addWidget(self.btn_cancel)
        vbtn.addWidget(self.btn_retry)
        vbtn.addWidget(self.btn_dark)
        middle_layout.addLayout(vbtn)
        middle_layout.addStretch(1)
        middle_widget.setLayout(middle_layout)
        main_splitter.addWidget(middle_widget)

        # Right: Parameters + Progress Bar
        right_widget = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout()
        self.lbl_params = QtWidgets.QLabel(self.tr("params"))
        self.lbl_params.setAlignment(QtCore.Qt.AlignHCenter)
        right_layout.addWidget(self.lbl_params)

        form = QtWidgets.QFormLayout()
        self.lbl_sr    = QtWidgets.QLabel(self.tr("sr_label"))
        self.lbl_fmt   = QtWidgets.QLabel(self.tr("fmt_label"))
        form.addRow(self.lbl_sr,    self.cb_sr)
        form.addRow(self.lbl_fmt,   self.cb_format)
        right_layout.addLayout(form)
        right_layout.addSpacing(20)

        vprog = QtWidgets.QVBoxLayout()
        self.lbl_file_prog = QtWidgets.QLabel(self.tr("file_prog"))
        self.lbl_all_prog  = QtWidgets.QLabel(self.tr("all_prog"))
        vprog.addWidget(self.lbl_file_prog)
        vprog.addWidget(self.pb_file)
        vprog.addWidget(self.lbl_all_prog)
        vprog.addWidget(self.pb_all)
        vprog.addWidget(self.lbl_stats)
        vprog.addWidget(self.lbl_eta)
        vprog.addSpacing(10)

        # Select Language
        lang_layout = QtWidgets.QHBoxLayout()
        self.lbl_lang = QtWidgets.QLabel(self.tr("lang_label"))
        lang_layout.addWidget(self.lbl_lang)
        lang_layout.addWidget(self.cb_lang)
        vprog.addLayout(lang_layout)
        vprog.addStretch(1)

        right_layout.addLayout(vprog)
        right_widget.setLayout(right_layout)
        main_splitter.addWidget(right_widget)

        main_splitter.setSizes([300, 300, 400])

        # Vertical Splitter
        vertical_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        vertical_splitter.addWidget(main_splitter)

        log_widget = QtWidgets.QWidget()
        log_layout = QtWidgets.QVBoxLayout()
        self.lbl_log = QtWidgets.QLabel(self.tr("log"))
        log_layout.addWidget(self.lbl_log)
        log_layout.addWidget(self.te_log)
        log_widget.setLayout(log_layout)
        vertical_splitter.addWidget(log_widget)
        vertical_splitter.setSizes([600, 200])

        main_layout = QtWidgets.QVBoxLayout()
        main_layout.addWidget(vertical_splitter)
        self.central_widget.setLayout(main_layout)

        # Signal Connection
        self.btn_add.clicked.connect(self.on_add_files)
        self.btn_clear.clicked.connect(self.on_clear_files)
        self.btn_remove_selected.clicked.connect(self.on_remove_selected)
        self.btn_outdir.clicked.connect(self.on_choose_outdir)
        self.btn_start.clicked.connect(self.on_start)
        self.btn_cancel.clicked.connect(self.on_cancel)
        self.btn_retry.clicked.connect(self.on_retry_failed)
        self.list_files.itemSelectionChanged.connect(self.update_button_states)
        self.cb_lang.currentIndexChanged.connect(self._on_lang_changed)

        self.worker: Optional[DSREWorker] = None
        self.config_file = get_config_path("DSRE.json")

        # Load Settings
        self.load_config()

        # Auto-save when parameters are changed
        self.cb_sr.currentIndexChanged.connect(self.save_config)
        self.le_outdir.textChanged.connect(self.save_config)
        self.cb_format.currentTextChanged.connect(self.save_config)

        # Menu Bar / Status Bar
        self.create_menu_bar()
        self.statusBar().showMessage(self.tr("ready"))

        self.update_recent_files_menu()

        # Language applied after initialization
        self.retranslate_ui()

        # Theme / Welcome Message
        self.apply_theme()

        self.append_log(self.tr("log_author"))
        self.append_log(self.tr("log_localize"))
        self.append_log(self.tr("log_mods"))

    def tr(self, key: str) -> str:
        return STRINGS.get(self.lang, STRINGS["ko"]).get(key, key)

    def retranslate_ui(self):
        self.setWindowTitle(self.tr("title"))
        self.lbl_files.setText(self.tr("input_files"))
        self.lbl_ops.setText(self.tr("convert"))
        self.lbl_outdir.setText(self.tr("output_dir"))
        self.btn_add.setText(self.tr("add_files"))
        self.btn_clear.setText(self.tr("clear_files"))
        self.btn_remove_selected.setText(self.tr("remove_sel"))
        self.btn_outdir.setText(self.tr("select_dir"))
        self.btn_start.setText(self.tr("start"))
        self.btn_cancel.setText(self.tr("cancel"))
        self.btn_retry.setText(self.tr("retry"))
        self.lbl_now.setText(self.tr("convert"))
        self.btn_dark.setText(
            self.tr("light_mode") if self.dark_mode else self.tr("dark_mode"))
        self.lbl_params.setText(self.tr("params"))
        self.lbl_sr.setText(self.tr("sr_label"))
        self.lbl_fmt.setText(self.tr("fmt_label"))
        self.lbl_lang.setText(self.tr("lang_label"))
        self.lbl_file_prog.setText(self.tr("file_prog"))
        self.lbl_all_prog.setText(self.tr("all_prog"))
        self.lbl_log.setText(self.tr("log"))

    def _on_lang_changed(self):
        self.lang = self.cb_lang.currentData()
        self.retranslate_ui()
        self.menuBar().clear()
        self.create_menu_bar()
        self.update_recent_files_menu()
        self.statusBar().showMessage(self.tr("ready"))
        self.save_config()

    def create_menu_bar(self):
        menubar = self.menuBar()

        file_menu = menubar.addMenu(self.tr("file_menu"))

        add_action = QAction(self.tr("add_files"), self)
        add_action.setShortcut(QKeySequence.StandardKey.Open)
        add_action.triggered.connect(self.on_add_files)
        file_menu.addAction(add_action)

        clear_action = QAction(self.tr("clear_files"), self)
        clear_action.setShortcut('Ctrl+L')
        clear_action.triggered.connect(self.on_clear_files)
        file_menu.addAction(clear_action)

        file_menu.addSeparator()

        self.recent_menu = file_menu.addMenu(self.tr("recent_files"))

        file_menu.addSeparator()

        exit_action = QAction(self.tr("exit"), self)
        exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        process_menu = menubar.addMenu(self.tr("process_menu"))

        start_action = QAction(self.tr("start"), self)
        start_action.setShortcut('F5')
        start_action.triggered.connect(self.on_start)
        process_menu.addAction(start_action)

        cancel_action = QAction(self.tr("cancel"), self)
        cancel_action.setShortcut('Escape')
        cancel_action.triggered.connect(self.on_cancel)
        process_menu.addAction(cancel_action)

        retry_action = QAction(self.tr("retry"), self)
        retry_action.setShortcut('Ctrl+R')
        retry_action.triggered.connect(self.on_retry_failed)
        process_menu.addAction(retry_action)

        help_menu = menubar.addMenu(self.tr("help_menu"))
        about_action = QAction(self.tr("about"), self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def toggle_dark_mode(self):
        self.dark_mode = not self.dark_mode
        self.apply_theme()
        self.save_config()
        self.btn_dark.setText(
            self.tr("light_mode") if self.dark_mode else self.tr("dark_mode"))

    def apply_theme(self):
        if self.dark_mode:
            self.setStyleSheet("""
                QMainWindow, QWidget {
                    background-color: #2b2b2b; color: #ffffff; }
                QListWidget {
                    background-color: #3c3c3c; color: #ffffff;
                    border: 2px dashed #666666; border-radius: 5px; }
                QListWidget::item {
                    padding: 5px; border-bottom: 1px solid #555555; }
                QListWidget::item:hover { background-color: #4a4a4a; }
                QListWidget::item:selected {
                    background-color: #0078d4; color: white; }
                QPushButton {
                    background-color: #404040; color: #ffffff;
                    border: 1px solid #666666; padding: 5px;
                    border-radius: 3px; }
                QPushButton:hover { background-color: #505050; }
                QPushButton:pressed { background-color: #606060; }
                QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
                    background-color: #3c3c3c; color: #ffffff;
                    border: 1px solid #666666; padding: 5px; }
                QProgressBar {
                    background-color: #3c3c3c;
                    border: 1px solid #666666; text-align: center; }
                QProgressBar::chunk { background-color: #0078d4; }
                QTextEdit {
                    background-color: #3c3c3c; color: #ffffff;
                    border: 1px solid #666666; }
                QMenuBar {
                    background-color: #2b2b2b; color: #ffffff;
                    border-bottom: 1px solid #666666; }
                QMenuBar::item {
                    background-color: transparent; padding: 4px 8px; }
                QMenuBar::item:selected { background-color: #404040; }
                QMenu {
                    background-color: #3c3c3c; color: #ffffff;
                    border: 1px solid #666666; }
                QMenu::item { padding: 4px 20px; }
                QMenu::item:selected { background-color: #404040; }
                QStatusBar {
                    background-color: #2b2b2b; color: #ffffff;
                    border-top: 1px solid #666666; }
                QSplitter::handle { background-color: #666666; }
            """)
        else:
            self.setStyleSheet("""
                QMainWindow, QWidget {
                    background-color: #ffffff; color: #333333; }
                QListWidget {
                    border: 2px dashed #aaa; border-radius: 5px;
                    background-color: #f9f9f9; }
                QListWidget::item {
                    padding: 5px; border-bottom: 1px solid #eee; }
                QListWidget::item:hover { background-color: #e3f2fd; }
                QListWidget::item:selected {
                    background-color: #2196f3; color: white; }
                QPushButton {
                    background-color: #f0f0f0; color: #333333;
                    border: 1px solid #cccccc; padding: 5px;
                    border-radius: 3px; }
                QPushButton:hover { background-color: #e0e0e0; }
                QPushButton:pressed { background-color: #d0d0d0; }
                QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
                    background-color: #ffffff; color: #333333;
                    border: 1px solid #cccccc; padding: 5px; }
                QProgressBar {
                    background-color: #f0f0f0;
                    border: 1px solid #cccccc; text-align: center; }
                QProgressBar::chunk { background-color: #2196f3; }
                QTextEdit {
                    background-color: #ffffff; color: #333333;
                    border: 1px solid #cccccc; }
                QMenuBar {
                    background-color: #f0f0f0; color: #333333;
                    border-bottom: 1px solid #cccccc; }
                QMenuBar::item {
                    background-color: transparent; padding: 4px 8px; }
                QMenuBar::item:selected { background-color: #e0e0e0; }
                QMenu {
                    background-color: #ffffff; color: #333333;
                    border: 1px solid #cccccc; }
                QMenu::item { padding: 4px 20px; }
                QMenu::item:selected { background-color: #e0e0e0; }
                QStatusBar {
                    background-color: #f0f0f0; color: #333333;
                    border-top: 1px solid #cccccc; }
                QSplitter::handle { background-color: #cccccc; }
            """)

    def show_about(self):
        QtWidgets.QMessageBox.about(self, self.tr("about"),
            "DSRE EX\n\n"
            "IIR Network Frequency Shift\n"
            "Audio Upscaler.\n\n"
            "Built-in ARDFTSRC Resampler.\n"
            "Auto parameter adjustment.\n\n"
            "Origin: Qu LeFan\n"
            "Localization: Noir16\n"
            "Mods: fuyuka3725")

    def on_add_files(self):
        filters = (
            "Audio Files (*.wav *.mp3 *.m4a *.flac *.ogg *.aiff *.aif *.aac *.wma *.mka *.opus);;"
            "All Files (*.*)"
        )
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, self.tr("add_files"), "", filters)
        for f in files:
            if f and (self.list_files.findItems(
                    f, QtCore.Qt.MatchFlag.MatchExactly) == []):
                self.list_files.addItem(f)
                self.add_to_recent_files(f)
        self.update_button_states()

    def on_clear_files(self):
        self.list_files.clear()
        self.lbl_stats.setText(self.tr("ready"))
        self.lbl_eta.setText("")
        self.pb_all.setValue(0)
        self.pb_file.setValue(0)
        self.update_button_states()

    def on_remove_selected(self):
        for item in reversed(self.list_files.selectedItems()):
            self.list_files.takeItem(self.list_files.row(item))
        self.update_button_states()

    def update_button_states(self):
        self.btn_remove_selected.setEnabled(
            len(self.list_files.selectedItems()) > 0)
        self.btn_retry.setEnabled(len(self.failed_files) > 0)

    def on_choose_outdir(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(
            self, self.tr("select_dir"), self.le_outdir.text() or "")
        if d:
            self.le_outdir.setText(d)

    def load_config(self):
        for widget in [self.cb_sr, self.le_outdir, self.cb_format, self.cb_lang]:
            widget.blockSignals(True)
        try:
            if os.path.exists(self.config_file):
                if os.path.getsize(self.config_file) == 0:
                    return
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                target_sr = config.get('target_sr', 96000)
                for i in range(self.cb_sr.count()):
                    if self.cb_sr.itemData(i) == target_sr:
                        self.cb_sr.setCurrentIndex(i)
                        break
                self.le_outdir.setText(
                    config.get('output_dir', os.path.abspath("output")))
                fmt_map = {'FLAC': 0, 'ALAC': 1}
                self.cb_format.setCurrentIndex(
                    fmt_map.get(config.get('format', 'FLAC'), 0))
                self.recent_files = config.get('recent_files', [])
                self.dark_mode = config.get('dark_mode', True)
                self.btn_dark.setText(
                    self.tr("light_mode") if self.dark_mode
                    else self.tr("dark_mode"))

                saved_lang = config.get('lang', 'ko')
                self.lang = saved_lang
                for i in range(self.cb_lang.count()):
                    if self.cb_lang.itemData(i) == saved_lang:
                        self.cb_lang.setCurrentIndex(i)
                        break

        except Exception as e:
            self.append_log(f"設定の保存に失敗: {e}")

        finally:
            for widget in [self.cb_sr, self.le_outdir, self.cb_format, self.cb_lang]:
                widget.blockSignals(False)

    def save_config(self):
        try:
            config = {
                'target_sr':    self.cb_sr.currentData(),
                'output_dir':   self.le_outdir.text(),
                'format':       self.cb_format.currentText(),
                'recent_files': self.recent_files,
                'dark_mode':    self.dark_mode,
                'lang':         self.lang,
            }
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            self.append_log(f"設定の保存に失敗: {e}")

    def add_to_recent_files(self, file_path: str):
        if file_path in self.recent_files:
            self.recent_files.remove(file_path)
        self.recent_files.insert(0, file_path)
        if len(self.recent_files) > self.max_recent_files:
            self.recent_files = self.recent_files[:self.max_recent_files]
        self.update_recent_files_menu()

    def update_recent_files_menu(self):
        self.recent_menu.clear()
        if not self.recent_files:
            action = QAction(self.tr("no_recent"), self)
            action.setEnabled(False)
            self.recent_menu.addAction(action)
        else:
            for fp in self.recent_files:
                action = QAction(os.path.basename(fp), self)
                action.triggered.connect(
                    lambda checked, path=fp: self.load_recent_file(path))
                self.recent_menu.addAction(action)

    def load_recent_file(self, file_path: str):
        if os.path.exists(file_path):
            if not self.list_files.findItems(
                    file_path, QtCore.Qt.MatchFlag.MatchExactly):
                self.list_files.addItem(file_path)
        else:
            if file_path in self.recent_files:
                self.recent_files.remove(file_path)
                self.update_recent_files_menu()
            QtWidgets.QMessageBox.warning(
                self, self.tr("file_not_found"),
                f"{self.tr('file_not_found_msg')}: {file_path}")

    def params(self):
        return dict(
            target_sr=self.cb_sr.currentData(),
            bit_depth=24,
            format=self.cb_format.currentText(),
            lang=self.lang,
        )

    def append_log(self, s: str):
        self.te_log.append(s)
        self.te_log.moveCursor(QTextCursor.End)

    def on_start(self):
        files = [self.list_files.item(i).text()
                 for i in range(self.list_files.count())]
        if not files:
            QtWidgets.QMessageBox.warning(
                self, self.tr("convert_error"),
                self.tr("no_files_warning"))
            return
        outdir = self.le_outdir.text().strip() or os.path.abspath("output")

        self.pb_all.setValue(0)
        self.pb_file.setValue(0)
        self.lbl_now.setText(self.tr("processing"))
        self.lbl_stats.setText(
            self.tr("preparing").format(n=len(files)))
        self.append_log(
            self.tr("preparing").format(n=len(files)))

        self.failed_files.clear()
        self.btn_start.setEnabled(False)
        self.btn_cancel.setEnabled(True)

        self.worker = DSREWorker(files, outdir, self.params())
        self.worker.sig_log.connect(self.append_log)
        self.worker.sig_file_progress.connect(self.on_file_progress)
        self.worker.sig_step_progress.connect(self.on_step_progress)
        self.worker.sig_overall_progress.connect(self.on_overall_progress)
        self.worker.sig_file_done.connect(self.on_file_done)
        self.worker.sig_error.connect(self.on_error)
        self.worker.sig_finished.connect(self.on_finished)
        self.worker.start()

    @QtCore.Slot(int, int, str)
    def on_file_progress(self, cur, total, fname):
        self.lbl_now.setText(f"[{cur}/{total}] {fname}")
        self.pb_file.setValue(0)
        self.statusBar().showMessage(f"[{cur}/{total}] {fname}")

    @QtCore.Slot(int, str)
    def on_step_progress(self, pct, fname):
        self.pb_file.setValue(pct)

    @QtCore.Slot(int, int)
    def on_overall_progress(self, done, total):
        pct = int(done * 100 / max(1, total))
        self.pb_all.setValue(pct)
        self.lbl_stats.setText(f"{done}/{total}")
        self.statusBar().showMessage(f"{done}/{total} ({pct}%)")

    @QtCore.Slot(str, str)
    def on_file_done(self, in_path, out_path):
        self.append_log(
            f"{self.tr('log_saved').format(path=os.path.basename(in_path))} -> {out_path}")

    @QtCore.Slot(str, str)
    def on_error(self, fname, err):
        self.append_log(f"[{self.tr('log_error')}] {fname}: {err}")
        self.failed_files.append(fname)
        self.btn_retry.setEnabled(True)

    def on_retry_failed(self):
        if not self.failed_files:
            return
        self.append_log(
            self.tr("log_retrying").format(n=len(self.failed_files)))
        self.pb_all.setValue(0)
        self.pb_file.setValue(0)
        self.lbl_now.setText(self.tr("retrying"))
        self.btn_start.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.btn_retry.setEnabled(False)

        outdir = self.le_outdir.text().strip() or os.path.abspath("output")
        self.worker = DSREWorker(self.failed_files, outdir, self.params())
        self.worker.sig_log.connect(self.append_log)
        self.worker.sig_file_progress.connect(self.on_file_progress)
        self.worker.sig_step_progress.connect(self.on_step_progress)
        self.worker.sig_overall_progress.connect(self.on_overall_progress)
        self.worker.sig_file_done.connect(self.on_file_done)
        self.worker.sig_error.connect(self.on_error)
        self.worker.sig_finished.connect(self.on_finished)
        self.worker.start()

    def on_cancel(self):
        if self.worker and self.worker.isRunning():
            self.append_log(self.tr("cancelling"))
            self.statusBar().showMessage(self.tr("cancelling"))
            self.worker.abort()

    def on_finished(self):
        self.append_log(self.tr("finished"))
        self.lbl_now.setText(self.tr("convert"))
        self.lbl_stats.setText(self.tr("done"))
        self.statusBar().showMessage(self.tr("done"))
        self.btn_start.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.btn_retry.setEnabled(len(self.failed_files) > 0)
        self.worker = None

    def format_time(self, seconds):
        if seconds < 60:
            return f"{seconds:.0f}sec"
        elif seconds < 3600:
            return f"{seconds/60:.0f}min {seconds%60:.0f}sec"
        else:
            return f"{int(seconds//3600)}h {int((seconds%3600)//60)}min"


def main():
    import ctypes
    myappid = "org.fuyuka.dsre"
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

    app = QtWidgets.QApplication(sys.argv)

    icon_path = os.path.join(os.path.dirname(__file__), "logo.ico")
    app.setWindowIcon(QIcon(icon_path))

    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
