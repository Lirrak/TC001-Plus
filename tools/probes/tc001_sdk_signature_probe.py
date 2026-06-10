#!/usr/bin/env python3
"""
Try one libiruvc.dll stream_start signature in an isolated process.

Usage:
    py tc001_sdk_signature_probe.py variant_name

This is deliberately a small crash-tolerant probe. Some wrong ctypes signatures
can access-violate inside the vendor DLL, so the caller should run each variant
as a separate Python process.
"""

from __future__ import annotations

import ctypes
import os
import sys
import time


DLL_DIR = r"C:\Program Files\TOPDON\TopView\dll\dll_c001p"
APP_DIR = r"C:\Program Files\TOPDON\TopView"


def setup():
    os.add_dll_directory(DLL_DIR)
    os.add_dll_directory(APP_DIR)
    d = ctypes.CDLL(os.path.join(DLL_DIR, "libiruvc.dll"))

    d.uvc_camera_init.restype = ctypes.c_int
    d.uvc_camera_list.argtypes = [ctypes.c_void_p]
    d.uvc_camera_list.restype = ctypes.c_int
    d.uvc_camera_info_get.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    d.uvc_camera_info_get.restype = ctypes.c_int
    d.uvc_camera_open.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    d.uvc_camera_open.restype = ctypes.c_int
    d.uvc_camera_stream_close.restype = ctypes.c_int
    d.uvc_camera_close.restype = ctypes.c_int
    d.uvc_frame_buf_create.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint]
    d.uvc_frame_buf_create.restype = ctypes.c_void_p
    d.uvc_frame_buf_release.argtypes = [ctypes.c_void_p]
    d.uvc_frame_buf_release.restype = ctypes.c_int

    print("init", d.uvc_camera_init(), flush=True)
    dev = ctypes.create_string_buffer(8192)
    info = ctypes.create_string_buffer(8192)
    print("list", d.uvc_camera_list(ctypes.byref(dev)), flush=True)
    print("info", d.uvc_camera_info_get(ctypes.byref(dev), ctypes.byref(info)), flush=True)

    # Mode offset 160 is 256x384, frame size 196608, 25 FPS.
    mode = ctypes.byref(info, 160)
    print("open", d.uvc_camera_open(ctypes.byref(dev), mode), flush=True)

    ctx = ctypes.create_string_buffer(4096)
    fb = d.uvc_frame_buf_create(ctypes.byref(ctx), 196608, 4)
    print("fb", hex(fb or 0), flush=True)
    return d, dev, info, mode, fb


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: py tc001_sdk_signature_probe.py <variant>")
        return 2
    variant = sys.argv[1]
    d, dev, info, mode, fb = setup()

    cb_count = ctypes.c_int(0)

    # Try both common frame callback shapes. The first matches libuvc
    # uvc_frame_callback_t(frame*, void*). The second is a plain data callback
    # sometimes used by vendor wrappers.
    CB_FRAME = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p)
    CB_DATA = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p)

    def cb_frame(frame, user):
        cb_count.value += 1
        print("cb_frame", cb_count.value, hex(frame or 0), hex(user or 0), flush=True)

    def cb_data(data, length, user):
        cb_count.value += 1
        print("cb_data", cb_count.value, hex(data or 0), int(length), hex(user or 0), flush=True)

    cbf = CB_FRAME(cb_frame)
    cbd = CB_DATA(cb_data)
    function_name = "uvc_camera_stream_start"
    if ":" in variant:
        function_name, variant = variant.split(":", 1)

    f = getattr(d, function_name)
    f.restype = ctypes.c_int

    variants = {
        "cb_user_flags": ([CB_FRAME, ctypes.c_void_p, ctypes.c_ubyte], [cbf, None, 0]),
        "mode_cb_user_flags": ([ctypes.c_void_p, CB_FRAME, ctypes.c_void_p, ctypes.c_ubyte], [mode, cbf, None, 0]),
        "dev_mode_cb_user_flags": ([ctypes.c_void_p, ctypes.c_void_p, CB_FRAME, ctypes.c_void_p, ctypes.c_ubyte], [ctypes.byref(dev), mode, cbf, None, 0]),
        "fb_cb_user_flags": ([ctypes.c_void_p, CB_FRAME, ctypes.c_void_p, ctypes.c_ubyte], [fb, cbf, None, 0]),
        "cb_user_fb_flags": ([CB_FRAME, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ubyte], [cbf, None, fb, 0]),
        "data_cb_user_flags": ([CB_DATA, ctypes.c_void_p, ctypes.c_ubyte], [cbd, None, 0]),
        "mode_data_cb_user_flags": ([ctypes.c_void_p, CB_DATA, ctypes.c_void_p, ctypes.c_ubyte], [mode, cbd, None, 0]),
        "cb_user_fb": ([CB_FRAME, ctypes.c_void_p, ctypes.c_void_p], [cbf, None, fb]),
    }
    if variant not in variants:
        print("unknown variant", variant)
        print("known:", ", ".join(sorted(variants)))
        return 2

    argtypes, args = variants[variant]
    f.argtypes = argtypes
    print("function", function_name, "variant", variant, flush=True)
    print("stream_start", f(*args), flush=True)
    time.sleep(2)
    print("cb_count", cb_count.value, flush=True)

    try:
        print("stream_close", d.uvc_camera_stream_close(), flush=True)
    except Exception as exc:
        print("stream_close_exc", exc, flush=True)
    try:
        print("fb_release", d.uvc_frame_buf_release(fb), flush=True)
    except Exception as exc:
        print("fb_release_exc", exc, flush=True)
    try:
        print("close", d.uvc_camera_close(), flush=True)
    except Exception as exc:
        print("close_exc", exc, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
