#!/usr/bin/env python3
"""
Capture one TC001 Plus raw frame via TOPDON's libiruvc.dll.

This uses the vendor SDK call pattern recovered from libiruvc.dll:

    uvc_camera_stream_start(stream_param*, user_callback_context_or_null)
    uvc_frame_get(raw_data_buffer)

The stream parameter struct layout used here is the subset read by
uvc_camera_stream_start/uvc_frame_buf_create:

    +0x10 char* format ("YUY2" or "MJPEG")
    +0x18 int width
    +0x1c int height
    +0x20 int frame_size
    +0x24 int fps
    +0x28 int user_frame_size_or_flag
"""

from __future__ import annotations

import ctypes
import os
import time
from pathlib import Path

import numpy as np


DLL_DIR = r"C:\Program Files\TOPDON\TopView\dll\dll_c001p"
APP_DIR = r"C:\Program Files\TOPDON\TopView"
WIDTH = 256
HEIGHT = 384
FRAME_SIZE = WIDTH * HEIGHT * 2
FPS = 25


def temp_stats(raw: bytes) -> str:
    mat = np.frombuffer(raw[:FRAME_SIZE], dtype="<u2").reshape((HEIGHT, WIDTH))
    temp_raw = mat[HEIGHT // 2 :, :]
    temp_c = temp_raw.astype(np.float32) / 64.0 - 273.15
    return (
        f"min={np.nanmin(temp_c):.2f}C, "
        f"avg={np.nanmean(temp_c):.2f}C, "
        f"center={temp_c[temp_c.shape[0] // 2, temp_c.shape[1] // 2]:.2f}C, "
        f"max={np.nanmax(temp_c):.2f}C"
    )


def main() -> int:
    os.add_dll_directory(DLL_DIR)
    os.add_dll_directory(APP_DIR)
    dll = ctypes.CDLL(os.path.join(DLL_DIR, "libiruvc.dll"))

    dll.uvc_camera_init.restype = ctypes.c_int
    dll.uvc_camera_list.argtypes = [ctypes.c_void_p]
    dll.uvc_camera_list.restype = ctypes.c_int
    dll.uvc_camera_info_get.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    dll.uvc_camera_info_get.restype = ctypes.c_int
    dll.uvc_camera_open.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    dll.uvc_camera_open.restype = ctypes.c_int
    dll.uvc_camera_stream_start.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    dll.uvc_camera_stream_start.restype = ctypes.c_int
    dll.uvc_frame_buf_create.argtypes = [ctypes.c_void_p]
    dll.uvc_frame_buf_create.restype = ctypes.c_void_p
    dll.uvc_frame_get.argtypes = [ctypes.c_void_p]
    dll.uvc_frame_get.restype = ctypes.c_int
    dll.uvc_camera_stream_close.restype = ctypes.c_int
    dll.uvc_frame_buf_release.argtypes = [ctypes.c_void_p]
    dll.uvc_frame_buf_release.restype = ctypes.c_int
    dll.uvc_camera_close.restype = ctypes.c_int
    dll.uvc_camera_release.restype = ctypes.c_int

    fmt = ctypes.create_string_buffer(b"YUY2")
    stream = ctypes.create_string_buffer(0x40)
    ctypes.c_void_p.from_buffer(stream, 0x10).value = ctypes.addressof(fmt)
    ctypes.c_int.from_buffer(stream, 0x18).value = WIDTH
    ctypes.c_int.from_buffer(stream, 0x1C).value = HEIGHT
    ctypes.c_int.from_buffer(stream, 0x20).value = FRAME_SIZE
    ctypes.c_int.from_buffer(stream, 0x24).value = FPS
    ctypes.c_int.from_buffer(stream, 0x28).value = FRAME_SIZE

    print("init", dll.uvc_camera_init())
    dev = ctypes.create_string_buffer(8192)
    info = ctypes.create_string_buffer(8192)
    print("list", dll.uvc_camera_list(ctypes.byref(dev)))
    print("info", dll.uvc_camera_info_get(ctypes.byref(dev), ctypes.byref(info)))

    # Any valid mode works for open; actual format is selected by stream_start.
    mode_256x384 = ctypes.byref(info, 160)
    print("open", dll.uvc_camera_open(ctypes.byref(dev), mode_256x384))

    frame_buf = dll.uvc_frame_buf_create(ctypes.byref(stream))
    print("frame_buf", hex(frame_buf or 0))
    if not frame_buf:
        return 2

    print("stream_start", dll.uvc_camera_stream_start(ctypes.byref(stream), None))

    raw_buf = ctypes.create_string_buffer(FRAME_SIZE)
    deadline = time.time() + 8.0
    count = 0
    try:
        while time.time() < deadline:
            ret = dll.uvc_frame_get(ctypes.byref(raw_buf))
            count += 1
            if ret == 0:
                raw = bytes(raw_buf)
                print("frame_get", ret, "attempt", count, temp_stats(raw))
                out = Path("tc001_sdk_last_frame.raw")
                out.write_bytes(raw)
                print(f"Saved raw frame: {out.resolve()}")
                return 0
            if count <= 5 or count % 20 == 0:
                print("frame_get", ret, "attempt", count)
            time.sleep(0.05)
        return 3
    finally:
        try:
            print("stream_close", dll.uvc_camera_stream_close())
        except Exception:
            pass
        try:
            print("frame_buf_release", dll.uvc_frame_buf_release(frame_buf))
        except Exception:
            pass
        try:
            print("close", dll.uvc_camera_close())
        except Exception:
            pass
        try:
            print("release", dll.uvc_camera_release())
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
