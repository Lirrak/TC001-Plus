from __future__ import annotations

import ctypes
import os
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


# TC001 / TC001 Plus commonly uses a 256 x 192 thermal sensor.
TC001_WIDTH = 256
TC001_HEIGHT = 192
TC001_FPS = 25


@dataclass
class TC001SdkFrame:
    temp_c: np.ndarray
    preview_bgr: Optional[np.ndarray]


class TC001SdkCamera:
    """Read real TC001 Plus radiometric frames through TOPDON's libiruvc.dll."""

    WIDTH = TC001_WIDTH
    HEIGHT = TC001_HEIGHT * 2
    FRAME_SIZE = WIDTH * HEIGHT * 2
    FPS = TC001_FPS

    def __init__(self, dll_dir: str) -> None:
        self.dll_dir = dll_dir
        self.app_dir = os.path.dirname(os.path.dirname(dll_dir))
        self.dll: Optional[ctypes.CDLL] = None
        self.dev = ctypes.create_string_buffer(8192)
        self.info = ctypes.create_string_buffer(8192)
        self.stream = ctypes.create_string_buffer(0x40)
        self.format_name = ctypes.create_string_buffer(b"YUY2")
        self.frame_buf: Optional[int] = None
        self.raw_buf = ctypes.create_string_buffer(self.FRAME_SIZE)

    def _check(self, code: int, name: str) -> None:
        if code < 0:
            raise RuntimeError(f"{name} failed: {code}")

    def open(self) -> None:
        if not os.path.exists(os.path.join(self.dll_dir, "libiruvc.dll")):
            raise RuntimeError(f"libiruvc.dll was not found in: {self.dll_dir}")

        os.add_dll_directory(self.dll_dir)
        if os.path.isdir(self.app_dir):
            os.add_dll_directory(self.app_dir)

        dll = ctypes.CDLL(os.path.join(self.dll_dir, "libiruvc.dll"))
        self.dll = dll

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

        ctypes.c_void_p.from_buffer(self.stream, 0x10).value = ctypes.addressof(self.format_name)
        ctypes.c_int.from_buffer(self.stream, 0x18).value = self.WIDTH
        ctypes.c_int.from_buffer(self.stream, 0x1C).value = self.HEIGHT
        ctypes.c_int.from_buffer(self.stream, 0x20).value = self.FRAME_SIZE
        ctypes.c_int.from_buffer(self.stream, 0x24).value = self.FPS
        ctypes.c_int.from_buffer(self.stream, 0x28).value = self.FRAME_SIZE

        self._check(dll.uvc_camera_init(), "uvc_camera_init")
        self._check(dll.uvc_camera_list(ctypes.byref(self.dev)), "uvc_camera_list")
        self._check(dll.uvc_camera_info_get(ctypes.byref(self.dev), ctypes.byref(self.info)), "uvc_camera_info_get")

        # Offset 160 is the SDK's 256x384@25 mode in the info table.
        self._check(dll.uvc_camera_open(ctypes.byref(self.dev), ctypes.byref(self.info, 160)), "uvc_camera_open")

        self.frame_buf = dll.uvc_frame_buf_create(ctypes.byref(self.stream))
        if not self.frame_buf:
            raise RuntimeError("uvc_frame_buf_create failed")

        self._check(dll.uvc_camera_stream_start(ctypes.byref(self.stream), None), "uvc_camera_stream_start")

    def read_frame(self, timeout_s: float = 1.0) -> Optional[TC001SdkFrame]:
        if self.dll is None:
            raise RuntimeError("SDK camera is not open")
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            ret = self.dll.uvc_frame_get(ctypes.byref(self.raw_buf))
            if ret == 0:
                raw_bytes = np.frombuffer(bytes(self.raw_buf), dtype=np.uint8).reshape((self.HEIGHT, self.WIDTH, 2))
                preview_yuy2 = np.ascontiguousarray(raw_bytes[:TC001_HEIGHT, :, :])
                preview_bgr = None
                try:
                    preview_bgr = cv2.cvtColor(preview_yuy2, cv2.COLOR_YUV2BGR_YUY2)
                except cv2.error:
                    preview_bgr = None

                raw = raw_bytes.view("<u2").reshape((self.HEIGHT, self.WIDTH))
                temp_raw = raw[TC001_HEIGHT : self.HEIGHT, :]
                temp_c = temp_raw.astype(np.float32) / 64.0 - 273.15
                return TC001SdkFrame(temp_c=temp_c, preview_bgr=preview_bgr)
            time.sleep(0.005)
        return None

    def read_temp_c(self, timeout_s: float = 1.0) -> Optional[np.ndarray]:
        frame = self.read_frame(timeout_s=timeout_s)
        return frame.temp_c if frame is not None else None

    def close(self) -> None:
        if self.dll is None:
            return
        try:
            self.dll.uvc_camera_stream_close()
        except Exception:
            pass
        if self.frame_buf:
            try:
                self.dll.uvc_frame_buf_release(ctypes.c_void_p(self.frame_buf))
            except Exception:
                pass
        try:
            self.dll.uvc_camera_close()
        except Exception:
            pass
        try:
            self.dll.uvc_camera_release()
        except Exception:
            pass
        self.dll = None
