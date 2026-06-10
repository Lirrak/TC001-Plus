#!/usr/bin/env python3
"""
Minimal libusb/UVC probe for TOPDON TC001 Plus raw thermal stream.

This targets the TOPDON/Realtek thermal device exposed as 0BDA:5830 by the
TOPDON TopView libusbK driver. It tries to select the 256x384 YUYV/Y16-like
UVC mode, collect one full frame from isochronous endpoint 0x81, and decode the
bottom 256x192 half as TOPDON absolute temperature:

    Celsius = raw_uint16 / 64 - 273.15

The script is intentionally separate from the OpenCV viewer until we prove the
raw stream can be acquired reliably on this Windows machine.
"""

from __future__ import annotations

import ctypes
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import libusb
import numpy as np


VID = 0x0BDA
PID = 0x5830
VC_INTERFACE = 0
VS_INTERFACE = 1
ALT_SETTING = 5
EP_ISO_IN = 0x81
WIDTH = 256
HEIGHT = 384
FPS = 25
FRAME_SIZE = WIDTH * HEIGHT * 2
PACKETS_PER_TRANSFER = 32
PACKET_SIZE = 3072
NUM_TRANSFERS = 8


@dataclass
class FrameState:
    current: bytearray
    frames: list[bytes]
    last_fid: Optional[int] = None
    packets: int = 0
    payload_bytes: int = 0
    errors: int = 0
    statuses: dict[int, int] | None = None


def check(code: int, what: str) -> None:
    if code < 0:
        raise RuntimeError(f"{what} failed: {code}")


def make_probe_commit() -> bytes:
    # UVC 1.1 probe/commit control, 26 bytes.
    #
    # Format/frame index are inferred from the 0BDA:5830 descriptors exposed by
    # TOPDON's SDK: mode 2 is 256x384 at 25fps, frame size 196608 bytes.
    interval_100ns = int(10_000_000 / FPS)
    buf = bytearray(26)
    buf[0:2] = (1).to_bytes(2, "little")  # bmHint: dwFrameInterval is fixed
    buf[2] = 1  # bFormatIndex
    buf[3] = 2  # bFrameIndex: 256x384
    buf[4:8] = interval_100ns.to_bytes(4, "little")
    buf[18:22] = FRAME_SIZE.to_bytes(4, "little")  # dwMaxVideoFrameSize
    buf[22:26] = PACKET_SIZE.to_bytes(4, "little")  # dwMaxPayloadTransferSize
    return bytes(buf)


def ctrl(handle, request_type: int, request: int, value: int, index: int, data: bytes | bytearray, timeout=1000) -> int:
    arr = (ctypes.c_ubyte * len(data)).from_buffer_copy(bytes(data))
    return libusb.control_transfer(
        handle,
        request_type,
        request,
        value,
        index,
        arr,
        len(data),
        timeout,
    )


def ctrl_in(handle, request_type: int, request: int, value: int, index: int, length: int, timeout=1000) -> bytes:
    arr = (ctypes.c_ubyte * length)()
    n = libusb.control_transfer(
        handle,
        request_type,
        request,
        value,
        index,
        arr,
        length,
        timeout,
    )
    if n < 0:
        raise RuntimeError(f"control_transfer IN failed: {n}")
    return bytes(arr[:n])


def parse_uvc_payload(packet: bytes, state: FrameState) -> None:
    if len(packet) < 2:
        return
    header_len = packet[0]
    if header_len < 2 or header_len > len(packet):
        return
    flags = packet[1]
    fid = flags & 1
    eof = bool(flags & 2)
    err = bool(flags & 0x40)

    state.packets += 1
    if err:
        state.errors += 1
        return

    if state.last_fid is None:
        state.last_fid = fid
    elif fid != state.last_fid:
        if len(state.current) >= FRAME_SIZE:
            state.frames.append(bytes(state.current[:FRAME_SIZE]))
        state.current.clear()
        state.last_fid = fid

    payload = packet[header_len:]
    if payload:
        state.current.extend(payload)
        state.payload_bytes += len(payload)

    if eof and len(state.current) >= FRAME_SIZE:
        state.frames.append(bytes(state.current[:FRAME_SIZE]))
        state.current.clear()


def decode_frame(frame: bytes) -> np.ndarray:
    raw = np.frombuffer(frame[:FRAME_SIZE], dtype="<u2").reshape((HEIGHT, WIDTH))
    temp_raw = raw[HEIGHT // 2 :, :]
    return temp_raw.astype(np.float32) / 64.0 - 273.15


def set_iso_packet_lengths_direct(transfer, packet_count: int, packet_size: int) -> None:
    base = ctypes.addressof(transfer.contents) + ctypes.sizeof(libusb.transfer)
    desc_array_type = libusb.iso_packet_descriptor * packet_count
    descs = desc_array_type.from_address(base)
    for desc in descs:
        desc.length = packet_size


def iso_packet_descriptors(transfer, packet_count: int):
    base = ctypes.addressof(transfer.contents) + ctypes.sizeof(libusb.transfer)
    desc_array_type = libusb.iso_packet_descriptor * packet_count
    return desc_array_type.from_address(base)


def main() -> int:
    ctx = ctypes.POINTER(libusb.context)()
    check(libusb.init(ctypes.byref(ctx)), "libusb.init")

    handle = libusb.open_device_with_vid_pid(ctx, VID, PID)
    if not handle:
        raise RuntimeError("Could not open 0BDA:5830. Close TopView and reconnect TC001 Plus.")

    transfers = []
    buffers = []
    callbacks = []
    state = FrameState(current=bytearray(), frames=[], statuses={})

    try:
        # Interface 0 is VideoControl, interface 1 is VideoStreaming.
        check(libusb.claim_interface(handle, VC_INTERFACE), "claim_control_interface")
        check(libusb.claim_interface(handle, VS_INTERFACE), "claim_interface")

        probe = make_probe_commit()
        # SET_CUR VS_PROBE_CONTROL, then VS_COMMIT_CONTROL.
        check(ctrl(handle, 0x21, 0x01, 0x0100, VS_INTERFACE, probe), "SET_CUR PROBE")
        accepted = ctrl_in(handle, 0xA1, 0x81, 0x0100, VS_INTERFACE, len(probe))
        print(f"accepted_probe={accepted.hex()}")
        check(ctrl(handle, 0x21, 0x01, 0x0200, VS_INTERFACE, accepted), "SET_CUR COMMIT")

        check(libusb.set_interface_alt_setting(handle, VS_INTERFACE, ALT_SETTING), "set altsetting")

        cb_type = libusb.transfer_cb_fn

        def on_transfer(transfer):
            try:
                t = transfer.contents
                if t.status == libusb.LIBUSB_TRANSFER_COMPLETED:
                    descs = iso_packet_descriptors(transfer, t.num_iso_packets)
                    data_base = ctypes.addressof(t.buffer.contents)
                    for idx in range(t.num_iso_packets):
                        desc = descs[idx]
                        if desc.status != libusb.LIBUSB_TRANSFER_COMPLETED or desc.actual_length <= 0:
                            continue
                        packet = ctypes.string_at(data_base + idx * PACKET_SIZE, desc.actual_length)
                        parse_uvc_payload(packet, state)
                elif state.statuses is not None:
                    state.statuses[int(t.status)] = state.statuses.get(int(t.status), 0) + 1
                if len(state.frames) < 1:
                    libusb.submit_transfer(transfer)
            except Exception:
                state.errors += 1

        for _ in range(NUM_TRANSFERS):
            transfer = libusb.alloc_transfer(PACKETS_PER_TRANSFER)
            if not transfer:
                raise RuntimeError("alloc_transfer failed")
            buf = (ctypes.c_ubyte * (PACKETS_PER_TRANSFER * PACKET_SIZE))()
            cb = cb_type(on_transfer)
            libusb.fill_iso_transfer(
                transfer,
                handle,
                EP_ISO_IN,
                buf,
                len(buf),
                PACKETS_PER_TRANSFER,
                cb,
                None,
                1000,
            )
            set_iso_packet_lengths_direct(transfer, PACKETS_PER_TRANSFER, PACKET_SIZE)
            transfers.append(transfer)
            buffers.append(buf)
            callbacks.append(cb)
            check(libusb.submit_transfer(transfer), "submit_transfer")

        deadline = time.time() + 5.0
        completed = ctypes.c_long(0)
        timeval = libusb.timeval(0, 50_000)
        event_results: dict[int, int] = {}
        while time.time() < deadline and not state.frames:
            rc = libusb.handle_events_timeout_completed(ctx, ctypes.byref(timeval), ctypes.byref(completed))
            event_results[int(rc)] = event_results.get(int(rc), 0) + 1

        print(
            f"packets={state.packets} payload_bytes={state.payload_bytes} "
            f"errors={state.errors} statuses={state.statuses} events={event_results} "
            f"frames={len(state.frames)}"
        )
        if not state.frames:
            print("No complete frame captured.")
            return 2

        temp_c = decode_frame(state.frames[0])
        print(
            "Temp C: "
            f"min={np.nanmin(temp_c):.2f}, "
            f"avg={np.nanmean(temp_c):.2f}, "
            f"center={temp_c[temp_c.shape[0] // 2, temp_c.shape[1] // 2]:.2f}, "
            f"max={np.nanmax(temp_c):.2f}"
        )

        out = Path("tc001_libusb_last_frame.raw")
        out.write_bytes(state.frames[0])
        print(f"Saved raw frame: {out.resolve()}")
        return 0
    finally:
        for transfer in transfers:
            try:
                libusb.cancel_transfer(transfer)
            except Exception:
                pass
        try:
            libusb.set_interface_alt_setting(handle, VS_INTERFACE, 0)
        except Exception:
            pass
        try:
            libusb.release_interface(handle, VS_INTERFACE)
        except Exception:
            pass
        try:
            libusb.release_interface(handle, VC_INTERFACE)
        except Exception:
            pass
        try:
            libusb.close(handle)
        except Exception:
            pass
        libusb.exit(ctx)


if __name__ == "__main__":
    raise SystemExit(main())
