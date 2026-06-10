#!/usr/bin/env python3
"""
Trace TOPDON TopView's calls into libiruvc.dll.

This is used only to recover the vendor SDK call signatures that TopView uses
for TC001 Plus raw/radiometric streaming.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import frida


TOPVIEW_EXE = r"C:\Program Files\TOPDON\TopView\TopView.exe"


JS = r"""
const names = [
  "uvc_camera_init",
  "uvc_camera_list",
  "uvc_camera_info_get",
  "uvc_camera_open",
  "uvc_frame_buf_create",
  "uvc_camera_stream_start",
  "uvc_frame_get",
  "uvc_camera_stream_close",
  "uvc_frame_buf_release",
  "preview_start",
  "preview_stop",
  "y16_preview_start",
  "y16_preview_stop",
  "get_capture_data",
  "start_capture",
  "exit_capture",
  "vdcmd_init",
  "vdcmd_init_by_type",
  "tpd_get_point_temp_info",
  "tpd_get_max_min_temp_info"
];

function hexptr(p) {
  if (p === null || p.isNull()) return "0x0";
  return p.toString();
}

function dumpPtr(p, n) {
  try {
    if (p === null || p.isNull()) return null;
    return hexdump(p, { offset: 0, length: n, header: false, ansi: false });
  } catch (e) {
    return null;
  }
}

function hookOne(name) {
  const addr = Module.findExportByName("libiruvc.dll", name);
  if (addr === null) return false;
  Interceptor.attach(addr, {
    onEnter(args) {
      this.name = name;
      this.args = [];
      for (let i = 0; i < 8; i++) this.args.push(ptr(args[i]));
      const argText = this.args.map(hexptr);
      const dumps = {};
      for (let i = 0; i < 4; i++) dumps["arg" + i] = dumpPtr(this.args[i], 64);
      send({ event: "enter", name: name, args: argText, dumps: dumps });
    },
    onLeave(retval) {
      send({ event: "leave", name: this.name, retval: retval.toString() });
    }
  });
  send({ event: "hooked", name: name, addr: addr.toString() });
  return true;
}

let hooked = {};
function tryHookAll() {
  for (const name of names) {
    if (!hooked[name] && hookOne(name)) hooked[name] = true;
  }
}

tryHookAll();
const timer = setInterval(tryHookAll, 250);
"""


def on_message(message, data):
    if message["type"] == "send":
        print(json.dumps(message["payload"], ensure_ascii=False), flush=True)
    else:
        print(json.dumps(message, ensure_ascii=False), flush=True)


def main() -> int:
    timeout = float(sys.argv[1]) if len(sys.argv) > 1 else 25.0
    device = frida.get_local_device()
    env = dict(os.environ)
    env["__COMPAT_LAYER"] = "RunAsInvoker"
    proc = subprocess.Popen([TOPVIEW_EXE], cwd=r"C:\Program Files\TOPDON\TopView", env=env)
    pid = proc.pid
    time.sleep(1.0)
    session = device.attach(pid)
    script = session.create_script(JS)
    script.on("message", on_message)
    script.load()

    deadline = time.time() + timeout
    try:
        while time.time() < deadline:
            time.sleep(0.2)
    finally:
        try:
            session.detach()
        except Exception:
            pass
        try:
            subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True, text=True, timeout=5)
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
