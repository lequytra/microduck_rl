#!/usr/bin/env python3
"""Drive infer_policy.py via a pty so keys can be injected through a FIFO.

Child's stdin is a pty slave (isatty() -> True, keyboard control enabled).
We forward bytes from /tmp/duckkeys.fifo to the pty master.
"""
import os, pty, subprocess, sys, threading

FIFO = "/tmp/duckkeys.fifo"
CMD = sys.argv[1:]

# mjpython dlopens the venv python and needs libpython3.12.dylib findable;
# setting it here avoids SIP-protected intermediaries (nohup, /usr/bin/python3)
# scrubbing DYLD_* from the environment.
_pylib = os.path.expanduser(
    "~/.local/share/uv/python/cpython-3.12.12-macos-aarch64-none/lib")
os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = (
    _pylib + ":" + os.environ.get("DYLD_FALLBACK_LIBRARY_PATH", ""))

master, slave = pty.openpty()
proc = subprocess.Popen(CMD, stdin=slave, close_fds=True)
os.close(slave)

if not os.path.exists(FIFO):
    os.mkfifo(FIFO)

def pump():
    while True:
        # Reopen each time the writer side closes (echo > fifo semantics)
        with open(FIFO, "rb", buffering=0) as f:
            while True:
                data = f.read(64)
                if not data:
                    break
                os.write(master, data)
        if proc.poll() is not None:
            break

t = threading.Thread(target=pump, daemon=True)
t.start()
proc.wait()
try:
    os.unlink(FIFO)
except FileNotFoundError:
    pass
