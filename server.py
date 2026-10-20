#!/usr/bin/env python3
"""
Spike: FLUX.2 [klein] 4B on CPU with stable-diffusion.cpp.
POST /generate with the prompt as the body, PNG in response.

On start it downloads the models (once, into SD_MODEL_DIR), starts sd-server on localhost
and forwards requests to it. Settings are environment variables, see below. Command line
arguments are ignored.
"""
import base64
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

# flag: (environment variable, default "<owner>/<repo>/<file>" on Hugging Face, its sha256 checked after download)
MODELS = {
    "--diffusion-model": ("SD_DIFFUSION_MODEL", "leejet/FLUX.2-klein-4B-GGUF/flux-2-klein-4b-Q4_0.gguf",
                          "d1023499ef3f2f82ff7c50e6778495195c1b6cc34835741778868428111f9ff4"),
    "--llm": ("SD_LLM", "unsloth/Qwen3-4B-GGUF/Qwen3-4B-Q4_K_M.gguf",
              "f6f851777709861056efcdad3af01da38b31223a3ba26e61a4f8bf3a2195813a"),
    "--vae": ("SD_VAE", "black-forest-labs/FLUX.2-small-decoder/full_encoder_small_decoder.safetensors",
              "ea4273f02d1fafbf8e1d1c2cf6018ed8748652eb0bf34f2dd91171f16f15ab62"),
}
MODEL_DIR = os.getenv("SD_MODEL_DIR", os.path.expanduser("~/.cache/huggingface/sdcpp"))
# more sd-server arguments, e.g. "-v" or "-W 768 -H 768 --steps 8": the last copy of a flag wins
EXTRA_ARGS = os.getenv("SD_EXTRA_ARGS", "").split()

HOST, PORT = "0.0.0.0", 8081
BACKEND = "http://127.0.0.1:1234"  # sd-server's default --listen-ip and --listen-port
SD_SERVER = "/opt/sd.cpp/sd-server"


def download(spec, sha256):
    owner, repo, name = spec.split("/", 2)
    path = os.path.join(MODEL_DIR, os.path.basename(name))
    if os.path.exists(path):
        return path
    os.makedirs(MODEL_DIR, exist_ok=True)
    part = path + ".part"
    have = os.path.getsize(part) if os.path.exists(part) else 0
    url = f"https://huggingface.co/{owner}/{repo}/resolve/main/{name}"
    req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
    with urllib.request.urlopen(req, timeout=60) as r:
        if r.status != 206:
            have = 0
        left = int(r.headers["Content-Length"])
        total = have + left
        free = shutil.disk_usage(MODEL_DIR).free
        if free < left + 2**28:
            sys.exit(f"not enough disk space for {name}: need {left / 2**30:.1f} GB, free {free / 2**30:.1f} GB")
        print(f"downloading {spec} ({total / 2**30:.2f} GB)", flush=True)
        last = time.monotonic()
        with open(part, "ab" if have else "wb") as f:
            while chunk := r.read(2**20):
                f.write(chunk)
                have += len(chunk)
                if time.monotonic() - last > 30:
                    print(f"  {have * 100 // total}% of {name}", flush=True)
                    last = time.monotonic()
    if sha256:
        with open(part, "rb") as f:
            digest = hashlib.file_digest(f, "sha256").hexdigest()
        if digest != sha256:
            os.remove(part)
            sys.exit(f"sha256 mismatch for {name}: got {digest}, deleted the file")
    os.rename(part, path)
    return path


def start_backend():
    # --seed -1: sd-server picks a new random seed for every image
    cmd = [SD_SERVER, "--cfg-scale", "1.0", "--steps", "4", "-W", "512", "-H", "512",
           "--seed", "-1", "--mmap", "--flash-attn", "off,diffusion=on"]
    for flag, (env, default, sha256) in MODELS.items():
        spec = os.getenv(env, default)
        cmd += [flag, download(spec, sha256 if spec == default else None)]
    cmd += EXTRA_ARGS
    print("starting", " ".join(cmd), flush=True)
    backend = subprocess.Popen(cmd)

    def exit_with_backend():
        code = backend.wait()
        print(f"sd-server exited with code {code}", flush=True)
        os._exit(1)

    threading.Thread(target=exit_with_backend, daemon=True).start()
    while True:
        try:
            urllib.request.urlopen(BACKEND + "/v1/models", timeout=5).close()
            return
        except OSError:
            time.sleep(1)


def generate(prompt):
    # size and format: the defaults of sd-server, -W/-H and png
    body = json.dumps({"prompt": prompt}).encode("utf-8")
    req = urllib.request.Request(BACKEND + "/v1/images/generations", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        return base64.b64decode(json.load(r)["data"][0]["b64_json"])


class GenerationServer(BaseHTTPRequestHandler):
    def do_POST(self):
        if not self.path.startswith("/generate"):
            self.send_error(404, "Not found")
            return
        prompt = self.rfile.read(int(self.headers["Content-Length"])).decode("utf-8")
        print("inference..", flush=True)
        start = time.monotonic()
        try:
            png = generate(prompt)
        except Exception as e:
            print(f"inference failed: {e!r}", flush=True)
            self.send_error(500)
            return
        print(f"inference done in {time.monotonic() - start:.1f}s", flush=True)
        self.send_response(200)
        self.send_header("Content-type", "image/png")
        self.send_header("Content-Length", str(len(png)))
        self.end_headers()
        self.wfile.write(png)


if __name__ == "__main__":
    # PID 1 in a container gets no default SIGTERM handler: exit on docker stop (and quietly on
    # Ctrl-C). When PID 1 exits, the kernel kills sd-server.
    for sig in signal.SIGTERM, signal.SIGINT:
        signal.signal(sig, lambda *_: sys.exit(0))
    start_backend()
    print(f"Server started http://{HOST}:{PORT}", flush=True)
    HTTPServer((HOST, PORT), GenerationServer).serve_forever()
