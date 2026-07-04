#!/usr/bin/env python3
"""Minimal RunPod control plane for the Semantic-VAD build (stdlib only).

Runs on the laptop making only tiny API calls -- all heavy work happens on the pod.
Reads secrets from ``.env`` (RUNPOD_API_KEY, HF_TOKEN) and never prints their values.

Subcommands:
    create      launch a US CPU pod with SSH (PUBLIC_KEY) enabled; saves deploy/pod.json
    wait        poll until publicIp + SSH port mapping are ready; prints host/port
    status      print current pod status JSON (secrets stripped)
    ssh-info    print "host port" for the SSH connection
    terminate   delete the pod
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(ROOT, ".env")
POD_JSON = os.path.join(ROOT, "deploy", "pod.json")
PUBKEY_PATH = os.path.join(ROOT, "deploy", "runpod_key.pub")
API = "https://rest.runpod.io/v1"

US_DCS = ["US-IL-1", "US-TX-3", "US-TX-1", "US-KS-2", "US-GA-1",
          "US-NC-1", "US-CA-2", "US-WA-1", "US-DE-1"]


def load_env() -> dict[str, str]:
    env: dict[str, str] = {}
    if os.path.exists(ENV_PATH):
        for line in open(ENV_PATH):
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def api(method: str, path: str, key: str, body: dict | None = None) -> dict:
    url = f"{API}{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {key}")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            raw = r.read().decode()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        detail = e.read().decode()
        raise SystemExit(f"RunPod API {method} {path} failed: {e.code} {detail}")


def save_pod(pod: dict) -> None:
    os.makedirs(os.path.dirname(POD_JSON), exist_ok=True)
    with open(POD_JSON, "w") as f:
        json.dump({"id": pod.get("id"), "name": pod.get("name")}, f, indent=2)


def pod_id() -> str:
    if not os.path.exists(POD_JSON):
        raise SystemExit("no deploy/pod.json -- run `create` first")
    return json.load(open(POD_JSON))["id"]


def ssh_endpoint(pod: dict) -> tuple[str, int] | None:
    ip = pod.get("publicIp") or ""
    pm = pod.get("portMappings") or {}
    port = pm.get("22") or pm.get(22)
    if ip and port:
        return ip, int(port)
    return None


def cmd_create(args, env, key):
    pub = open(PUBKEY_PATH).read().strip()
    penv = {"PUBLIC_KEY": pub}
    if env.get("HF_TOKEN"):
        penv["HF_TOKEN"] = env["HF_TOKEN"]
        penv["HUGGING_FACE_HUB_TOKEN"] = env["HF_TOKEN"]
    body = {
        "name": args.name,
        "computeType": "CPU",
        "cloudType": "SECURE",
        "imageName": args.image,
        "containerDiskInGb": args.disk,
        "volumeInGb": 0,  # no /workspace network volume -- work under / (container disk)
        "ports": ["22/tcp"],
        "cpuFlavorIds": [args.flavor],
        "dataCenterIds": US_DCS,
        "env": penv,
    }
    pod = api("POST", "/pods", key, body)
    save_pod(pod)
    print(f"created pod id={pod.get('id')} status={pod.get('desiredStatus')}")


def cmd_status(args, env, key):
    pod = api("GET", f"/pods/{pod_id()}", key)
    pod.pop("env", None)  # strip secrets
    print(json.dumps(pod, indent=2)[:2000])


def cmd_wait(args, env, key):
    pid = pod_id()
    deadline = time.time() + args.timeout
    while time.time() < deadline:
        pod = api("GET", f"/pods/{pid}", key)
        ep = ssh_endpoint(pod)
        status = pod.get("desiredStatus")
        if ep:
            print(f"READY {ep[0]} {ep[1]} (status={status})")
            return
        print(f"waiting... status={status} ip={pod.get('publicIp')!r}", flush=True)
        time.sleep(args.interval)
    raise SystemExit("timed out waiting for pod SSH endpoint")


def cmd_ssh_info(args, env, key):
    pod = api("GET", f"/pods/{pod_id()}", key)
    ep = ssh_endpoint(pod)
    if not ep:
        raise SystemExit("SSH endpoint not ready")
    print(f"{ep[0]} {ep[1]}")


def cmd_terminate(args, env, key):
    api("DELETE", f"/pods/{pod_id()}", key)
    print("terminated")
    if os.path.exists(POD_JSON):
        os.remove(POD_JSON)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("create")
    c.add_argument("--name", default="semantic-vad-cpu")
    c.add_argument("--image", default="runpod/base:0.5.1-cpu")
    c.add_argument("--flavor", default="cpu3g")
    c.add_argument("--disk", type=int, default=60)
    c.set_defaults(func=cmd_create)

    for name, fn in [("status", cmd_status), ("ssh-info", cmd_ssh_info),
                     ("terminate", cmd_terminate)]:
        s = sub.add_parser(name)
        s.set_defaults(func=fn)

    w = sub.add_parser("wait")
    w.add_argument("--timeout", type=int, default=300)
    w.add_argument("--interval", type=int, default=8)
    w.set_defaults(func=cmd_wait)

    args = p.parse_args()
    env = load_env()
    key = env.get("RUNPOD_API_KEY")
    if not key:
        raise SystemExit("RUNPOD_API_KEY not found in .env")
    args.func(args, env, key)


if __name__ == "__main__":
    main()
