"""Resolve Malaysian-STT audio, which lives inside multi-GB zip archives on the Hub.

The dataset's ``audio_filenames`` (e.g. ``malaysian-segment/0-0.mp3``) are members of large
zips (``malaysian-segment-*.zip``, ~4.9 GB each, ~345k members). Rather than download whole
archives, we use ``remotezip`` to read the central directory + just the bytes of each needed
member over HTTP range requests -- light on disk, works on a small pod.
"""

from __future__ import annotations

import io

import numpy as np

REPO_BASE = "https://huggingface.co/datasets/malaysia-ai/Malaysian-STT/resolve/main/"


def discover_zip_names(prefix: str, token: str | None = None) -> list[str]:
    """List archive names in the repo that start with ``prefix`` (e.g. ``malaysian-segment``)."""
    from huggingface_hub import list_repo_files

    files = list_repo_files("malaysia-ai/Malaysian-STT", repo_type="dataset", token=token)
    return sorted(f for f in files if f.startswith(prefix) and f.endswith(".zip"))


class ZipAudioResolver:
    """Lazily index a set of remote zips and read/decode members on demand.

    Zips are indexed (namelist only) one at a time until a requested member is found, so a
    small run that only touches the first archive never fetches the others' directories.
    """

    def __init__(self, zip_names: list[str], token: str | None = None):
        self.zip_names = list(zip_names)
        self.headers = {"Authorization": f"Bearer {token}"} if token else {}
        self._open: dict[str, object] = {}
        self._index: dict[str, str] = {}   # member path -> zip name
        self._indexed: set[str] = set()

    def _remote(self, name: str):
        if name not in self._open:
            from remotezip import RemoteZip

            self._open[name] = RemoteZip(REPO_BASE + name, headers=self.headers)
        return self._open[name]

    def index_zip(self, name: str) -> int:
        """Read one archive's namelist into the index. Returns members added."""
        if name in self._indexed:
            return 0
        z = self._remote(name)
        added = 0
        for member in z.namelist():
            if member not in self._index:
                self._index[member] = name
                added += 1
        self._indexed.add(name)
        return added

    def available_members(self) -> set[str]:
        return set(self._index)

    def read_bytes(self, member: str) -> bytes:
        if member not in self._index:
            for name in self.zip_names:
                if name not in self._indexed:
                    self.index_zip(name)
                    if member in self._index:
                        break
        if member not in self._index:
            raise KeyError(member)
        return self._remote(self._index[member]).read(member)

    def read_audio(self, member: str) -> tuple[np.ndarray, int]:
        import soundfile as sf

        arr, sr = sf.read(io.BytesIO(self.read_bytes(member)), dtype="float32", always_2d=False)
        return np.asarray(arr), int(sr)

    def close(self) -> None:
        for z in self._open.values():
            try:
                z.close()
            except Exception:  # noqa: BLE001
                pass


class DownloadZipResolver:
    """Download whole zip archives (Xet-accelerated) and read members from them locally.

    Local reads (~1 ms) are far cheaper than per-member HTTP range requests (~0.3 s each),
    so once the archives are resident, throughput is bound by decode/encode, not the network.
    With ``in_ram=True`` the zip bytes are held in memory (the pod has ~1 TB RAM) and the
    on-disk copy is deleted, so many archives fit without the 20 GB disk cap biting.
    """

    def __init__(self, zip_names: list[str], token: str | None = None, in_ram: bool = True,
                 workdir: str = "/root/data/zdl"):
        import io
        import os
        import zipfile

        from huggingface_hub import hf_hub_download

        os.makedirs(workdir, exist_ok=True)
        self._zips: dict[str, zipfile.ZipFile] = {}
        self._index: dict[str, str] = {}
        for name in zip_names:
            local = hf_hub_download("malaysia-ai/Malaysian-STT", name, repo_type="dataset",
                                    token=token, local_dir=workdir)
            if in_ram:
                with open(local, "rb") as f:
                    data = f.read()
                os.remove(local)
                zf = zipfile.ZipFile(io.BytesIO(data))
            else:
                zf = zipfile.ZipFile(local)
            self._zips[name] = zf
            for m in zf.namelist():
                self._index.setdefault(m, name)
            print(f"[dl] {name}: {len(zf.namelist())} members "
                  f"(index now {len(self._index)})", flush=True)

    def available_members(self) -> set[str]:
        return set(self._index)

    def read_audio(self, member: str) -> tuple[np.ndarray, int]:
        import soundfile as sf

        if member not in self._index:
            raise KeyError(member)
        data = self._zips[self._index[member]].read(member)
        arr, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
        return np.asarray(arr), int(sr)

    def close(self) -> None:
        for z in self._zips.values():
            try:
                z.close()
            except Exception:  # noqa: BLE001
                pass
