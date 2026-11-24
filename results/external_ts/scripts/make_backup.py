"""Tier-1 backup of results/external_ts/ : zip + SHA-256 manifest, verified, never overwriting.

* Manifest: every file under results/external_ts (relative path, size, SHA-256), sorted.
* Archive : <dest>/external_ts_tier1_<UTC>.zip (+ MANIFEST.sha256 inside), written to a temp name,
            fsync'd, verified, then renamed. An existing name is never reused or overwritten.
* Verify  : zip CRC test, every member re-hashed against the manifest, member set == manifest set,
            and the manifest re-checked against the live files (detects changes during the backup).
* Sidecar : <archive>.sha256 (SHA-256 of the zip) and <archive>.manifest.sha256 (sha256sum format).
Older backups are never touched. Read-only on the project.

Usage: python -B -I make_backup.py --dest C:/Anvith/FINALYEARPROJECTTHRASHA/backups/external_ts [--verify-only <zip>]
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import zipfile
from pathlib import Path

SRC = Path(__file__).resolve().parents[1]          # results/external_ts
PROJECT = SRC.parents[1]


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def build_manifest() -> list[tuple[str, int, str]]:
    out = []
    for p in sorted(SRC.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts:
            rel = p.relative_to(PROJECT).as_posix()
            out.append((rel, p.stat().st_size, sha_file(p)))
    return out


def manifest_text(m: list[tuple[str, int, str]]) -> str:
    return "".join(f"{h}  {rel}\n" for rel, _, h in m)


def verify(zip_path: Path, expect_live: bool) -> dict:
    with zipfile.ZipFile(zip_path) as z:
        bad = z.testzip()
        man = z.read("MANIFEST.sha256").decode()
        entries = {ln[66:]: ln[:64] for ln in man.splitlines() if ln}
        members = set(z.namelist()) - {"MANIFEST.sha256", "BACKUP_INFO.json"}
        mism = [n for n in entries if sha_bytes(z.read(n)) != entries[n]]
        info = json.loads(z.read("BACKUP_INFO.json"))
    res = {"zip": str(zip_path), "zip_sha256": sha_file(zip_path), "files": len(entries), "crc_ok": bad is None,
           "members_equal_manifest": members == set(entries), "member_hash_mismatches": mism,
           "manifest_sha256": sha_bytes(man.encode()), "info": info}
    if expect_live:
        res["live_changed_since_manifest"] = [n for n, h in entries.items()
                                              if not (PROJECT / n).is_file() or sha_file(PROJECT / n) != h]
    res["ok"] = res["crc_ok"] and res["members_equal_manifest"] and not mism and not res.get("live_changed_since_manifest")
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", type=Path)
    ap.add_argument("--verify-only", type=Path)
    a = ap.parse_args(argv)
    if a.verify_only:
        r = verify(a.verify_only, expect_live=False)
        print(json.dumps(r, indent=1))
        return 0 if r["ok"] else 1
    dest = a.dest.resolve()
    if dest == SRC or SRC in dest.parents:
        raise SystemExit("destination must be outside results/external_ts")
    dest.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    final = dest / f"external_ts_tier1_{stamp}.zip"
    if final.exists():
        raise SystemExit(f"{final} exists - refusing to overwrite")
    m = build_manifest()
    man = manifest_text(m)
    info = {"created_utc": stamp, "source": str(SRC), "files": len(m), "bytes": sum(s for _, s, _ in m),
            "manifest_sha256": sha_bytes(man.encode()),
            "eval_subjects_done": sorted(p.stem for p in (SRC / "eval/subjects").glob("*.done"))}
    tmp = dest / f".tmp_{final.name}.{os.getpid()}"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for rel, _, _ in m:
            z.write(PROJECT / rel, rel)
        z.writestr("MANIFEST.sha256", man)
        z.writestr("BACKUP_INFO.json", json.dumps(info, indent=1, sort_keys=True))
    with open(tmp, "rb+") as f:
        os.fsync(f.fileno())
    r = verify(tmp, expect_live=True)
    if not r["ok"]:
        raise SystemExit(f"verification FAILED - temp archive kept for inspection at {tmp}: {r}")
    os.replace(tmp, final)
    r["zip"] = str(final)
    # bytes, so Windows text mode cannot turn "\n" into "\r\n" (sha256sum -c needs LF)
    (dest / f"{final.name}.sha256").write_bytes(f"{r['zip_sha256']}  {final.name}\n".encode())
    (dest / f"{final.name}.manifest.sha256").write_bytes(man.encode())
    print(json.dumps({k: v for k, v in r.items() if k != "info"} | {"info": info}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
