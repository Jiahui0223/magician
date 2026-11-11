"""Partial (HTTP Range) access to a remote zip on Hugging Face.

EXP7 / UAVScenes feasibility (Step 5.1).  The HF dataset ships each modality
as one monolithic zip (6.8 GB mesh zip, 28.7 GB CAM_LIDAR zip), so we read the
central directory and individual members with HTTP Range requests instead of
downloading the whole archive.

Usage:
  python -I http_zip.py list    <repo_file> <out_tsv>
  python -I http_zip.py extract <repo_file> <out_dir> <member> [<member> ...]
  python -I http_zip.py extract_list <repo_file> <out_dir> <members_txt>

Every extracted member is CRC-checked by zipfile and its sha256 is appended to
<out_dir>/_provenance.tsv together with the exact source URL.
"""
import hashlib
import io
import os
import sys
import time
import zipfile

import requests

REPO = "sijieaaa/UAVScenes"
REV = "04a224fd14b0b8fdfd2633e68f20471b85c611a5"  # HF dataset commit (2025-08-06)
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/{REV}/"


class HTTPRangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP Range with a small block cache."""

    def __init__(self, url, block=4 << 20):
        self.url = url
        self.block = block
        self.sess = requests.Session()
        self.sess.headers["User-Agent"] = "generic-research-client/1.0"
        self._resolve()
        self.pos = 0
        self.cache = {}
        self.nreq = 0
        self.nbytes = 0

    def _resolve(self):
        r = self.sess.head(self.url, allow_redirects=False, timeout=60)
        r.raise_for_status() if r.status_code not in (301, 302, 307) else None
        self.cdn = r.headers.get("Location", self.url)
        self.size = int(r.headers.get("X-Linked-Size") or r.headers["Content-Length"])
        self.etag = r.headers.get("X-Linked-ETag", "").strip('"')

    def _get(self, a, b):  # inclusive range
        for attempt in range(6):
            try:
                r = self.sess.get(self.cdn, headers={"Range": f"bytes={a}-{b}"}, timeout=120)
                if r.status_code in (401, 403):  # signed URL expired
                    self._resolve()
                    continue
                if r.status_code != 206:
                    raise IOError(f"HTTP {r.status_code} for range {a}-{b}")
                data = r.content
                if len(data) != b - a + 1:
                    raise IOError(f"short read {len(data)} != {b - a + 1}")
                self.nreq += 1
                self.nbytes += len(data)
                return data
            except (requests.RequestException, IOError) as e:
                print(f"  retry {attempt}: {e}", file=sys.stderr)
                time.sleep(2 + 3 * attempt)
        raise IOError("giving up")

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        if whence == 0:
            self.pos = off
        elif whence == 1:
            self.pos += off
        else:
            self.pos = self.size + off
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = max(0, min(n, self.size - self.pos))
        if n == 0:
            return b""
        if n > 4 * self.block:  # large member payload: fetch directly
            data = self._get(self.pos, self.pos + n - 1)
            self.pos += n
            return data
        out = bytearray()
        while n > 0:
            bi = self.pos // self.block
            if bi not in self.cache:
                a = bi * self.block
                b = min(self.size, a + self.block) - 1
                if len(self.cache) > 8:
                    self.cache.clear()
                self.cache[bi] = self._get(a, b)
            blk = self.cache[bi]
            o = self.pos - bi * self.block
            take = min(n, len(blk) - o)
            out += blk[o:o + take]
            self.pos += take
            n -= take
        return bytes(out)

    def readinto(self, b):
        d = self.read(len(b))
        b[:len(d)] = d
        return len(d)


def open_zip(repo_file):
    f = HTTPRangeFile(BASE + repo_file)
    return f, zipfile.ZipFile(f)


def cmd_list(repo_file, out_tsv):
    f, z = open_zip(repo_file)
    infos = z.infolist()
    with open(out_tsv, "w") as fo:
        fo.write("filename\tfile_size\tcompress_size\tcompress_type\theader_offset\n")
        for i in infos:
            fo.write(f"{i.filename}\t{i.file_size}\t{i.compress_size}\t{i.compress_type}\t{i.header_offset}\n")
    print(f"{repo_file}: size={f.size} etag(sha256)={f.etag} members={len(infos)} "
          f"requests={f.nreq} bytes_fetched={f.nbytes}")


def cmd_extract(repo_file, out_dir, members):
    os.makedirs(out_dir, exist_ok=True)
    f, z = open_zip(repo_file)
    prov = os.path.join(out_dir, "_provenance.tsv")
    new = not os.path.exists(prov)
    with open(prov, "a") as fp:
        if new:
            fp.write("member\tbytes\tsha256\tsource_zip\tsource_zip_sha256\turl\n")
        for m in members:
            dst = os.path.join(out_dir, m)
            if not os.path.realpath(dst).startswith(os.path.realpath(out_dir)):
                raise ValueError(f"unsafe member path {m}")
            if os.path.exists(dst):
                print(f"skip existing {m}")
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            t0 = time.time()
            h = hashlib.sha256()
            n = 0
            with z.open(m) as src, open(dst + ".part", "wb") as fo:  # CRC checked on EOF
                while True:
                    chunk = src.read(8 << 20)
                    if not chunk:
                        break
                    h.update(chunk)
                    fo.write(chunk)
                    n += len(chunk)
            os.replace(dst + ".part", dst)
            fp.write(f"{m}\t{n}\t{h.hexdigest()}\t{repo_file}\t{f.etag}\t{BASE + repo_file}\n")
            fp.flush()
            print(f"{m}: {n} B sha256={h.hexdigest()} {time.time() - t0:.1f}s")
    print(f"requests={f.nreq} bytes_fetched={f.nbytes}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "list":
        cmd_list(sys.argv[2], sys.argv[3])
    elif cmd == "extract":
        cmd_extract(sys.argv[2], sys.argv[3], sys.argv[4:])
    elif cmd == "extract_list":
        mem = [l.strip() for l in open(sys.argv[4]) if l.strip()]
        cmd_extract(sys.argv[2], sys.argv[3], mem)
    else:
        raise SystemExit(__doc__)
