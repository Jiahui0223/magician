"""Read only the PLY header (first bytes) of every Mesh.ply / Block.ply inside the HF mesh zip,
plus each scene's metadata.xml, without downloading the meshes.

python -I remote_ply_headers.py <out_json>
"""
import json
import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from http_zip import HTTPRangeFile, BASE  # noqa: E402
import zipfile

out = sys.argv[1]
f = HTTPRangeFile(BASE + "terra_3dmap_pointcloud_mesh.zip", block=256 << 10)
z = zipfile.ZipFile(f)
res = {}
for info in z.infolist():
    n = info.filename
    if n.endswith("/Mesh.ply") or n.endswith("metadata.xml") or n.endswith("/cloud_merged.ply"):
        with z.open(info) as fh:
            head = fh.read(4096 if n.endswith(".ply") else 2048)
        txt = head.decode("latin1")
        if n.endswith(".ply"):
            hdr = txt.split("end_header")[0]
            rec = {"bytes": info.file_size,
                   "format": re.search(r"format (\S+)", hdr).group(1),
                   "vertex": int(re.search(r"element vertex (\d+)", hdr).group(1)),
                   "face": int(re.search(r"element face (\d+)", hdr).group(1)) if "element face" in hdr else 0,
                   "n_texture_comments": hdr.count("TextureFile"),
                   "properties": re.findall(r"property (?:list \S+ \S+|\S+) (\S+)", hdr)}
        else:
            rec = {"srs": re.search(r"<SRS>(.*?)</SRS>", txt).group(1),
                   "srs_origin": re.search(r"<SRSOrigin>(.*?)</SRSOrigin>", txt).group(1)}
        res[n] = rec
        print(n, rec)
json.dump(res, open(out, "w"), indent=1)
print("requests", f.nreq, "bytes", f.nbytes)
