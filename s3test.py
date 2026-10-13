"""Vypíše, co účet vidí na S3 (jen čtení, klíče se nikam nevypisují).

    python s3list.py                       # seznam bucketů
    python s3list.py bp_mahdal_bukov       # obsah bucketu (do 2 úrovní)
    python s3list.py bp_mahdal_bukov/temperature_monitoring.zarr
"""
import os
import sys

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import s3fs

endpoint = os.environ.get("ZF_S3_ENDPOINT_URL", "https://s3.cl2.du.cesnet.cz")
for k in ("ZF_S3_ACCESS_KEY", "ZF_S3_SECRET_KEY"):
    if k not in os.environ:
        sys.exit(f"Chybí {k} (v .env nebo v prostředí)")

fs = s3fs.S3FileSystem(
    key=os.environ["ZF_S3_ACCESS_KEY"],
    secret=os.environ["ZF_S3_SECRET_KEY"],
    endpoint_url=endpoint,
)
print("Endpoint:", endpoint)

path = sys.argv[1] if len(sys.argv) > 1 else ""
try:
    items = fs.ls(path)
except Exception as e:  # noqa: BLE001
    sys.exit(f"Nelze vypsat '{path}': {type(e).__name__}: {e}")

if not items:
    print(f"'{path}': prázdné nebo neexistuje")
for item in items:
    print(item)
    if path and fs.isdir(item):  # ještě jedna úroveň
        try:
            for sub in fs.ls(item)[:15]:
                print("   ", sub)
        except Exception:  # noqa: BLE001
            pass