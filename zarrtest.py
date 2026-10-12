"""Test čtení tenzorových polí z úložiště zarr_fuse (S3 / lokální).

Nastavení se čte z proměnných prostředí ZF_* nebo ze souboru .env
(ten nikdy nedávej do gitu). Úložiště se otevírá jen pro čtení.

    python zarrtest.py [schema.yaml]
"""
import os
import sys

try:
    from dotenv import load_dotenv
    load_dotenv()  # načte .env z aktuální složky, pokud existuje
except ImportError:
    pass  # bez python-dotenv stačí proměnné nastavit v terminálu

import zarr_fuse


def main() -> None:
    store_url = os.environ.get("ZF_STORE_URL", "")
    if store_url.startswith("s3://"):
        missing = [k for k in ("ZF_S3_ACCESS_KEY", "ZF_S3_SECRET_KEY")
                   if k not in os.environ]
        if missing:
            sys.exit(f"Chybí proměnné: {', '.join(missing)} (viz .env.example)")

    if len(sys.argv) < 2:
        sys.exit("python zarrtest.py schema.yaml")
    
    schema = sys.argv[1]
    
    node = zarr_fuse.open_store(schema, MODE="r")

    ds = node.dataset 
    print(ds)
    print("\nDimenze:", dict(ds.sizes))
    print("Souřadnice:", list(ds.coords))
    print("Proměnné:", list(ds.data_vars))

if __name__ == "__main__":
    main()