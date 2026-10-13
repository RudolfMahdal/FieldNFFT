"""Test čtení polí z úložiště zarr (S3 / lokální), jen pro čtení.

Nastavení se čte z proměnných ZF_* nebo ze souboru .env (ten nedávej do gitu):
    ZF_STORE_URL, ZF_S3_ENDPOINT_URL, ZF_S3_ACCESS_KEY, ZF_S3_SECRET_KEY

    python zarrtest.py                      # nastavení jen z ZF_* / .env, všechny grupy
    python zarrtest.py schema.yaml          # nastavení (STORE_URL, endpoint) ze schématu
    python zarrtest.py schema.yaml bukov    # + konkrétní grupa (uzel schématu)

Schéma se používá jen jako zdroj nastavení. Úložiště se vždy otevírá jen pro čtení
(zarr_fuse.open_store by při otevření zapisoval metadata, proto ho nepoužíváme).
Proměnné ZF_* mají přednost před hodnotami ve schématu.
"""
import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()  # načte .env z aktuální složky, pokud existuje
except ImportError:
    pass  # bez python-dotenv stačí proměnné nastavit v terminálu

import xarray as xr
import zarr

import zarr_fuse
from zarr_fuse import zarr_storage


def check_keys(options: dict) -> None:
    url = options.get("STORE_URL")
    if not url:
        sys.exit("Chybí ZF_STORE_URL nebo STORE_URL ve schématu (viz .env.example)")
    missing = [k for k in ("S3_ACCESS_KEY", "S3_SECRET_KEY") if k not in options]
    if url.startswith("s3://") and missing:
        sys.exit(f"Chybí proměnné: {', '.join('ZF_' + k for k in missing)} (viz .env.example)")


def list_groups(root: zarr.Group, prefix: str = "") -> list[str]:
    """Cesty všech grup, které obsahují pole (datasety)."""
    found = []
    if any(True for _ in root.array_keys()):
        found.append(prefix)
    for name, member in root.groups():
        found += list_groups(member, f"{prefix}/{name}".lstrip("/"))
    return found


def open_datasets_direct(schema: str | None, group: str | None) -> dict[str, xr.Dataset]:
    # Stejné nastavení úložiště jako zarr_fuse (schéma ATTRS, pak ZF_* proměnné)
    node_schema = zarr_fuse.schema.deserialize(Path(schema)) if schema else None
    options = zarr_storage._zarr_fuse_options(node_schema)
    check_keys(options)
    print("Úložiště:", options.get("STORE_URL"), "| endpoint:", options.get("S3_ENDPOINT_URL"))
    options["MODE"] = "r"
    store = zarr_storage._zarr_store_open(options)
    groups = [group] if group else list_groups(zarr.open_group(store, mode="r"))
    print("Nalezené grupy s daty:", groups or "žádné")
    return {g: xr.open_zarr(store, group=g or None, consolidated=False)
            for g in groups}


def describe(name: str, ds: xr.Dataset) -> None:
    print(f"\n=== grupa: '{name}' ===")
    print(ds)
    print("Dimenze:", dict(ds.sizes))
    print("Souřadnice:", list(ds.coords))
    print("Proměnné:", list(ds.data_vars))
    # Malý vzorek, ať se ověří skutečné čtení dat a ne jen metadat
    for var in ds.data_vars:
        sample = ds[var].isel({d: slice(0, 2) for d in ds[var].dims})
        print(f"\n{var}: dtype={ds[var].dtype}, shape={ds[var].shape}")
        print(sample.values)


def main() -> None:
    schema = sys.argv[1] if len(sys.argv) > 1 else None
    group = sys.argv[2] if len(sys.argv) > 2 else None
    for name, ds in open_datasets_direct(schema, group).items():
        describe(name, ds)


if __name__ == "__main__":
    main()