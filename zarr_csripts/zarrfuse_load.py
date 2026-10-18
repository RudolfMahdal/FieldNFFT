"""Načtení dat přes zarr_fuse (nastavení ze schématu + proměnné ZF_*).

Nastavení (STORE_URL, S3_ENDPOINT_URL) bere zarr_fuse ze schématu (ATTRS),
proměnné ZF_* ve .env / prostředí mají přednost:
    ZF_STORE_URL, ZF_S3_ENDPOINT_URL, ZF_S3_ACCESS_KEY, ZF_S3_SECRET_KEY

    from zarrfuse_load import load_fuse

    ds = load_fuse("schema.yaml")                    # jen čtení (nic nezapisuje)
    ds = load_fuse("schema.yaml", "bukov")           # konkrétní grupa
    node = load_fuse("schema.yaml", writable=True)   # zarr_fuse.open_node -> node.dataset
"""
from __future__ import annotations

from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import xarray as xr
import zarr

import zarr_fuse


def _walk(group: zarr.Group, prefix: str = "") -> list[str]:
    """Cesty všech grup, které obsahují pole ('' = kořen)."""
    found = []
    if any(True for _ in group.array_keys()):
        found.append(prefix)
    for name, sub in group.groups():
        found += _walk(sub, f"{prefix}/{name}".lstrip("/"))
    return found


def load_fuse(schema: str | Path | None = None, group: str | None = None, *,
              writable: bool = False, **open_kwargs):
    """
    writable=False: xarray.Dataset, úložiště se otevře jen pro čtení.
    writable=True:  zarr_fuse node (node.dataset je xarray.Dataset); může zapisovat metadata.
    """
    if writable:
        if schema is None:
            raise ValueError("Pro writable=True je schéma povinné.")
        return zarr_fuse.open_node(Path(schema))

    from zarr_fuse import zarr_storage         
    
    node_schema = zarr_fuse.schema.deserialize(Path(schema)) if schema else None
    options = zarr_storage._zarr_fuse_options(node_schema)   # schéma ATTRS + ZF_*
    if not options.get("STORE_URL"):
        raise ValueError("Chybí STORE_URL (schéma nebo ZF_STORE_URL).")
    options["MODE"] = "r"
    store = zarr_storage._zarr_store_open(options)

    if group is None:
        groups = _walk(zarr.open_group(store, mode="r"))
        if len(groups) != 1:
            raise ValueError(f"Zadej group, nalezeno: {groups or 'žádné'}")
        group = groups[0]
    return xr.open_zarr(store, group=group or None, consolidated=False, **open_kwargs)


if __name__ == "__main__":
    import sys
    schema = sys.argv[1] if len(sys.argv) > 1 else None
    group = sys.argv[2] if len(sys.argv) > 2 else None
    print(load_fuse(schema, group))