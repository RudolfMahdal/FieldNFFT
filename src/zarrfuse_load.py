"""Read-only loading of a Zarr dataset via zarr_fuse.

Settings (STORE_URL, S3_ENDPOINT_URL) come from the schema ATTRS; ZF_* environment
variables (or a .env file) take precedence:
    ZF_STORE_URL, ZF_S3_ENDPOINT_URL, ZF_S3_ACCESS_KEY, ZF_S3_SECRET_KEY
"""
from pathlib import Path

import xarray as xr
import zarr_fuse
from dotenv import load_dotenv
from zarr_fuse import zarr_storage
import os
load_dotenv()



def setup_environment(zarr_url):
    """Configures S3 environment variables and returns the dataset name."""
    os.environ["ZF_STORE_URL"] = zarr_url
    os.environ["ZF_S3_ENDPOINT_URL"] = "https://s3.cl4.du.cesnet.cz"
    return zarr_url.split("/")[-1]

def load_fuse(schema, group):
    """Returns an xarray.Dataset; the store is opened read-only."""
    node_schema = zarr_fuse.schema.deserialize(Path(schema))
    options = zarr_storage._zarr_fuse_options(node_schema)  # schema ATTRS + ZF_* vars
    if not options.get("STORE_URL"):
        raise ValueError("Missing STORE_URL (set it in the schema or via ZF_STORE_URL).")
    options["MODE"] = "r"
    store = zarr_storage._zarr_store_open(options)
    return xr.open_zarr(store, group=group, consolidated=False)