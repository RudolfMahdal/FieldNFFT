"""Jednoduché načtení dat ze zarr úložiště (S3 i lokální), jen pro čtení.
Schéma není potřeba, struktura je uložená přímo v datech.

Klíče: ZF_S3_ACCESS_KEY a ZF_S3_SECRET_KEY v .env (nebo v prostředí).

    from bukov_load import load
    ds = load()                                  # Bukov teploty
    ds = load("bukov", "temperature_monitoring_steady.zarr")
"""
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import xarray as xr

ENDPOINT = "https://s3.cl2.du.cesnet.cz"


def load(group: str = "bukov", store: str = "temperature_monitoring.zarr",
         bucket: str = "bukov") -> xr.Dataset:
    if os.path.exists(store):                       # lokální úložiště
        return xr.open_zarr(store, group=group, consolidated=False)
    return xr.open_zarr(
        f"s3://{bucket}/{store}", group=group, consolidated=False,
        storage_options={"key": os.environ["ZF_S3_ACCESS_KEY"],
                         "secret": os.environ["ZF_S3_SECRET_KEY"],
                         "endpoint_url": os.environ.get("ZF_S3_ENDPOINT_URL", ENDPOINT)},
    )


if __name__ == "__main__":
    ds = load()
    print(ds)

    temp = ds["rock_temp"].sel(borehole=ds.borehole.values[0])
    temp = temp.dropna("date_time", how="all")
    print("\nVrt:", temp.borehole.item(), "| kroků s daty:", temp.sizes["date_time"])
    print(temp.isel(date_time=slice(0, 3)).to_pandas())