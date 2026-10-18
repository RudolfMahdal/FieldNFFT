"""Zkouška načtení původních Bukov dat (teploty) z CL2 přes load_fuse.

Úložiště a endpoint se nastavují přímo tady, takže nezáleží na tom, co je v .env
nebo ve Windows pro ZF_STORE_URL / ZF_S3_ENDPOINT_URL (ty mají jinak přednost před schématem
a právě kvůli nim vycházelo Forbidden). Klíče ZF_S3_ACCESS_KEY a ZF_S3_SECRET_KEY musí být
v .env klíče k CL2.
"""
import os

from zarrfuse_load import load_fuse

SCHEMA = "bukov_temperatures_schema.yaml"
GROUP = "bukov"
STORE_URL = "s3://random-tensor-fields/temperature_monitoring_steady.zarr"   # kopie dat; originál: temperature_monitoring.zarr
ENDPOINT = "https://s3.cl4.du.cesnet.cz"

# přepíše hodnoty z .env / prostředí
os.environ["ZF_STORE_URL"] = STORE_URL
os.environ["ZF_S3_ENDPOINT_URL"] = ENDPOINT

ds = load_fuse(SCHEMA, GROUP)          # jen čtení

print(ds)
print("\nDimenze:", dict(ds.sizes))
print("Proměnné:", list(ds.data_vars))
print("Vrty:", ds.borehole.values[:5], "...")

# malý vzorek z každé proměnné (skutečné čtení dat)
for name in ds.data_vars:
    da = ds[name]
    print(f"\n{name}: shape={da.shape}, dtype={da.dtype}")
    print(da.isel({d: slice(0, 2) for d in da.dims}).values)

# teploty horniny v prvním vrtu, jen časy s daty
temp = ds["rock_temp"].sel(borehole=ds.borehole.values[0]).dropna("date_time", how="all")
print("\nVrt:", temp.borehole.item(), "| kroků s daty:", temp.sizes["date_time"])
print(temp.isel(date_time=slice(0, 3)).to_pandas())