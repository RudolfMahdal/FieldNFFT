"""Test for loading data with zarrfuse."""
from src.zarrfuse_load import load_fuse

SCHEMA = "tensor_schema.yaml"     
GROUP = None                      

ds = load_fuse(SCHEMA, GROUP)

print(ds)
print("\nDimenze:", dict(ds.sizes))
print("Proměnné:", list(ds.data_vars))

for name in ds.data_vars:
    da = ds[name]
    print(f"\n{name}: shape={da.shape}, dtype={da.dtype}")
    print(da.isel({d: slice(0, 2) for d in da.dims}).values)