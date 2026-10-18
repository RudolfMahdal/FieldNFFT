"""Zkouška načtení dat přes load_fuse. Nahoře si nastav schéma a grupu."""
from zarrfuse_load import load_fuse

SCHEMA = "tensor_schema.yaml"     # cesta ke schématu
GROUP = None                      # název grupy, nebo None (vezme se jediná s daty)

ds = load_fuse(SCHEMA, GROUP)

print(ds)
print("\nDimenze:", dict(ds.sizes))
print("Proměnné:", list(ds.data_vars))

# malý vzorek z každé proměnné (skutečné čtení dat)
for name in ds.data_vars:
    da = ds[name]
    print(f"\n{name}: shape={da.shape}, dtype={da.dtype}")
    print(da.isel({d: slice(0, 2) for d in da.dims}).values)