import fsspec
import s3fs
import ujson
import xarray as xr

endpoint_url = "https://mghp.osn.xsede.org"
json_path = "cnh-bucket-1/llc_surf/kerchunk_files/llc4320_Eta-U-V-W-Theta-Salt_f10_k0_iter_180288.json"
nc_path = "cnh-bucket-1/llc_surf/netcdf_files/llc4320_Eta-U-V-W-Theta-Salt_f10_k0_iter_180288.nc"

print("--- 1. Testing Direct S3 Reachability on OSN ---")
fs_s3 = s3fs.S3FileSystem(anon=True, client_kwargs={"endpoint_url": endpoint_url})

try:
    nc_exists = fs_s3.exists(nc_path)
    print(f"Target NetCDF file found on OSN: {nc_exists}")
    if nc_exists:
        print(f"File size: {fs_s3.info(nc_path)['size'] / (1024**2):.2f} MB")
except Exception as e:
    print(f"S3 Direct Connection Failed: {e}")

print("\n--- 2. Loading Kerchunk JSON Metadata ---")
try:
    with fs_s3.open(json_path, "rb") as f:
        ref_dict = ujson.load(f)
    print("Successfully read JSON metadata.")
except Exception as e:
    print(f"Failed to read JSON: {e}")

print("\n--- 3. Testing ReferenceFileSystem with Remote OSN Options ---")
try:
    remote_options = {
        "client_kwargs": {"endpoint_url": endpoint_url},
        "anon": True,
    }

    # remote_protocol and remote_options tell Kerchunk how to read the target .nc files
    rfs = fsspec.filesystem(
        "reference",
        fo=ref_dict,
        remote_protocol="s3",
        remote_options=remote_options,
    )
    print("ReferenceFileSystem initialized successfully.")
except Exception as e:
    print(f"ReferenceFileSystem Initialization Failed: {e}")

import fsspec
import xarray as xr

# Re-use endpoint and ref_dict from earlier steps
endpoint_url = "https://mghp.osn.xsede.org"

# =====================================================================
# APPROACH 4A: Pass pre-created `fs_s3` directly (RECOMMENDED)
# Pass the existing working `fs_s3` instance from Step 1 so Kerchunk
# reuses it instead of creating a new internal S3 filesystem.
# =====================================================================
print("\n--- Approach 4A: Reusing existing `fs_s3` instance ---")
try:
    rfs_a = fsspec.filesystem("reference", fo=ref_dict, fs=fs_s3)
    mapper_a = rfs_a.get_mapper("")
    ds_a = xr.open_dataset(
        mapper_a,
        engine="zarr",
        backend_kwargs={"consolidated": False},
        chunks={"i": 720, "j": 720},
    )
    print("SUCCESS (Approach 4A):")
    print(ds_a)
except Exception as e:
    print(f"Failed (Approach 4A): {e}")

# =====================================================================
# APPROACH 4B: Explicit `asynchronous=False` on BOTH remote & reference
# =====================================================================
print(
    "\n--- Approach 4B: Explicit `asynchronous=False` on both levels ---"
)
try:
    rfs_b = fsspec.filesystem(
        "reference",
        fo=ref_dict,
        remote_protocol="s3",
        remote_options={
            "client_kwargs": {"endpoint_url": endpoint_url},
            "anon": True,
            "asynchronous": False,
        },
        asynchronous=False,
    )
    mapper_b = rfs_b.get_mapper("")
    ds_b = xr.open_dataset(
        mapper_b,
        engine="zarr",
        backend_kwargs={"consolidated": False},
        chunks={"i": 720, "j": 720},
    )
    print("SUCCESS (Approach 4B):")
    print(ds_b)
except Exception as e:
    print(f"Failed (Approach 4B): {e}")

# =====================================================================
# APPROACH 4C: Explicit `asynchronous=True` on BOTH remote & reference
# =====================================================================
print("\n--- Approach 4C: Explicit `asynchronous=True` on both levels ---")
try:
    rfs_c = fsspec.filesystem(
        "reference",
        fo=ref_dict,
        remote_protocol="s3",
        remote_options={
            "client_kwargs": {"endpoint_url": endpoint_url},
            "anon": True,
            "asynchronous": True,
        },
        asynchronous=True,
    )
    mapper_c = rfs_c.get_mapper("")
    ds_c = xr.open_dataset(
        mapper_c,
        engine="zarr",
        backend_kwargs={"consolidated": False},
        chunks={"i": 720, "j": 720},
    )
    print("SUCCESS (Approach 4C):")
    print(ds_c)
except Exception as e:
    print(f"Failed (Approach 4C): {e}")

# =====================================================================
# APPROACH 4D: Using fsspec.get_mapper with target/remote options
# =====================================================================
print("\n--- Approach 4D: fsspec.get_mapper URL string wrapper ---")
try:
    mapper_d = fsspec.get_mapper(
        "reference://",
        fo=ref_dict,
        remote_protocol="s3",
        remote_options={
            "client_kwargs": {"endpoint_url": endpoint_url},
            "anon": True,
            "asynchronous": False,
        },
        asynchronous=False,
    )
    ds_d = xr.open_dataset(
        mapper_d,
        engine="zarr",
        backend_kwargs={"consolidated": False},
        chunks={"i": 720, "j": 720},
    )
    print("SUCCESS (Approach 4D):")
    print(ds_d)
except Exception as e:
    print(f"Failed (Approach 4D): {e}")