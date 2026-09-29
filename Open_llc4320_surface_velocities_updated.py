# %% [markdown]
# # Opening llc4320 data and grid from the Open Storage Network
#
# C Spencer Jones (Texas A&M University), Chris Hill (Massachusetts Institute of Technology) and Thomas Haine (Johns Hopkins University)
#
# Adapted from original repository: https://github.com/cspencerjones/OSN_LLC4320/tree/v1.0.0

# %%
from functools import partial
import dask
import fsspec
import numpy as np
import s3fs
import ujson
import xarray as xr


def _multi_file_closer(closers):
    for closer in closers:
        closer()


def open_reference_dataset(ref_dict, endpoint_url, chunks=None):
    """Helper function using explicit asynchronous=True flags for OSN Kerchunk references."""
    rfs = fsspec.filesystem(
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
    mapper = rfs.get_mapper("")
    return xr.open_dataset(
        mapper,
        engine="zarr",
        backend_kwargs={"consolidated": False},
        chunks=chunks or {},
    )


open_delayed = dask.delayed(open_reference_dataset)
getattr_ = dask.delayed(getattr)

# %% [markdown]
# ### Select Data Subsets

# %%
print("==================================================")
print("1. Configuring parameters...")
print("==================================================")

face_range = range(10, 11)  # Range 0 to 12
start_from = 1180  # ~1180 is the first record for wind
length_in_hours = 12
time_step_in_hours = 3

start_iter = 10368 + start_from * 144
end_iter = 10368 + start_from * 144 + length_in_hours * 144
iter_range = np.arange(start_iter, end_iter, time_step_in_hours * 144)

get_Eta_files = True

print(f"Selected face range : {list(face_range)}")
print(f"Time iterations     : {len(iter_range)} files selected")

# %% [markdown]
# ### 2. Open Surface Data

# %%
print("\n==================================================")
print("2. Fetching Surface Metadata from OSN...")
print("==================================================")

endpoint_url = "https://mghp.osn.xsede.org"
fs = s3fs.S3FileSystem(anon=True, client_kwargs={"endpoint_url": endpoint_url})

if get_Eta_files:
    filelist = [
        f"cnh-bucket-1/llc_surf/kerchunk_files/llc4320_Eta-U-V-W-Theta-Salt_f{var1}_k0_iter_{var}.json"
        for var1 in face_range
        for var in iter_range
    ]
else:
    filelist = [
        f"cnh-bucket-1/llc_wind/kerchunk_files/llc4320_KPPhbl-PhiBot-oceTAUX-oceTAUY-SIarea_f{var1}_k0_iter_{var}.json"
        for var1 in face_range
        for var in iter_range
    ]

print(f"Downloading {len(filelist)} surface JSON reference file(s)...")
mapper_list = [fs.open(file, mode="rb") for file in filelist]
reflist = [ujson.load(m) for m in mapper_list]
print("Metadata downloaded successfully.")

print("Building virtual dataset pointers via Dask...")
datasets = [
    open_delayed(p, endpoint_url, chunks={"i": 720, "j": 720}) for p in reflist
]
closers = [getattr_(ds, "_close") for ds in datasets]
datasets, closers = dask.compute(datasets, closers)

print("Combining surface time/space slices...")
ds = xr.combine_by_coords(
    list(datasets),
    compat="override",
    coords="minimal",
    combine_attrs="override",
)

for ds1 in datasets:
    ds1.close()

ds.set_close(partial(_multi_file_closer, closers))

print("\n--- Surface Dataset Summary ---")
print(ds)

# %% [markdown]
# ### 3. Open LLC4320 Grid Data

# %%
print("\n==================================================")
print("3. Fetching Grid Metadata from OSN...")
print("==================================================")

filelist_grid = [
    f"cnh-bucket-1/llc_surf/kerchunk_files/llc4320_grid_f{var1}.json"
    for var1 in range(0, 13)
]

print(f"Downloading {len(filelist_grid)} grid JSON reference file(s)...")
mapper_grid = [fs.open(file, mode="rb") for file in filelist_grid]
reflist_grid = [ujson.load(m) for m in mapper_grid]
print("Grid metadata downloaded successfully.")

print("Building grid dataset pointers via Dask...")
datasets_grid = [
    open_delayed(p, endpoint_url, chunks={}) for p in reflist_grid
]
closers_grid = [getattr_(ds, "_close") for ds in datasets_grid]
datasets_grid, closers_grid = dask.compute(datasets_grid, closers_grid)

print("Combining grid faces...")
co = xr.combine_by_coords(
    list(datasets_grid),
    compat="override",
    coords="minimal",
    combine_attrs="override",
)

for ds1 in datasets_grid:
    ds1.close()

co.set_close(partial(_multi_file_closer, closers_grid))

print("\n--- Grid Dataset Summary ---")
print(co)

print("\n==================================================")
print("Done! Both datasets are lazily loaded and ready in memory.")
print("==================================================")

# Saves Eta variable for the loaded faces/times locally
print("Saving surface Eta field locally...")
ds["Eta"].to_netcdf("llc4320_Eta_subset.nc")
print("Saved to llc4320_Eta_subset.nc!")
