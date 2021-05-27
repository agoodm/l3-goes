import argparse
import s3fs
import pandas as pd
import xarray as xr
import numpy as np
import os

def index_to_dt(i):
    hour = str(i % 24).zfill(2)
    day = str((i // 24) + 1).zfill(3)
    return day, hour

parser = argparse.ArgumentParser(description='Generate hourly GOES L3 data')
parser.add_argument('-y', '--year', default=2020)
parser.add_argument('-H', '--hour', default='00')
parser.add_argument('-D', '--day', default='001')
parser.add_argument('-v', '--variable', default='DSI')
parser.add_argument('-d', '--domain', default='C')
parser.add_argument('-s', '--satellite', default=16)
args = parser.parse_args()

xr.set_options(keep_attrs=True)
fs = s3fs.S3FileSystem(anon=True)
variable = args.variable
sat = args.satellite
domain = args.domain
year = args.year
day = str(args.day).zfill(3)
hour = str(args.hour).zfill(2)
i = os.environ.get('AWS_BATCH_JOB_ARRAY_INDEX')
if i is not None:
    day, hour = index_to_dt(int(i))
#variables = ['DSI', 'LVTP', 'TPW'] # LVMP
#sats = [16, 17]
#domains = ['C', 'F']
#year = 2021
#day = '094'
#hour = '00'

s3path = f'noaa-goes{sat}/ABI-L2-{variable}{domain}/{year}/{day}/{hour}/*'
print(f'Checking: {s3path}')
idx = xr.open_dataset(f'/mnt/efs/data/util/idx_{sat}_{domain}.nc')
paths = fs.glob(s3path)
print('Files found:')
for p in paths:
    print(p)
files = [fs.open(fname) for fname in paths]
datasets = [xr.open_dataset(f) for f in files]
ds = xr.concat(datasets, dim='t')
dropped = set(ds.variables.keys()).difference([variable, 'x', 'y', 't', 'pressure'])
mask = ds.DQF_Overall == 0
dsm = ds.drop_vars(dropped).where(mask).mean('t').stack(grid=['y', 'x']).reset_index('grid')
dsg = dsm.isel(grid=idx.pixel_index).mean('pix')
dsg['good_pixel_count'] = (mask.stack(grid=['y', 'x'])
                               .reset_index('grid')
                               .isel(grid=idx.pixel_index)
                               .sum('pix')
                               .sum('t'))
prefix = os.path.basename(paths[0]).split('_s')[0].replace('L2', 'L3')
ts = pd.Timestamp(ds.t.values[0]).strftime('%Y%m%d')
fname = f'/mnt/efs/data/goes/{sat}/{domain}/{variable}/{prefix}_{ts}T{hour}30Z.nc'
ds_out = xr.merge([dsg, idx])
ds_out.lat.attrs['units'] = 'degrees_north'
ds_out.lon.attrs['units'] = 'degrees_east'
ds_out.good_pixel_count.attrs['description'] = 'Total Number of geostationary projection pixels per hour where DQF_Overall is 0'
ds_out.good_pixel_count.attrs['long_name'] = 'Good Pixel Count'
ds_out.pixel_count.attrs['description'] = 'Number of geostationary projection pixels binned to MERRA2 grid cell'
ds_out.pixel_count.attrs['long_name'] = 'Pixel Count'
ds_out = ds_out.assign_coords(time=pd.Timestamp(f'{ts}T{hour}30'))
dropped = set(ds_out.variables.keys()).difference([variable, 'lat', 'lon', 'time', 'pressure', 'good_pixel_count', 'pixel_count'])
ds_out = ds_out.drop_vars(dropped)
if 'pressure' in ds_out:
    ds_out.pressure.attrs['units'] = 'hPa'
print('Processing Finished. Saving output...')
ds_out.to_netcdf(fname)
print(f'Saved: {fname}')
for d in datasets:
    d.close()

