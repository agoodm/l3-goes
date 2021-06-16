import argparse
import s3fs
import pandas as pd
import xarray as xr
import numpy as np
import os

#variables = ['DSI', 'LVTP', 'TPW' 'LVMP']
def index_to_dt(i, sat, variable, domain):
    base_path = f'noaa-goes{sat}/ABI-L2-{variable}{domain}'
    years = [p.split('/')[-1] for p in fs.ls(base_path)]
    start_year = min(years)
    days = [p.split('/')[-1] for p in fs.ls(f'{base_path}/{start_year}')]
    start_day = min(days)
    start_dt = pd.to_datetime(f'{start_year}-{start_day}', format='%Y-%j')
    dt = start_dt + pd.Timedelta(i, unit='D')
    return str(dt.year), str(dt.dayofyear).zfill(3)

def process_l2_goes(sat, variable, domain, year, day, hour):
    pvar = variable if not variable.endswith('P') else variable[:-1]
    s3path = f'noaa-goes{sat}/ABI-L2-{variable}{domain}/{year}/{day}/{hour}/*'
    print(f'Checking: {s3path}')
    idx = xr.open_dataset(f'/mnt/efs/data/util/idx_{sat}_{domain}.nc')
    paths = fs.glob(s3path)
    if not paths:
        print('Files not found for this date.')
        return
    print('Files found:')
    for p in paths:
        print(p)
    datasets = []
    for fname in paths:
        try:
            f = fs.open(fname)
            fds = xr.open_dataset(f, engine='h5netcdf', decode_times=False)
            if 'pressure' in fds:
                fds = fds.isel(pressure=slice(0, 38))
            datasets.append(fds)
        except Exception as e:
            print(f'Could not open file: {fname}')
            print(e)
    if not datasets:
        print('Could not open any files. Exiting...')
        return
    ds = xr.concat(datasets, dim='t')
    dropped = set(ds.variables.keys()).difference([pvar, 'x', 'y', 't', 'pressure'])
    mask = ds.DQF_Overall == 0
    dsm = ds.drop_vars(dropped).where(mask).mean('t').stack(grid=['y', 'x']).reset_index('grid')
    dsg = dsm.isel(grid=idx.pixel_index).mean('pix')
    dsg['good_pixel_count'] = (mask.stack(grid=['y', 'x'])
                                   .reset_index('grid')
                                   .isel(grid=idx.pixel_index)
                                   .sum('pix')
                                   .sum('t'))
    prefix = os.path.basename(paths[0]).split('_s')[0].replace('L2', 'L3')
    time = pd.to_datetime(f'{year}-{day}-{hour}-30', format='%Y-%j-%H-%M')
    ts = time.strftime('%Y%m%d')
    fname = f'/mnt/efs/data/goes/{sat}/{domain}/{variable}/{prefix}_{ts}T{hour}30Z.nc'
    if os.path.exists(fname):
        os.remove(fname)
    ds_out = xr.merge([dsg, idx])
    ds_out.lat.attrs['units'] = 'degrees_north'
    ds_out.lon.attrs['units'] = 'degrees_east'
    ds_out.good_pixel_count.attrs['description'] = 'Total Number of geostationary projection pixels per hour where DQF_Overall is 0'
    ds_out.good_pixel_count.attrs['long_name'] = 'Good Pixel Count'
    ds_out.pixel_count.attrs['description'] = 'Number of geostationary projection pixels binned to MERRA2 grid cell'
    ds_out.pixel_count.attrs['long_name'] = 'Pixel Count'
    ds_out = ds_out.assign_coords(time=pd.Timestamp(f'{ts}T{hour}30'))
    dropped = set(ds_out.variables.keys()).difference([pvar, 'lat', 'lon', 'time', 'pressure', 'good_pixel_count', 'pixel_count'])
    ds_out = ds_out.drop_vars(dropped).assign_coords(time=time).expand_dims('time')
    if 'pressure' in ds_out:
        ds_out.pressure.attrs['units'] = 'hPa'
    print('Processing Finished. Saving output...')
    ds_out.to_netcdf(fname)
    print(f'Saved: {fname}')
    for d in datasets:
        d.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate hourly GOES L3 data')
    parser.add_argument('-y', '--year', default=2020)
    parser.add_argument('-H', '--hour', default=None)
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
    if not args.hour:
        hours = [p.split('/')[-1] for p in fs.ls(f'noaa-goes{sat}/ABI-L2-{variable}{domain}/{year}/{day}')]
    else:
        hours = [str(h).zfill(2) for h in args.hour.split(',')]
    i = os.environ.get('AWS_BATCH_JOB_ARRAY_INDEX')
    if i is not None:
        year, day = index_to_dt(int(i), sat, variable, domain)
    for hour in hours:
        process_l2_goes(sat, variable, domain, year, day, hour)

