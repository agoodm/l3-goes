import s3fs
import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Proj


def isin_merra_cell(lat, lon, latm, lonm):
    dlat, dlon = 0.5, 0.625
    lat1, lat2 = latm - dlat/2, latm + dlat/2
    lon1, lon2 = lonm - dlon/2, lonm + dlon/2
    lon_slices = [(lon1, lon2)]
    if lon2 > 180:
       lon_slices.append((lon1, 180))
       lon_slices.append((-180, lon2 - 360))
    elif lon1 <= -180:
       lon_slices.append((-180, lon2))
       lon_slices.append((lon1 + 360, 180))
    for slc in lon_slices:
       lon1, lon2 = slc
       isin_cell = (lat1 <= lat <= lat2) & (lon1 <= lon <= lon2)
       if isin_cell:
           return True
    return False


def merra2_idx2(lat, lon, latmg, lonmg):
    dlat, dlon = 0.25, 0.3125
    lat1, lat2 = lat - dlat, lat + dlat
    lon1, lon2 = lon - dlon, lon + dlon
    lonmask = (lonmg >= lon1) & (lonmg <= lon2)
    if lon2 > 180:
       lonmask |= (lonmg <= lon2 + dlon - 360)
    mask = lonmask & (latmg >= lat1) & (latmg <= lat2)
    iidx = np.arange(latmg.size).reshape(latmg.shape)
    for i in iidx[mask]:
       if isin_merra_cell(lat, lon, latmg.flat[i], lonmg.flat[i]):
           return i
    return np.nan
    

fs = s3fs.S3FileSystem(anon=True)
sats = [16, 17]
domains = ['C', 'F']

for sat in sats:
    for domain in domains:
        f = fs.open(fs.ls(f'noaa-goes{sat}/ABI-L2-DSI{domain}/2020/001/01')[0])
        ds = xr.open_dataset(f)
        h = ds.goes_imager_projection.perspective_point_height[0]
        lon_0 = ds.goes_imager_projection.longitude_of_projection_origin[0]
        sweep = ds.goes_imager_projection.sweep_angle_axis
        p = Proj(proj='geos', h=h, lon_0=lon_0, sweep=sweep)
        x, y = np.meshgrid(h*ds.x, h*ds.y)
        lon, lat = p(x, y, inverse=True)
        lon[lon == 1e30] = np.nan
        lat[lat == 1e30] = np.nan
        ds = ds.assign_coords(lat=(('y', 'x'), lat), lon=(('y', 'x'), lon))
        npts = ds.x.size * ds.y.size
        latm = np.arange(-90, 90.5, 0.5)
        lonm = np.arange(-180, 180, 0.625)
        lonmg, latmg = np.meshgrid(lonm, latm)
        m2i = [merra2_idx2(ds.lat.values.flat[i], ds.lon.values.flat[i], latmg, lonmg) for i in range(npts)]
        groups = {}
        for i, v in enumerate(m2i):
            if np.isnan(v):
                continue
            v = int(v)
            if v not in groups:
                groups[v] = []
            groups[v].append(i)
        group_idx = np.asarray(list(groups.keys())).astype(int)
        pixel_count = np.zeros((latmg.size), dtype=int)
        pixel_count[group_idx] = np.asarray([len(g) for g in groups.values()])
        merra_grid = np.zeros((pixel_count.max(), latmg.size))
        for i, g in groups.items():
            merra_grid[:len(g), int(i)] = g
        space = pd.MultiIndex.from_product([latm, lonm], names=['lat', 'lon'])
        idx = xr.Dataset(coords=dict(space=space))
        idx['pixel_index'] = ('pix', 'space'), merra_grid.astype(int)
        idx['pixel_count'] = ('space'), pixel_count
        idx.unstack('space').to_netcdf(f'idx_{sat}_{domain}.nc')
        print(f'Saved: idx_{sat}_{domain}.nc')

