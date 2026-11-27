import argparse
import io
import json
import os
from concurrent.futures import ThreadPoolExecutor

import laspy
import numpy as np
import pandas as pd
import requests
from PIL import Image
from scipy import ndimage

cd = os.path.dirname(os.path.abspath(__file__))
MERC = 20037508.342789244

# ASPRS LAS 1.4 classes excluded from the surface: low noise, wire guard, wire conductor, high noise [ASPRS, 2019]
EXCLUDED_CLASSES = [7, 13, 14, 18]

parser = argparse.ArgumentParser()
parser.add_argument('--ept', default='https://s3-us-west-2.amazonaws.com/usgs-lidar-public/CO_DRCOG_1_2020/')
parser.add_argument('--sites', default=os.path.join(cd, 'data', 'FC_DLR_sites.xlsx'))
parser.add_argument('--lat', type=float, help='center latitude (default: centroid of sites)')
parser.add_argument('--lon', type=float, help='center longitude (default: centroid of sites)')
parser.add_argument('--half-width', type=float, default=2000, help='[m] half side of the square area')
parser.add_argument('--zoom', type=int, default=16, help='Web Mercator zoom level of the output pixel grid')
parser.add_argument('--out', default=os.path.join(cd, 'data', 'dsm'))
parser.add_argument('--workers', type=int, default=16)
args = parser.parse_args()

#%% Area and output grid aligned with Web Mercator pixels at the given zoom
if args.lat is None or args.lon is None:
    sites = pd.read_excel(args.sites)
    args.lat, args.lon = sites['Latitude'].mean(), sites['Longitude'].mean()

res = 2 * MERC / (256 * 2 ** args.zoom)
xc = np.radians(args.lon) * 6378137
yc = np.log(np.tan(np.pi / 4 + np.radians(args.lat) / 2)) * 6378137
hw = args.half_width / np.cos(np.radians(args.lat))

px0 = int(np.floor((xc - hw + MERC) / res))
py0 = int(np.floor((MERC - (yc + hw)) / res))
nx = ny = int(np.ceil(2 * hw / res))
xmin, ymax = px0 * res - MERC, MERC - py0 * res
xmax, ymin = xmin + nx * res, ymax - ny * res

#%% EPT octree traversal down to the depth where point spacing is below half a pixel
session = requests.Session()
ept = session.get(args.ept + 'ept.json', timeout=60).json()
b = ept['bounds']
cube = b[3] - b[0]
depth_max = int(np.ceil(np.log2(cube / ept['span'] / (res / 2))))


def node_bounds(key):
    d, x, y, _ = map(int, key.split('-'))
    size = cube / 2 ** d
    return b[0] + x * size, b[1] + y * size, b[0] + (x + 1) * size, b[1] + (y + 1) * size


def overlaps(key):
    x0, y0, x1, y1 = node_bounds(key)
    return x0 < xmax and x1 > xmin and y0 < ymax and y1 > ymin


def collect(hier_key, nodes):
    hier = session.get(f"{args.ept}ept-hierarchy/{hier_key}.json", timeout=60).json()
    for key, count in hier.items():
        if int(key.split('-')[0]) > depth_max or not overlaps(key):
            continue
        if count == -1:
            collect(key, nodes)
        elif count > 0:
            nodes.append(key)


nodes = []
collect('0-0-0-0', nodes)
print(f'{len(nodes)} octree nodes up to depth {depth_max} ({cube / ept["span"] / 2 ** depth_max:.2f} m spacing) '
      f'for a {nx}x{ny} grid at {res * np.cos(np.radians(args.lat)):.2f} m')

#%% Highest non-noise return per pixel [Khosravipour et al., 2014]
dsm = np.full(ny * nx, -np.inf)


def fetch(key):
    r = session.get(f"{args.ept}ept-data/{key}.laz", timeout=120)
    r.raise_for_status()
    las = laspy.read(io.BytesIO(r.content))
    x, y, z = np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)
    keep = (x >= xmin) & (x < xmax) & (y > ymin) & (y <= ymax) & ~np.isin(np.asarray(las.classification), EXCLUDED_CLASSES)
    i = ((x[keep] - xmin) / res).astype(int)
    j = ((ymax - y[keep]) / res).astype(int)
    return j * nx + i, z[keep]


n_pts = 0
with ThreadPoolExecutor(args.workers) as pool:
    for n, (idx, z) in enumerate(pool.map(fetch, nodes)):
        np.maximum.at(dsm, idx, z)
        n_pts += len(z)
        if n % 100 == 0:
            print(f'{n}/{len(nodes)} nodes, {n_pts:,} points')

dsm = dsm.reshape(ny, nx)
dsm[~np.isfinite(dsm)] = np.nan

# isolated spikes (birds, residual noise): drop pixels > 5 m above the 3x3 median
med = ndimage.median_filter(np.nan_to_num(dsm, nan=-1e4), size=3)
dsm[dsm - med > 5] = np.nan
print(f'{n_pts:,} points, {np.isnan(dsm).mean() * 100:.1f}% empty pixels (bare-earth fallback in the planner)')

#%% Output: Terrarium-encoded RGBA PNG (alpha = 0 where empty) and metadata
v = np.nan_to_num(np.round(dsm, 1), nan=0) + 32768
rgba = np.zeros((ny, nx, 4), np.uint8)
rgba[..., 0] = (v // 256).astype(np.uint8)
rgba[..., 1] = (v % 256).astype(np.uint8)
rgba[..., 2] = ((v % 1) * 256).astype(np.uint8)
rgba[..., 3] = np.where(np.isnan(dsm), 0, 255)
Image.fromarray(rgba, 'RGBA').save(args.out + '.png', optimize=True)

meta = {'zoom': args.zoom, 'px0': px0, 'py0': py0, 'width': nx, 'height': ny,
        'source': args.ept, 'product': 'Highest non-noise lidar return per pixel (USGS 3DEP point cloud)',
        'excluded_classes': EXCLUDED_CLASSES, 'center': [args.lat, args.lon], 'half_width_m': args.half_width}
with open(args.out + '.json', 'w') as f:
    json.dump(meta, f, indent=1)
print(f'Saved {args.out}.png ({os.path.getsize(args.out + ".png") / 1e6:.1f} MB) and {args.out}.json')
