from flask import Flask, send_file, url_for
from flask_restx import Api, Resource, reqparse
import datetime
import fnmatch
import math
import numpy as np
import os
import rasterio

from io import BytesIO
from PIL import Image
from qwc_services_core.api import CaseInsensitiveArgument
from qwc_services_core.app import app_nocache
from qwc_services_core.auth import auth_manager, optional_auth, get_identity
from qwc_services_core.tenant_handler import (
    TenantHandler, TenantPrefixMiddleware, TenantSessionInterface)
from qwc_services_core.runtime_config import RuntimeConfig
from qwc_services_core.permissions_reader import PermissionsReader
from rasterio.enums import Resampling
from rasterio.windows import from_bounds, Window


# Flask application
app = Flask(__name__)
app_nocache(app)
api = Api(app, version='1.0', title='ObliqueImagery service API',
          description="""API for QWC ObliqueImagery service.

Serves imagery for the QWC oblique aerial image view.
          """,
          default_label='Oblique Imagery operations', doc='/api/')
app.config.SWAGGER_UI_DOC_EXPANSION = 'list'

# disable verbose 404 error message
app.config['ERROR_404_HELP'] = False

auth = auth_manager(app, api)

tenant_handler = TenantHandler(app.logger)
app.wsgi_app = TenantPrefixMiddleware(app.wsgi_app)
app.session_interface = TenantSessionInterface()


image_cache = {}


def resolve_dataset(dataset):

    identity = get_identity()
    tenant = tenant_handler.tenant()
    config_handler = RuntimeConfig("obliqueImagery", app.logger)
    config = config_handler.tenant_config(tenant)

    resources = (config.resources() or {}).get('oblique_image_datasets', [])
    resource = next((x for x in resources if x["name"] == dataset), None)
    if not resource:
        app.logger.warning("Dataset does not exist: %s" % dataset)
        return None, None, None

    permissions_handler = PermissionsReader(tenant, app.logger)
    permitted_resources = permissions_handler.resource_permissions(
        'oblique_image_datasets', identity
    )
    if not dataset in permitted_resources:
        app.logger.warning("Dataset not permitted: %s" % dataset)
        return None, None, None

    dataset_basedir = config.get('dataset_basedir', '/images')
    dataset_dir = os.path.join(dataset_basedir, dataset)

    conf = {
        "tile_size": config.get("tile_size", 256),
        "margin": config.get("margin", 500)
    }

    return resource, dataset_dir, conf


def get_images(resource, dataset_dir):
    images = []

    if not os.path.isdir(dataset_dir):
        return images

    pattern_map = {
        'n': resource.get('pattern_north'),
        'e': resource.get('pattern_east'),
        's': resource.get('pattern_south'),
        'w': resource.get('pattern_west')
    }

    for filename in os.listdir(dataset_dir):

        path = os.path.join(dataset_dir, filename)

        timestamp = datetime.datetime.fromtimestamp(os.path.getmtime(path), datetime.UTC)
        if path not in image_cache or timestamp > image_cache[path]['mtime']:

            for direction, pattern in pattern_map.items():
                if pattern and fnmatch.fnmatch(filename, pattern):
                    file_direction = direction
                    break
            else:
                continue

            with rasterio.open(path) as ds:
                bounds = list(ds.bounds)
                res = (bounds[2] - bounds[0]) / ds.width

                dpi = 90.71428571428571
                ext_width = (bounds[2] - bounds[0])
                img_width = ds.width / dpi * 0.0254
                img_scale = ext_width / img_width
                img_res =  ext_width / (ds.width / dpi * 0.0254) * 0.0254 / dpi

                data = {
                    "file": filename,
                    "direction": file_direction,
                    "bounds": bounds,
                    "center": [0.5 * (bounds[0] + bounds[2]), 0.5 * (bounds[1] + bounds[3])],
                    "resolutions": sorted(list([res] + [res * o for o in ds.overviews(1)]), reverse=True)
                }
                image_cache[path] = {'data': data, 'mtime': timestamp}

        if path in image_cache:
            images.append(image_cache[path]['data'])

    return images


def compute_full_extent(images, margin):
    all_bounds = [img["bounds"] for img in images]
    global_extent = [
        min(b[0] for b in all_bounds),
        min(b[1] for b in all_bounds),
        max(b[2] for b in all_bounds),
        max(b[3] for b in all_bounds)
    ]
    global_extent[0] -= margin
    global_extent[1] -= margin
    global_extent[2] += margin
    global_extent[3] += margin
    return global_extent

def window_from_bounds(bounds, transform):
    minx, miny, maxx, maxy = bounds
    inv = ~transform

    # Four corners in pixel space
    px = []
    py = []

    for x, y in [(minx, miny), (minx, maxy), (maxx, miny), (maxx, maxy)]:
        px_i, py_i = inv * (x, y)
        px.append(px_i)
        py.append(py_i)

    row_start = int(min(py))
    row_stop  = int(max(py))
    col_start = int(min(px))
    col_stop  = int(max(px))

    # return ((row_start, row_stop), (col_start, col_stop))
    return Window(col_start, row_start, col_stop - col_start, row_stop - row_start)
    # return Window(row_start, col_start, row_stop - row_start, col_stop - col_start)

# routes
@api.route('/<dataset>/config')
class DatasetConfig(Resource):
    @api.doc('Get dataset config')
    @optional_auth
    def get(self, dataset):
        resource, dataset_dir, config = resolve_dataset(dataset)
        if not resource:
            return "Dataset does not exist or is not permitted", 404

        images = get_images(resource, dataset_dir)

        global_extent = compute_full_extent(images, config['margin'])

        return {
            "extent": global_extent,
            "origin": [global_extent[0], global_extent[3]],
            "tileSize": config['tile_size'],
            "resolutions": images[0]["resolutions"], # Assume same resolution
            "crs": resource['crs'],
            "url": url_for("tile", dataset=dataset, direction='dir', x=0, y=0, z=0, _external=True).replace("dir/0/0/0.png", "{direction}/{z}/{x}/{y}.png"),
            "image_centers": {
                "n": [img["center"] for img in images if img['direction'] == 'n'],
                "e": [img["center"] for img in images if img['direction'] == 'e'],
                "w": [img["center"] for img in images if img['direction'] == 'w'],
                "s": [img["center"] for img in images if img['direction'] == 's']
            }
        }
        return result


gettile_parser = reqparse.RequestParser(argument_class=CaseInsensitiveArgument)
gettile_parser.add_argument('img', help='Closest image index', required=False, type=int)

@api.route("/<dataset>/tiles/<direction>/<int:z>/<int:x>/<int:y>.png", endpoint="tile")
class GetTile(Resource):
    @api.doc('Get requested tile')
    @api.expect(gettile_parser)
    @optional_auth
    def get(self, dataset, direction, z, x, y):

        args = gettile_parser.parse_args()
        best_idx = args.get('img', None)

        resource, dataset_dir, config = resolve_dataset(dataset)
        if not resource:
            return "Dataset does not exist or is not permitted", 404

        images = get_images(resource, dataset_dir)
        dir_images = [img for img in images if img["direction"] == direction]
        if not dir_images:
            return "Dataset is empty", 404

        tile_size = config['tile_size']
        global_extent = compute_full_extent(images, config['margin'])
        res = images[0]['resolutions'][z]

        # Compute tile bounds
        minx = global_extent[0] + x * tile_size * res
        maxx = global_extent[0] + (x+1) * tile_size * res
        maxy = global_extent[3] - y * tile_size * res
        miny = global_extent[3] - (y+1) * tile_size * res

        if best_idx is not None:
            best = dir_images[best_idx]
        else:
            # Use tile center for nearest-image selection
            cx = 0.5 * (minx + maxx)
            cy = 0.5 * (miny + maxy)

            # Find image closest to view center
            best = dir_images[0]
            best_dist = (cx - best['center'][0])**2 + (cy - best['center'][1])**2

            for img in dir_images[1:]:
                dist = (cx - img['center'][0])**2 + (cy - img['center'][1])**2
                if dist < best_dist:
                    best_dist = dist
                    best = img

        data = np.zeros((tile_size, tile_size, 4), dtype=np.uint8)
        image_angle = 0
        try:
            with rasterio.open(os.path.join(dataset_dir, best['file'])) as ds:
                window = window_from_bounds([minx, miny, maxx, maxy], ds.transform)
                image_window = Window(0, 0, ds.width, ds.height)
                clamped_window = window.intersection(image_window)

                clamp_tile_x = int((clamped_window.col_off - window.col_off) * tile_size / window.width)
                clamp_tile_y = int((clamped_window.row_off - window.row_off) * tile_size / window.height)
                clamp_tile_width = math.ceil(clamped_window.width * tile_size / window.width)
                clamp_tile_height = math.ceil(clamped_window.height * tile_size / window.height)

                clamped_data = ds.read(
                    out_shape=(ds.count, clamp_tile_height, clamp_tile_width),
                    window=clamped_window,
                    resampling=Resampling.bilinear,
                    boundless=False
                )
                if clamped_data.shape[0] == 3:
                    alpha = np.where(np.any(clamped_data != 0, axis=0), 255, 0).astype(np.uint8)
                    clamped_data = np.concatenate([clamped_data, alpha[None, ...]], axis=0)

                # Reorder <channels><height><width> => <height><width><channels> for Pillow
                clamped_data = np.moveaxis(clamped_data, 0, -1)

                # Write clamped data into full width data
                data[
                    clamp_tile_y : clamp_tile_y + clamp_tile_height,
                    clamp_tile_x : clamp_tile_x + clamp_tile_width,
                    :
                ] = clamped_data

                A, B, C, D, E, F = ds.transform[:6]
                image_angle = math.degrees(math.atan2(-B, A)) % 360
                image_angle = min([0, 90, 180, 270, 360], key=lambda a: abs(a - image_angle)) % 360
        except Exception as e:
            app.logger.debug("Failed to read image for window [%d, %d, %d, %d]: %s" % (minx, miny, maxx, maxy, str(e)))

        # Convert to JPEG
        tile = Image.fromarray(data.astype(np.uint8), mode="RGBA")
        if image_angle != 0:
            tile = tile.rotate(
                -image_angle,
                resample=Image.BILINEAR
            )
        bio = BytesIO()
        tile.save(bio, "PNG")
        bio.seek(0)

        return send_file(bio, mimetype="image/png")

""" readyness probe endpoint """
@app.route("/ready", methods=['GET'])
def ready():
    return jsonify({"status": "OK"})


""" liveness probe endpoint """
@app.route("/healthz", methods=['GET'])
def healthz():
    return jsonify({"status": "OK"})


# local webserver
if __name__ == '__main__':
    app.run(host='localhost', port=5020, debug=True)
