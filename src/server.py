from flask import Flask, send_file, url_for
from flask_restx import Api, Resource, reqparse
import datetime
import fnmatch
import os
import numpy as np
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
from rasterio.windows import from_bounds


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


def get_images(resource, dataset_dir, direction):
    images = []

    if not os.path.isdir(dataset_dir):
        return images

    direction_map = {
        'n': 'pattern_north',
        'e': 'pattern_east',
        's': 'pattern_south',
        'w': 'pattern_west'
    }
    if not direction in direction_map:
        return []

    pattern = resource[direction_map[direction]]

    for filename in os.listdir(dataset_dir):
        if not fnmatch.fnmatch(filename, pattern):
            continue

        path = os.path.join(dataset_dir, filename)
        timestamp = datetime.datetime.fromtimestamp(os.path.getmtime(path), datetime.UTC)
        if path not in image_cache or timestamp > image_cache[path]['mtime']:
            with rasterio.open(path) as ds:
                bounds = list(ds.bounds)

                data = {
                    "file": filename,
                    "bounds": bounds,
                    "center": [0.5 * (bounds[0] + bounds[2]), 0.5 * (bounds[1] + bounds[3])],
                    "resolutions": sorted(list([ds.res[0]] + [ds.res[0] * o for o in ds.overviews(1)]), reverse=True)
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


# routes
@api.route('/<dataset>/config')
class DatasetConfig(Resource):
    @api.doc('Get dataset config')
    @optional_auth
    def get(self, dataset):
        resource, dataset_dir, config = resolve_dataset(dataset)
        if not resource:
            return "Dataset does not exist or is not permitted", 404

        result = {}
        for direction in ["n", "e", "s", "w"]:
            images = get_images(resource, dataset_dir, direction)
            if not images:
                continue

            global_extent = compute_full_extent(images, config['margin'])

            result[direction] = {
                "extent": global_extent,
                "origin": [global_extent[0], global_extent[3]],
                "tileSize": config['tile_size'],
                "resolutions": images[0]["resolutions"], # Assume same resolution
                "crs": resource['crs'],
                "url": url_for("tile", dataset=dataset, direction=direction, x=0, y=0, z=0, _external=True).replace("/0/0/0.png", "/{z}/{x}/{y}.png"),
                "image_centers": [img["center"] for img in images]
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

        images = get_images(resource, dataset_dir, direction)
        if not images:
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
            best = images[best_idx]
        else:
            # Use tile center for nearest-image selection
            cx = 0.5 * (minx + maxx)
            cy = 0.5 * (miny + maxy)

            # Find image closest to view center
            best = images[0]
            best_dist = (cx - best['center'][0])**2 + (cy - best['center'][1])**2

            for img in images[1:]:
                dist = (cx - img['center'][0])**2 + (cy - img['center'][1])**2
                if dist < best_dist:
                    best_dist = dist
                    best = img

        with rasterio.open(os.path.join(dataset_dir, best['file'])) as ds:
            window = from_bounds(minx, miny, maxx, maxy, transform=ds.transform)

            try:
                data = ds.read(
                    out_shape=(ds.count, tile_size, tile_size),
                    window=window,
                    resampling=Resampling.bilinear,
                    boundless=True,
                    fill_value=0   # white value for missing data
                )
                if data.shape[0] == 3:
                    alpha = np.where(np.any(data != 0, axis=0), 255, 0).astype(np.uint8)
                    data = np.concatenate([data, alpha[None, ...]], axis=0)

                data = np.moveaxis(data, 0, -1)
            except:
                # White tile
                data = np.zeros((tile_size, tile_size, 4), dtype=np.uint8)

        # Convert to JPEG
        tile = Image.fromarray(data.astype(np.uint8), mode="RGBA")
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
