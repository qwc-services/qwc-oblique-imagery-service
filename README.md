[![](https://github.com/qwc-services/qwc-oblique-imagery-service/workflows/build/badge.svg)](https://github.com/qwc-services/qwc-oblique-imagery-service/actions)
[![docker](https://img.shields.io/docker/v/sourcepole/qwc-oblique-imagery-service?label=Docker%20image&sort=semver)](https://hub.docker.com/r/sourcepole/qwc-oblique-imagery-service)

QWC oblique imagery service
===========================

Serves oblique imagery, consumable as an XYZ layer source.

Place your oblique images below `<dataset_basedir>/<dataset_name>/*.tif`. For each dataset, add a resource configuration entry needs to be added, specifying the CRS, and patterns to distinguish images based on their filename for the four possible cardinal directions.

The dataset configuration, queried via

    http://localhost:5020/<dataset_name>/config
    
returns, for each available cardinal direction, the tile grid configuration, the tile URL as well as the centers of the available images. The tile grid configuration can be used to set up an OpenLayers XYZ source.

Tiles can be queried via the returned tile URL

    http://localhost:5020/<dataset_name>/tiles/<cardinal_direction>/{z}/{x}/{y}.png

The client can also ensure only tiles from the closest image are queried by calculating the closest image (based on the image centers returned by `/<dataset_name>/config`), and passing the array index of the closest image to the tiles query

    http://localhost:5020/Wolfsburg/tiles/<cardinal_direction>/{z}/{x}/{y}.png?img=<index>


Configuration
-------------

The static config files are stored as JSON files in `$CONFIG_PATH` with subdirectories for each tenant,
e.g. `$CONFIG_PATH/default/*.json`. The default tenant name is `default`.

### Elevation Service config

* [JSON schema](schemas/qwc-oblique-imagery-service.json)
* File location: `$CONFIG_PATH/<tenant>/obliqueImageryConfig.json`

Example:
```json
{
  "$schema": "https://raw.githubusercontent.com/qwc-services/qwc-oblique-imagery-service/master/schemas/qwc-oblique-imagery-service.json",
  "service": "elevation",
  "config": {
    "dataset_basedir": "/images",
    "tile_size": 256,
    "margin": 500
  },
  "resources": {
    "document_templates": [
      {
        "name":"<dataset_name>",
        "crs":"EPSG:3857",
        "pattern_north":"*_North_*.tif",
        "pattern_east":"*_East_*.tif",
        "pattern_south":"*_South_*.tif",
        "pattern_west":"*_West_*.tif"
      }
    ]
  }
}
```

### Environment variables

Config options in the config file can be overridden by equivalent uppercase environment variables.

| Variable          | Description                    |
|-------------------|--------------------------------|
| `DATASET_BASEDIR` | Basedir of datasets.           |
| `TILE_SIZE`       | Tile size.                     |
| `MARGIN`          | Dataset margin (in CRS units). |

Docker usage
------------

The docker image can be run with the following command:

    docker run -p 5000:9090 sourcepole/qwc-oblique-imagery-service


Development
-----------

Install dependencies and run service:

    uv run src/server.py
