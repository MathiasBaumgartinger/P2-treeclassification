from typing import Tuple
from osgeo import ogr, gdal, gdalconst
import sys, os

path = sys.argv[1]
layername = sys.argv[2]
resolution = sys.argv[3]

shp_driver: ogr.Driver = ogr.GetDriverByName("ESRI Shapefile")
ds: ogr.DataSource = shp_driver.Open(path, gdalconst.GA_ReadOnly)
layer: ogr.Layer = ds.GetLayer(layername)

buffered_ds: ogr.DataSource = shp_driver.CreateDataSource("buffered.shp")
buffered_layer: ogr.Layer = buffered_ds.CreateLayer("trees", srs=layer.GetSpatialRef(), geom_type=ogr.wkbPolygon)
buffered_layer.CreateField(ogr.FieldDefn("type", ogr.OFTInteger))
feature_defn: ogr.FeatureDefn = buffered_layer.GetLayerDefn()

types = []
feature: ogr.Feature
for feature in layer:
    # {3.0: '7-9 m', 1.0: '0-3 m', 2.0: '4-6 m', 4.999999999999999: '13-15 m', 4.0: '10-12 m', 0.0: 'nicht bekannt', 6.0: '16-18 m', 8.0: '>21 m', 7.000000000000001: '19-21 m'}
    diameter_id: int = feature.GetField("KRONENDURC")
    # Many different types, simpliest way to "enumerate" them is to store them in array
    # (e.g. Fichte, Ahron, ...)
    ttype: str = feature.GetField("GATTUNG_AR")
    if not ttype in types:
        types.append(ttype)

    radius: float
    # Diameter is unknown, take a guessed average
    if diameter_id == 0:
        radius = 5.0
    # The diameter id is always somewhere between 2 and 3 times the actual diameter
    else:
        radius = diameter_id * 2.5 / 2
    
    center = feature.GetGeometryRef().GetPoint_2D()    
    center_proj: ogr.Geometry = ogr.CreateGeometryFromWkt(f"POINT ({center[0]} {center[1]})")
    buffer: ogr.Geometry = center_proj.Buffer(radius, 10)
    bufferPoly = ogr.ForceToPolygon(buffer.GetGeometryRef(0))

    buffered_feature: ogr.Feature = ogr.Feature(feature_defn)
    buffered_feature.SetGeometry(bufferPoly)
    buffered_feature.SetField("type", types.index(ttype))

    buffered_layer.CreateFeature(buffered_feature)

buffered_ds = ds = buffered_layer = buffered_feature = None  
        
rasterize_options = gdal.RasterizeOptions(xRes=resolution, yRes=resolution, noData=-9999, attribute="type")
gdal.Rasterize("buffered.tif", "buffered.shp", options=rasterize_options)

