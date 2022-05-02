import sys
from typing_extensions import IntVar
from osgeo import gdal, ogr, gdalconst
import numpy as np
import math
import matplotlib.pyplot as plt
import progressbar
import threading
import os
from multiprocessing import Pool

#
# Early, manual version of creating a radial geodataset from (as described in)
# from the tree cadaster. 
#


PIXEL_SIZE = 5000
MAX_THREADS = math.ceil(os.cpu_count() + os.cpu_count() * 0.1)
THREADED = True
PATH_TO_DATASET = "D:/boku/data/BAUMKATOGD.shp"

source_ds : ogr.DataSource = ogr.Open(PATH_TO_DATASET, gdalconst.GA_ReadOnly)
source_layer : ogr.Layer = source_ds.GetLayer()
x_min, x_max, y_min, y_max = source_layer.GetExtent()

# Possible crown-tree radius configurations (m -> metres)
# {2.0: '4-6 m', 4.0: '10-12 m', 3.0: '7-9 m', 1.0: '0-3 m', 5.0: '13-15 m', 0.0: 'nicht bekannt', 6.0: '16-18 m', 8.0: '>21 m', 7.0: '19-21 m'}

driver : gdal.Driver = gdal.GetDriverByName("GTiff")
driver.Register()
x_res = int((x_max - x_min) / PIXEL_SIZE)
y_res = int((y_max - y_min) / PIXEL_SIZE)
target_ds : gdal.Dataset = driver.Create("rasterized_tree_cadastres.tif", PIXEL_SIZE, PIXEL_SIZE, 1, gdal.GDT_Int16)
#target_ds.SetGeoTransform(
#    x_min, PIXEL_SIZE, 0,
#    y_max, 0, -PIXEL_SIZE
#)

# Setup a progress bar as the process takes a very long time
widgets = [' [',
         progressbar.Timer(format= 'elapsed time: %(elapsed)s'),
         '] ',
           progressbar.Bar('*'),' (',
           progressbar.ETA(), ') ',
          ]
bar = progressbar.ProgressBar(max_value=source_layer.GetFeatureCount(), 
                              widgets=widgets).start()

feature : ogr.Feature
array = np.zeros((PIXEL_SIZE + 1, PIXEL_SIZE + 1))

def findRadius(vector_pos, radius_proj, inside_circle, radial):
    retvals = np.empty((0, 2), int)
    raster_pos = (math.floor((vector_pos[0] - x_min) / (x_max - x_min) * PIXEL_SIZE), math.floor((vector_pos[1] - y_min) / (y_max - y_min) * PIXEL_SIZE))
    
    # Find the point position inside the raster
    circle_y = raster_pos[0]
    circle_x = PIXEL_SIZE - raster_pos[1]

    if radial:
        #i = 1
        #while i + (circle_x) < PIXEL_SIZE:
        #    j = 1
        #    while j + (circle_x) < PIXEL_SIZE:
        #        if inside_circle(circle_x, circle_y, radius_proj, circle_x + i, circle_y + j):
        #            retvals = np.append(retvals, [[min(PIXEL_SIZE, circle_x + i), min(PIXEL_SIZE, circle_y + j)]], axis=0)
#
        #        if inside_circle(circle_x, circle_y, radius_proj, circle_x + i, circle_y - j):
        #            retvals = np.append(retvals, [[min(PIXEL_SIZE, circle_x + i), max(0, circle_y - j)]], axis=0)
#
        #        if inside_circle(circle_x, circle_y, radius_proj, circle_x - i, circle_y + j):
        #            retvals = np.append(retvals, [[max(0, circle_x - i), min(PIXEL_SIZE, circle_y + j)]], axis=0)
#
        #        if inside_circle(circle_x, circle_y, radius_proj, circle_x - i, circle_y - j):
        #            retvals = np.append(retvals, [[max(0, circle_x - i), max(0, circle_y - j)]], axis=0)
        #        else:
        #            break
        #        j += 1
        #    i += 1

        # Function for filling a circle around the point
        def fill(retvals, x, y):
            # over circle mid y-coord
            if circle_y + y > circle_y:
                while circle_y + y >= circle_y:                 
                    retvals = np.append(retvals, [[min(PIXEL_SIZE, circle_x + x), min(PIXEL_SIZE, circle_y + y)]], axis=0)
                    y -= 1
            else: 
                while circle_y + y <= circle_y:                 
                    retvals = np.append(retvals, [[min(PIXEL_SIZE, circle_x + x), min(PIXEL_SIZE, circle_y + y)]], axis=0)
                    y += 1
            
            return retvals

        # Create circles following the description of https://www.geeksforgeeks.org/bresenhams-circle-drawing-algorithm/
        # and  https://de.wikipedia.org/wiki/Bresenham-Algorithmus#Kreisvariante_des_Algorithmus
        f = 1 - radius_proj
        ddF_x = 0
        ddF_y = -2 * radius_proj
        x = 0
        y = radius_proj

        retvals = fill(retvals, 0, radius_proj)
        retvals = fill(retvals, 0, -radius_proj)
        retvals = fill(retvals, radius_proj, 0)
        retvals = fill(retvals, -radius_proj, 0)

        while(x < y):
            if(f >= 0):
                y -= 1
                ddF_y += 2
                f += ddF_y
            x += 1
            ddF_x += 2
            f += ddF_x + 1

            retvals = fill(retvals, x, y)
            retvals = fill(retvals, -x, y)
            retvals = fill(retvals, x, -y)
            retvals = fill(retvals, -x, -y)
            retvals = fill(retvals, y, x)
            retvals = fill(retvals, -y, x)
            retvals = fill(retvals, y, -x)
            retvals = fill(retvals, -y, -x)

    retvals = np.append(retvals, [[circle_x, circle_y]], axis=0)
    #semaphore = threading.Semaphore()
    #semaphore.acquire()
    processed_features.append(1)
    bar.update(len(processed_features))
    #semaphore.release()

    return retvals

# Different functions (trying with different parallelization techniques - does not work well)

def applyRadial(vector_poss, radii, inside_circle, result, radial=False):
    #print("Thread %d starting" % threading.get_native_id())
    for i in range(len(radii)):
        #print("Thread %d at %.2f percent" % (threading.get_native_id(), i / len(radii)) * 100)
        test = findRadius(vector_poss[i], radii[i], inside_circle, radial)
        for entry in test:
            result.append(entry)
    #print("Thread %d ending" % threading.get_native_id())

    return result


def threadedRadial(feature_start, feature_count, inside_circle):
    radii = np.array([])
    vector_poss = np.array([])

    feature_end = feature_start + feature_count
    while feature_start < feature_end:
        feature = source_layer.GetFeature(feature_start)
        if feature: 
            radii = np.append(radii, (feature.GetField("KRONENDURC")) * 3 / 2)
            vector_poss = np.append(vector_poss, feature.GetGeometryRef().GetPoint_2D())

        feature_start = feature_start + 1

    result = []
    thread = threading.Thread(target=applyRadial, args=(vector_poss, radii, inside_circle, result))
    return (thread, result)


def asyncRadial(feature_start, feature_count, inside_circle, pool):
    radii = np.array([])
    vector_poss = np.array([])

    feature_end = feature_start + feature_count
    while feature_start < feature_end:
        feature = source_layer.GetFeature(feature_start)
        if feature: 
            radii = np.append(radii, (feature.GetField("KRONENDURC")) * 3 / 2)
            vector_poss = np.append(vector_poss, feature.GetGeometryRef().GetPoint_2D())

        feature_start = feature_start + 1

    return pool.apply_async(applyRadial, (vector_poss, radii, inside_circle, np.array([]), False))

inside_circle = lambda circle_x, circle_y, radius, x, y: True if (((x - circle_x) ** 2) + ((y - circle_y) ** 2) <= radius ** 2) else False

threads = []
feature_count_per_thread = math.ceil(source_layer.GetFeatureCount() / MAX_THREADS)
current_start_thread = 0

global processed_features
processed_features = []

# Experimental
if THREADED: 
    for i in range(MAX_THREADS):
        if current_start_thread + feature_count_per_thread < source_layer.GetFeatureCount():
            threads.append(threadedRadial(current_start_thread, feature_count_per_thread, inside_circle))
            current_start_thread += feature_count_per_thread + 1
        else:
            threads.append(threadedRadial(current_start_thread, source_layer.GetFeatureCount() - current_start_thread, inside_circle))
    
    for t in threads:
        t[0].start()

    for t in threads:
        try:
            t[0].join()
            result = t[1]
            for tuple in result:
                array[tuple[0], tuple[1]] = 1.0
        except KeyboardInterrupt:
            sys.exit()
    

    print("len: %d" % len(processed_features))
    print("get: %d" % source_layer.GetFeatureCount())
# Working
else:
    count = 0
    for feature in source_layer:
        radius_proj = (feature.GetField("KRONENDURC")) * 3 / 2
        vector_pos = feature.GetGeometryRef().GetPoint_2D()
        result = findRadius(vector_pos, radius_proj, inside_circle, True)
        for tuple in result:
            array[int(tuple[0]), int(tuple[1])] = 1.0
        count += 1
    
    print("count: %d" % count)
    print("len: %d" % len(processed_features))
    print("get: %d" % source_layer.GetFeatureCount())


plt.figure()
plt.imshow(array)
plt.show()

#target_band = target_ds.GetRasterBand(1)
#target_band.WriteArray()