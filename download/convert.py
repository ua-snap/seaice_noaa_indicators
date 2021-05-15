# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# DOWNLOAD AND CONVERT THE NSIDC 0051 DAILY SEA ICE CONCENTRATION
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def convert_GTiff( fn, output_path ):
    '''
    see here: https://nsidc.org/support/how/how-do-i-convert-nsidc-0051-sea-ice-concentration-data-binary-geotiff
    '''
    header = [u'ENVI',
        u'description = {}'.format(os.path.basename(fn)),
        u'samples = 304',
        u'lines   = 448',
        u'bands   = 1',
        u'header offset = 300',
        u'file type = ENVI Standard',
        u'data type = 1',
        u'interleave = bsq',
        u'byte order = 0',
        u'map info = {Polar Stereographic, 1, 1, -3850000, 5850000, 25000, 25000} projection info = {31, 6378273, 6356889.449, 70, -45, 0, 0, Polar Stereographic} coordinate system string = {PROJCS["Stereographic_North_Pole",GEOGCS["GCS_unnamed ellipse",DATUM["D_unknown",SPHEROID["Unknown",6378273,298.279411123064]],PRIMEM["Greenwich",0],UNIT["Degree",0.017453292519943295]],PROJECTION["Stereographic_North_Pole"],PARAMETER["standard_parallel_1",70],PARAMETER["central_meridian",-45],PARAMETER["faluse_easting",0],PARAMETER["false_northing",0],UNIT["Meter",1]]}',
        u'​band names = {Band 1}'.strip(u'\u200b')]

    with open(fn.replace('.bin','.bin.hdr'), 'w') as f:
        f.write( "\n".join(header) )

    dirname, basename = os.path.split( fn )
    out_tif = basename.replace('.bin', '.tif')
    # year = os.path.basename(dirname)

    out_path = os.path.join(output_path, 'north')
    try:
        if not os.path.exists( out_path ):
            os.makedirs( out_path )
    except:
        pass

    out_tif, ext = os.path.splitext( out_tif )
    # fix dots in naming convention since it is ugly
    out_tif = out_tif.replace('.','-')
    out_fn = os.path.join(out_path, out_tif + ext )
    command = "gdal_translate -q -of GTiff -a_srs '+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +k=1 +x_0=0 +y_0=0 +a=6378273 +b=6356889.449 +units=m \
            +no_defs' -a_nodata 255 -A_ullr -3850000.0 5850000.0 3750000.0 -5350000.0 {} {}".format(fn, out_fn)

    os.system( command )

    # cleanup
    _ = os.remove( fn.replace( '.bin','.bin.hdr' ) )
    print( out_fn ) # for the fun of it!
    rescale_values( out_fn, band=1 )

    # output a version cropped to Alaska.
    # output_filename = '/workspace/Shared/Tech_Projects/SeaIce_NOAA_Indicators/project_data/extents/ak_domain_epsg3411.shp'
    # make_aoi_polygon( output_filename )

    # this is simply the bounds of the aoi polygon from above.
    new_ext = '-2249452.313150307 408646.7125749469 -1702157.453825454 1149853.2079733'

    # now lets clip the files with gdalwarp and the new shapefile...
    cropped_fn = out_fn.replace( out_path, out_path.replace('north', 'alaska') )
    dirname, basename = os.path.split(cropped_fn)
    try:
        if not os.path.exists( dirname ):
            os.makedirs( dirname )
    except:
        pass
    
    command = 'gdalwarp -q -overwrite -te '+ new_ext + ' -srcnodata "250 251 252 253 254 255" -dstnodata "250 251 252 253 254 255" {} {}'.format( out_fn, cropped_fn )
    os.system( command )
    return cropped_fn

def rescale_values( fn, band=1 ):
    with rasterio.open( fn, 'r' ) as rst:
        arr = rst.read( band ).astype( np.float32 )
        ind = np.where( arr <= 250 )
        arr[ind] = arr[ind]/250.0
        meta = rst.meta.copy()
        meta.update(compress='lzw', dtype='float32')

    with rasterio.open( fn, 'w', **meta ) as out:
        out.write( arr.astype( np.float32 ), band )
    return fn

def make_aoi_polygon( output_filename ):
    # make a bounding polygon for the Alaska-based AOI
    from shapely.geometry import Polygon
    import geopandas as gpd

    pol = Polygon([ (-165,72), (-147,72), (-147,69), (-165,69), (-165,72) ]) # from Hajo's Paper we are replicating
    gdf = gpd.GeoDataFrame({'id':[1],'geometry':pol}, crs={'init':'epsg:4326'}, geometry='geometry' )
    gdf.to_crs(epsg=3411).to_file( output_filename )
    return output_filename


if __name__ == '__main__':
    import os, rasterio, shutil
    import numpy as np
    import multiprocessing as mp
    from functools import partial
    import getpass
    import argparse

    # parse some args
    parser = argparse.ArgumentParser( description='Convert raw NSIDC-0051 Dailies' )
    parser.add_argument( "-i", "--input_path", action='store', dest='input_path', type=str, help="directory path to NSIDC_0051 data downloaded using the NSIDC-0051 downloader script" )
    parser.add_argument( "-o", "--output_path", action='store', dest='output_path', type=str, help="directory path to NSIDC_0051 data downloaded using the NSIDC-0051 downloader script" )
    parser.add_argument( "-n", "--ncpus", action='store', dest='ncpus', type=int, help="number of cpus to use" )
    
    # unpack the args
    args = parser.parse_args()
    base_path = args.input_path
    output_path = args.output_path
    ncpus = args.ncpus

    # # # # # FOR TESTING
    # ncpus = 64
    # input_path = '/workspace/Shared/Tech_Projects/SeaIce_NOAA_Indicators/project_data/nsidc_0051/raw/daily'
    # output_path = '/workspace/Shared/Tech_Projects/SeaIce_NOAA_Indicators/project_data/nsidc_0051/prepped2'
    # # # # # # # # # # # 

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # [new] way to download the data is via a Python script thatis returned to the user
    #   when they order the data from this page...
    #   GO HERE: https://nsidc.org/data/nsidc-0051
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

    # list the filenames from input_path
    filenames = [ os.path.join(r,fn) for r,s,files in os.walk( input_path ) for fn in files if fn.endswith( '.bin' )]

    pool = mp.Pool( ncpus )
    f = partial( convert_GTiff, output_path=output_path )
    out = pool.map( f, filenames )
    pool.close()
    pool.join()
