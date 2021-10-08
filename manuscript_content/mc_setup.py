"""Functions for setting up the environment necessary for each piece of content
created in manuscript_content.ipynb
"""

import mc_functions as mcf


def poi_trends_setup():
    global fubu
    global orac
    global poi_coords
    global poi_data
    
    try: fubu
    except NameError: 
        fubu = mcf.load_fubu(fubu_fp)

    try: orac
    except NameError: 
        orac = mcf.load_orac(orac_fp, fubu)

    try: poi_coords
    except NameError:
        poi_coords = mcf.load_poi_coords(poi_coords_fp)

    try: poi_data
    except NameError:
        poi_data = mcf.get_poi_data(poi_coords, fubu, orac)
        
    return poi_data



def poi_masie_trends_setup():
    global fubu
    global orac
    global masie_polys
    global affine
    global zs_df
    global poi_coords
    global poi_data
    global poi_masie_df
    
    try: fubu
    except NameError: 
        fubu = mcf.load_fubu(fubu_fp)

    try: orac
    except NameError: 
        orac = mcf.load_orac(orac_fp, fubu)
        
    try: masie_polys
    except NameError:
        masie_polys = mcf.get_masie_polys(masie_fp)

    try: affine
    except NameError:
        affine = mcf.get_fubu_affine(fubu_fp)
    
    try: zs_df
    except NameError:
        zs_df = mcf.run_zonal_stats(fubu, orac, masie_polys, affine)

    try: poi_coords
    except NameError:
        poi_coords = mcf.load_poi_coords(poi_coords_fp)

    try: poi_data
    except NameError:
        poi_data = mcf.get_poi_data(poi_coords, fubu, orac)
    
    try: poi_masie_df
    except NameError:
        poi_masie_df = mcf.concat_masie_poi_data(zs_df, poi_data)
        
    return poi_masie_df


def masie_trends_setup():
    global fubu
    global orac
    global masie_polys
    global affine
    global zs_df
    
    try: fubu
    except NameError: 
        fubu = mcf.load_fubu(fubu_fp)

    try: orac
    except NameError: 
        orac = mcf.load_orac(orac_fp, fubu)
        
    try: masie_polys
    except NameError:
        masie_polys = mcf.get_masie_polys(masie_fp)

    try: affine
    except NameError:
        affine = mcf.get_fubu_affine(fubu_fp)
    
    try: zs_df
    except NameError:
        zs_df = mcf.run_zonal_stats(fubu, orac, masie_polys, affine)
        
    return zs_df