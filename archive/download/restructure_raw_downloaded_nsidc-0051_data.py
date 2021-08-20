def break_filename( fn ):
    ''' take a NSIDC-0051 raw filename and and break it into its parts'''
    name_elems = ['prefix','date','id','versionA','versionB','domain','ext']
    basename = os.path.basename(fn)
    basename = basename.replace('.','_')
    split_name = basename.split('_')
    return dict(zip(name_elems, split_name))

if __name__ == '__main__':
    import os, glob
    import pandas as pd
    import argparse

    DESC = '''
    Move NSIDC_0051 Data downloaded with their Python downloader script generated
    via their NSIDC-0051 Version 1 Website https://nsidc.org/data/nsidc-0051.

    The tool should be run in a clean, empty folder where it will dump out all of the 
    data files for both monthly and daily (we want the daily) into that empty folder.

    This script will look at the filenames and figure out which group (day/month) they 
    belong to and move them into sub-directories named 'Monthy' and 'Daily'.

    '''

    # parse some args
    parser = argparse.ArgumentParser( description=DESC )
    parser.add_argument( "-p", "--path", action='store', dest='path', type=str, help="path to the folder that the data were downloaded to" )
    
    # parse the args
    args = parser.parse_args()
    path = args.path

    # get the files listed from that directory
    files = glob.glob(os.path.join(path, '*.bin'))

    # breakup the base filenames and dump into a Pandas DataFrame
    df = pd.DataFrame([break_filename(fn) for fn in files])
    df['fn'] = files

    # find any that might be in the South domain -- which we DONT want
    south = df[df.domain == 's']['fn'].tolist()
    # remove 'em
    _ = [os.remove(fn) for fn in south]

    # subset to only the Norths
    df = df[df.domain != 's']
    df['timestep'] = df.date.apply(lambda x: 'daily' if len(x) == 8 else 'monthly')

    # now we can subset by monthly or daily and move into their directories
    for i in ['daily', 'monthly']:
        newpath = os.path.join(path, i)
        if not os.path.exists(newpath):
            _ = os.makedirs(newpath)

        # now move em
        for fn in df[df.timestep == i]['fn']:
            _ = os.system('mv {} {}'.format(fn, os.path.join(newpath,os.path.basename(fn))))
            _ = os.system('mv {} {}'.format(fn+'.xml', os.path.join(newpath,os.path.basename(fn+'.xml'))))
