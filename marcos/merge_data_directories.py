#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 11:14:31 2023

@author: marcos

It expects BASE_DIR to have files in directories, each directoy containing one
pair type (large-large, or small-random, for isntance). The program will place
all the files in a single directory given by data_dir, rename them according
to the pair type and create a file sumarizing some data regarding the runs.

"""

from pathlib import Path
import shutil

import pandas as pd

import pyabf

BASE_DIR = '/data/marcos/FloClock_data/todos los registros - mecamilamina'

BASE_DIR = Path(BASE_DIR)
data_dir = BASE_DIR.with_name( 'data - mecamilamina')

# if the data directory doesn't exist, create it
if not data_dir.exists():
    data_dir.mkdir()
    
# initialize the pair data file:
pair_guide_columns = 'rec', 'par', 'ch1', 'ch2', 'name', 'mec_start', 'mec_start(sec)', 'duration(min)', 'samplerate', 'comment', 
translation_keys = {'large-large' : 'LL', 'large-random' : 'LR',
                    'small-large': 'LS', 'small-random' : 'SR', 'small-small' : 'SS',
                    'large' : 'L', 'small': 'S', 'random' : 'R'}
par_guide_dict = {k:[] for k in pair_guide_columns}

# read all files
for pair_dir in BASE_DIR.iterdir():
    info_file = pd.read_excel(pair_dir / 'info_registros.xlsx')
    
    par_guide_dict['rec'].extend(info_file['archivo'])
    par_guide_dict['ch1'].extend(info_file['Canal 0'])
    par_guide_dict['ch2'].extend(info_file['Canal 2'])
    par_guide_dict['mec_start'].extend(info_file['entrada mecamilamina'])
    par_guide_dict['comment'].extend(info_file['comentario'])
    
    # save filename and copy files
    for i, file_name in enumerate(info_file.archivo):
        file = pair_dir / f'{file_name}.abf'
        name = translation_keys[pair_dir.name] + f'{i+1:02d}'
        par_guide_dict['name'].append(name)
        par_guide_dict['par'].append(translation_keys[pair_dir.name])
        
        shutil.copy2(file, data_dir / f'{name}.abf')
        
        #save dration and samplerate
        abf = pyabf.ABF(file)
        par_guide_dict['duration(min)'].append(f'{abf.dataLengthMin:.3f}')
        par_guide_dict['samplerate'].append(abf.sampleRate)        
        

# repace nans in the comments column with empty strings
par_guide_dict['comment'] = [c if isinstance(c, str) else '' for c in par_guide_dict['comment']]

# replce the channel values with the single leter versions
par_guide_dict['ch1'] = [translation_keys[c.strip()] for c in par_guide_dict['ch1']]
par_guide_dict['ch2'] = [translation_keys[c.strip()] for c in par_guide_dict['ch2']]

# set the times of effect of mec in seconds
convert_time = lambda t: 60*float(t.split("'")[0]) + float(t.split("'")[1])
par_guide_dict['mec_start(sec)'] = [convert_time(t) for t in par_guide_dict['mec_start']]

# save par_guide
par_guide = pd.DataFrame(par_guide_dict)
par_guide.to_excel(data_dir / 'par_guide.xlsx', index=False)