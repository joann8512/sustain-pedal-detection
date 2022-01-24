# sustain-pedal-detection

## Baseline
* [CNN approach](https://github.com/beiciliang/sustain-pedal-detection)
    - Data Preprocessing (MAESTRO Dataset):
        * Structure:
            --MAESTRO
            |---- Rendered
            |
            |---- Metadata
            |-------- filename-2004.txt
            |-------- filename-2005.txt
            |-------- ......
            |-------- train.txt (e.g. 2004/filename1, 2004/filename2)
            |-------- valid.txt
            |-------- test.txt
            |
            |---- pedal-dataset
            |-------- pedal-onset-dataset
            |------------ pedal
            |------------ non-pedal
            |-------- pedal-segment-dataset
            
        -  [x]Run `data_preprocess.ipynb` to create non-pedal midi pairs
        -  [x]Render pedal/non-pedal audio pairs
        * [x]Use [ReaRender](https://github.com/YatingMusic/ReaRender) to traverse through dataset.
        
        - [x]Run `dataset_preparation.ipynb` to create train/valid/test splits
        - [x]Run `main_preprocess.py` to prepare segments + save csv files.
        - [ ]Run `main_decode.py` to decode audio files into npy.  
        
    - Training: 