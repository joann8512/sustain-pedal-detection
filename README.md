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
            
        -  Run `data_preprocess.ipynb` to create non-pedal midi pairs
        -  Render pedal/non-pedal audio pairs
        * Use [ReaRender](https://github.com/YatingMusic/ReaRender) to traverse through dataset.
        
        - Run `dataset_preparation.ipynb` to create train/valid/test splits
        - Run `main_preprocess.py` to prepare segments + save csv files.
        - Run `main_decode.py` to decode audio files into npy.