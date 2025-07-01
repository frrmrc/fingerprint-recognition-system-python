#import os
import numpy as np
import cv2
import pandas as pd
import datetime
from pathlib import Path
import os


def upload(fp): #builds paths, creates folders if needed and saves data for the upload ( orientation map, minutiae, resized img)
    file_name = Path(fp.path).stem  
    output_dirs = {name: Path(name) for name in ["orientationmaps", "minutiae", "cron_upload"]}
    for file in output_dirs.values():
        file.mkdir(parents=True, exist_ok=True)
    output_map_path = Path("orientationmaps") / f"{file_name}.npy"
    output_minutiae_path = Path("minutiae") / f"{file_name}.csv"
    output_cron_upload_path = Path("cron_upload") / f"{file_name}.bmp"
    np.save(output_map_path, fp.orientation_map) 
    fp.minutiae.to_csv(output_minutiae_path, index=False)
    cv2.imwrite(str(output_cron_upload_path), fp.resized)             

def save_search(fp, src): #builds paths, creates the folder and saves data from the research. Returns the specific path of the search history
    dir_path = Path("cron_ricerche") / datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    dir_path.mkdir(parents=True, exist_ok=True)
    original_img=Path(fp.path)
    percorso_destinazione = dir_path / original_img.name
    percorso_destinazione.write_bytes(original_img.read_bytes())
    mtchs=Path(dir_path) / "risultati.csv"
    df=pd.DataFrame(src.matches, columns=["file", "score"])
    df.to_csv(mtchs, index=False)
    if fp.steps_img: fp.steps_img.savefig(dir_path / "steps.png") 
    if src.match_img: src.match_img.savefig(dir_path / "match.png") 
    return str(dir_path.resolve()) 

def check_db(): # checks if the user has already uploaded any files, returns false if not
    dir=Path('cron_upload')    
    return dir.is_dir() and  any(dir.iterdir()) 
