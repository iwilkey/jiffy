""" This script is used to translate the Jiffy Dataset's URLs into a folder of GIFs for a unique client ID.
Author: Ian Wilkey
Copyright (C) 2023. All rights reserved.
https://www.iwilkey.com
"""

import os
import requests
import csv
import random
from PIL import Image, ImageFilter, ImageSequence, ImageEnhance, ImageOps

class DATAQ:
    """ Dataset acquisition object for JIFF-Y.
    """
    
    def __init__(self, client_id : str, **kwargs):
        """ DATAQ Constructor.
        """
        self.client_id = client_id
        self.data_path = "./target/"
    
    def get_client_id(self):
        """ Returns the unique client ID for a DATAQ request.
        Returns:
            str: unique clinet ID.
        """
        return self.client_id
    
    def __download(self, prompt, url):
        """ Download a GIF from a URL and place it in the ./dl/ folder.
        """
        # Request the GIF as a stream.
        r = requests.get(url, stream=True)
        r.raise_for_status()
        # Write the GIF stream to the client's dataset folder.
        with open(self.data_path + "target.gif", 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    def __preprocess_dataset_directory(self):
        """ Checks to see if the dataset directory exists, and clears all files in it if it does.
        """
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
        else:
            # Clear all the files in the directory, if it exists.
            for filename in os.listdir(self.data_path):
                file_path = os.path.join(self.data_path, filename)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(f"Failed to delete {file_path}. Reason: {e}")

    def get(self, dataset_location : str, target_phrase : str, fixed_frames : int):
        """ Download a specified amount of GIFs from the Jiffy Dataset.
        """
        self.__preprocess_dataset_directory()
        # Keep a record of found matches.
        phrase_target : dict = {}
        # Open Data CSV.
        with open(dataset_location, "r") as f:
            # Tokenize data.
            reader = csv.reader(f, delimiter="\t")
            # Look through each description (label).
            for row in reader:
                # If the CSV is out of entries, stop looking.
                try:
                    phrase = row[1]
                except:
                    break
                # Check to see if the target is found in the label.
                if target_phrase in phrase:
                    # Keep track of the target GIFs.
                    phrase_target[row[1]] = row[0]
        # Check to see if there were any GIFs containing the target phrase.
        if len(phrase_target) == 0:
            print("[JIFF-Y DATAQ] There was no GIFs containing your target phrase.")
            return False
        # Pick a random GIF from the phrase_target GIFs to be the target GIF.
        target_gif_key = random.choice(list(phrase_target.keys()))
        # Populate directory with target GIF.
        self.__download(target_phrase, phrase_target[target_gif_key])
        self.__pad_target(fixed_frames)
        return True
    
    def __pad_target(self, fixed_frames : int):
        """ Pad the target GIF to a fixed number of frames.
        """
        path_to_target = self.data_path + "target.gif"
        target = Image.open(path_to_target)
        # get the number of frames in the GIF
        num_frames = target.n_frames
        # calculate the number of additional frames needed
        num_additional_frames = fixed_frames - num_frames
        if num_additional_frames < 0:
            # slice the original GIF to keep only the first `total_frames` frames
            frames = []
            for i in range(fixed_frames):
                target.seek(i)
                frame = target.copy()
                frames.append(frame)
        else:
            # create a new list of frames with the original frames followed by duplicates of the last frame
            frames = []
            for i in range(num_frames):
                target.seek(i)
                frame = target.copy()
                frames.append(frame)
            last_frame = frames[-1]
            for i in range(num_additional_frames):
                frames.append(last_frame.copy())
        frames[0].save(path_to_target, save_all=True, append_images=frames[1:], loop=0)

data = DATAQ("localhost")
data.get("./tgif-v1.0.csv", "cat", 16)
