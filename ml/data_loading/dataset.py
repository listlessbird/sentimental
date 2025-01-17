import numpy as np
from torch.utils.data import Dataset
import torch
import pandas as pd
from transformers import AutoTokenizer
import cv2
import os
class MeldDataset(Dataset):
    def __init__(self, csv_path: str, video_path: str) -> None:
        self.data = pd.read_csv(csv_path)
        self.video_path = video_path
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        self.emotion_map ={
            "anger": 0,
            "disgust": 1,
            "fear": 2,
            "joy": 3,
            "neutral": 4,
            "sadness": 5,
            "surprise": 6,
        }
        
        self.seniment_map = {
            'negative': 0,
            'neutral': 1,
            'positive': 2,
        }
        
    def _load_frames(self, video_path: str) -> torch.FloatTensor:
        frames = np.zeros((30, 224, 224, 3), dtype=np.float32)
        
        frames_appended = 0
        
        video_capture = cv2.VideoCapture(video_path)
        
        if not video_capture.isOpened():
            raise ValueError(f"Video not found: {video_path}")
        
        try:
            
            total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            # print("Total frames in video:", total_frames)

            seek_space = np.linspace(0, total_frames - 1, 30)
            
            # print("seek space: ", seek_space)
            
            ret, frame = video_capture.read()
            
            if not ret or frame is None:
                raise ValueError(f"Error loading video: {video_path}")
                
            for i in range(len(seek_space)):
                
                seek_to = round(seek_space[i])
                
                # print(f"Seeking to frame: {seek_to}")
                
                video_capture.set(cv2.CAP_PROP_POS_FRAMES, seek_to)
                   
                ret, frame = video_capture.read()
                
                # print(f"Frame {i} : { 'Success' if ret else 'Failed' }")
                
                if not ret or frame is None:
                    break
                
                frame = cv2.resize(frame, (224, 224))
                
                # TODO: see if i want to consider changing the color space
                # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = frame.astype(np.float32)
                frame /= 255.0
                frames[i] = frame
                frames_appended += 1
            
                            
        except Exception as e:
            raise ValueError(f"Error loading video: {video_path}, error: {str(e)}")
        
        finally:
            video_capture.release()
            
            
        if frames_appended == 0:
            raise ValueError(f"No frames could be extracted from video: {video_path}")
       
       
            
        # rearrange from np/cv2 to pytorch
        return torch.FloatTensor(frames).permute(0, 3, 1, 2)
            

        
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        
        row = self.data.iloc[idx]
        video_name = f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.mp4"
        video_path = os.path.join(self.video_path, video_name)
        
        if not os.path.exists(video_path):
            print(f"Video not found: {video_path}")
            raise ValueError(f"Video not found: {video_path}")
        
        text_inputs = self.tokenizer(row['Utterance'], return_tensors="pt", padding='max_length', truncation=True, max_length=128)
        
        frames = self._load_frames(video_path)
        
        # print(frames)
    
    
    
if __name__ == "__main__":
    meld = MeldDataset("data/metadata/dev.csv", "data/raw/extracted/clips/dev/dev_splits_complete")
    
    meld[0]
    
    