from typing import Optional, TypedDict, Union
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
from transformers import AutoTokenizer
import cv2
import os
import subprocess
import torchaudio

class MeldSample(TypedDict):
    video_frames: torch.FloatTensor
    audio_features: torch.FloatTensor
    emotion_label: torch.Tensor
    sentiment_label: torch.Tensor
    text_inputs: dict[str, torch.Tensor]


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
            
    def _load_audio(self, video_path: str) -> torch.FloatTensor:
        audio_path = video_path.replace(".mp4", ".wav")
        
        try:
            subprocess.run(
                [
                    'ffmpeg',
                    '-i',
                    video_path,
                    '-vn',
                    '-acodec',
                    'pcm_s16le',
                    '-ar',
                    '16000',
                    '-ac',
                    '1',
                    audio_path
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            
            waveform, sample_rate = torchaudio.load(audio_path)
            
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)

            mel_spectrogram = torchaudio.transforms.MelSpectrogram(
                sample_rate=16000,
                n_mels=64,
                n_fft=1024,
                hop_length=512,
            )
            
            mel_spec = mel_spectrogram(waveform)
            
            mel_spec = ( mel_spec - mel_spec.mean()) / mel_spec.std()
            
            # pad the time dimension if it is too short
            if mel_spec.size(2) < 300:
                padding = 300 - mel_spec.size(2)
                mel_spec = torch.nn.functional.pad(mel_spec, (0, padding))
            else:
                mel_spec = mel_spec[:, :, :300]
        
            return mel_spec
        except subprocess.CalledProcessError as e:
            raise ValueError(f"Error loading audio: {audio_path}, error: {str(e)}")
        
        except Exception as e:
            raise ValueError(f"Error loading audio: {audio_path}, error: {str(e)}")
        
        
        finally:
            if os.path.exists(audio_path):
                os.remove(audio_path)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx: Union[int, torch.Tensor]) -> Optional[MeldSample]:
        
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        
        try:
            row = self.data.iloc[idx]
            video_name = f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.mp4"
            video_path = os.path.join(self.video_path, video_name)

            if not os.path.exists(video_path):
                print(f"Video not found: {video_path}")
                raise ValueError(f"Video not found: {video_path}")

            text_inputs = self.tokenizer(row['Utterance'], return_tensors="pt", padding='max_length', truncation=True, max_length=128)

            frames = self._load_frames(video_path)
            audio = self._load_audio(video_path)

            emotion = self.emotion_map[row['Emotion'].lower()]
            sentiment = self.seniment_map[row['Sentiment'].lower()]

            return {
                "video_frames": frames,
                "audio_features": audio,
                'emotion_label': torch.tensor(emotion),
                'sentiment_label': torch.tensor(sentiment),
                "text_inputs": {
                        "input_ids": text_inputs['input_ids'].squeeze(),
                        "attention_mask": text_inputs['attention_mask'].squeeze(),
                },
            }
        except Exception as e:
            print(f"Error loading data: {idx}, error: {str(e)}")
            return None
    
def collate_fn(batch):
    batch = list(filter(None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

def prepare(
    train_csv_path: str,
    train_video_path: str,
    dev_csv_path: str,
    dev_video_path: str,
    test_csv_path: str,
    test_video_path: str,
    batch_size: int = 16,
    num_workers: int = 4,
    pin_memory: bool = True,
):
    train_dataset = MeldDataset(train_csv_path, train_video_path)
    test_dataset = MeldDataset(test_csv_path, test_video_path)
    dev_dataset = MeldDataset(dev_csv_path, dev_video_path)
    
    common_kwargs = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': pin_memory,
        'collate_fn': collate_fn,
    }
    
      
    train_loader = DataLoader(train_dataset, shuffle=True, **common_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **common_kwargs)
    dev_loader = DataLoader(dev_dataset, shuffle=False, **common_kwargs)
    
    return train_loader, test_loader, dev_loader
    
    
if __name__ == "__main__":
    
    train_loader, test_loader, dev_loader = prepare(
        'data/metadata/train.csv',
        'data/raw/extracted/clips/train',
        'data/metadata/dev.csv',
        'data/raw/extracted/clips/dev',
        'data/metadata/test.csv',
        'data/raw/extracted/clips/test',
        batch_size=16,
        num_workers=4,
        pin_memory=True,  
    )
    
    # for batch in train_loader:
    #     print(batch['text_inputs']['input_ids'].shape)
    #     print(batch['emotion_label'].shape)
    #     break