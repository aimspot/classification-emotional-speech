import os
import pandas as pd
import numpy as np
import librosa
from tqdm import tqdm

def noise(data):
    noise_amp = 0.035*np.random.uniform()*np.amax(data)
    data = data + noise_amp*np.random.normal(size=data.shape[0])
    return data

def stretch(data, rate=0.8):
    return librosa.effects.time_stretch(data, rate=rate)

def shift(data):
    shift_range=int(np.random.uniform(low=5,high=5)*1000)
    return np.roll(data, shift_range)

def pitch(data, sampling_rate, pitch_factor=0.7):
    return librosa.effects.pitch_shift(data, sr=sampling_rate, n_steps=pitch_factor)


def feature_extractors(data, sr):
    result=np.array([])
    zcr=np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result=np.hstack((result,zcr))
    
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
    result = np.hstack((result, chroma_stft))
    
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sr).T, axis=0)
    result = np.hstack((result, mfcc))
    
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms))
    
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sr).T, axis=0)
    result = np.hstack((result, mel)) 
    return result

def get_features (path):
    y, sr =librosa.load(path, duration=2.5, offset=0.6)
    res1=feature_extractors(y, sr)
    result=np.array(res1)
    
    noise_data=noise(y)
    res2=feature_extractors(noise_data, sr)
    result=np.vstack((result,res2))
    
    new_data=stretch(y)
    data_stretch_pitch=pitch(new_data, sr)
    res3=feature_extractors(data_stretch_pitch, sr)
    result=np.vstack((result,res3))
    return result


def main():
    Ravdess = "dataset/audio_speech_actors_01-24/"
    ravdess_directory_list = os.listdir(Ravdess)
    file_emotion = []
    file_path = []
    for dir in ravdess_directory_list:
        actor = os.listdir(Ravdess + dir)
        for file in actor:
            part = file.split('.')[0]
            part = part.split('-')
            file_emotion.append(int(part[2]))
            file_path.append(Ravdess + dir + '/' + file)
            
    emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])

    path_df = pd.DataFrame(file_path, columns=['Path'])
    Ravdess_df = pd.concat([emotion_df, path_df], axis=1)

    Ravdess_df.Emotions.replace({1:'neutral', 2:'calm', 3:'happy', 4:'sad', 5:'angry', 6:'fear', 7:'disgust', 8:'surprise'}, inplace=True)

    X, Y = [], []
    for path, emotion in tqdm(zip(Ravdess_df.Path, Ravdess_df.Emotions)):
        feature = get_features(path)
        for ele in feature:
            X.append(ele)
            Y.append(emotion)

    Features=pd.DataFrame(X)
    Features['Labels']=Y
    Features.to_csv('final_csv_actor.csv',index=False)


if __name__ == "__main__":
    main()