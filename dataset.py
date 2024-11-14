import tensorflow as tf
import numpy as np
import glob
from functools import partial


def load_wav(filename):
    # Load encoded wav file
    file_contents = tf.io.read_file(filename)
    # Decode wav (tensors by channels) 
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    # Removes trailing axis
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(16_000, dtype=tf.int64)
    return wav, sample_rate




def preprocess_train(utter_batch,speaker_batch, all_file_list):
    length = tf.random.uniform([], 4, 16, dtype=tf.int32)
    length = length * 160 * 40 - 1
    
    # Using TensorArray for dynamic size
    audio_batch = tf.TensorArray(dtype=tf.float32, size=tf.shape(utter_batch)[0])

    def get_wav(index):
        speaker_idx = index[0].numpy()
        utterance_idx = index[1].numpy()
        return tf.convert_to_tensor(all_file_list[speaker_idx][utterance_idx], dtype=tf.float32)

    for i in range(tf.shape(utter_batch)[0]):
        wav = tf.py_function(func=get_wav, inp=[utter_batch[i]], Tout=tf.float32)
        
        wav_length = tf.shape(wav)[0]
        wav = tf.tile(wav, multiples=[(length - 1) // wav_length + 1])
        
        start_slice = tf.random.uniform([], 0, tf.shape(wav)[0] - length, dtype=tf.int32)
        processed_wav = wav[start_slice:start_slice + length] + tf.random.normal([length], stddev=tf.random.uniform([], 0, 0.001))
        
        # Write processed wav to TensorArray
        audio_batch = audio_batch.write(i, processed_wav)

    # Convert TensorArray to Tensor
    audio_batch = audio_batch.stack()
    return audio_batch, speaker_batch

def preprocess_test(utter_batch,speaker_batch, all_file_list):
    length = 8
    length = length * 160 * 40 - 1
    
    # Using TensorArray for dynamic size
    audio_batch = tf.TensorArray(dtype=tf.float32, size=tf.shape(utter_batch)[0])

    def get_wav(index):
        speaker_idx = index[0].numpy()
        utterance_idx = index[1].numpy()
        return tf.convert_to_tensor(all_file_list[speaker_idx][utterance_idx], dtype=tf.float32)

    for i in range(tf.shape(utter_batch)[0]):
        wav = tf.py_function(func=get_wav, inp=[utter_batch[i]], Tout=tf.float32)
        
        wav_length = tf.shape(wav)[0]
        wav = tf.tile(wav, multiples=[(length - 1) // wav_length + 1])
        
        start_slice = 0
        processed_wav = wav[start_slice:start_slice + length]
        
        # Write processed wav to TensorArray
        audio_batch = audio_batch.write(i, processed_wav)

    # Convert TensorArray to Tensor
    audio_batch = audio_batch.stack()
    return audio_batch, speaker_batch



def create_all_file_list(directory='/kaggle/input/darpa-timit-acousticphonetic-continuous-speech/data/TRAIN/*/*'):
    speaker_list = sorted(glob.glob(directory))
    all_file = [[load_wav(filename)[0].numpy() for filename in sorted(glob.glob(speaker+r'/*.WAV.wav'))] for speaker in speaker_list]
    return all_file

def create_dataset(all_file_list, test=False, batch_size=16):
    num_speaker = len(all_file_list)
    range_tensor = tf.range(num_speaker*10)
    speaker_tensor = tf.math.floordiv(range_tensor, 10)
    utter_tensor = tf.math.floormod(range_tensor, 10)
    speakers = tf.transpose(tf.stack([speaker_tensor,utter_tensor], axis=0))
    speakers = tf.data.Dataset.from_tensor_slices(speakers)
    speaker_tensor = tf.one_hot(speaker_tensor,num_speaker)
    speakers = tf.data.Dataset.zip((speakers, tf.data.Dataset.from_tensor_slices(speaker_tensor) ))
    if not test:
        preprocess = partial(preprocess_train, all_file_list=all_file_list)
        speakers = tf.data.Dataset.shuffle(speakers, buffer_size=speakers.cardinality())
        speakers = tf.data.Dataset.batch(speakers, batch_size=batch_size)
        speakers = tf.data.Dataset.map(speakers, preprocess)
    else:
        preprocess = partial(preprocess_test, all_file_list=all_file_list)
        speakers = tf.data.Dataset.batch(speakers, batch_size=batch_size)
        speakers = tf.data.Dataset.map(speakers, preprocess)
    return speakers

