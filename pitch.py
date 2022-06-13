from pydub import AudioSegment
import numpy as np
import matplotlib.pyplot as plt
import lib
import array

"""
本文件为提取音频为音高数据的样例。
音高提取以之前的去噪为基础，采用的是去噪的输出结果denoised_audio.mp3.
"""

# 提取音频，判断是否为双声道
audio_name = "denoised_audio.mp3"
audio = AudioSegment.from_file(
    audio_name, format='mp3', frame_rate=44100)
audio_data = np.array(
    audio.get_array_of_samples(), dtype=np.int16)
if audio.channels == 2:
    print("channel = 2")
    audio_data = audio_data.reshape(2, -1)

# 设置取样率和算法的时间切片大小
frame_rate = 44100
piece_size = 0.1  # how many seconds as a FFT period
print("raw data: ", audio_data.shape)

# 归一化后进行短时FFT，提取频率
audio_data = audio_data[0]
audio_data = audio_data / (np.max(audio_data))
result = lib.SS_pitch_extraction(audio_data, frame_rate, piece_size)

# 最后，提取音高并输出
tune_list = lib.SS_tune_export(result)
for tune in tune_list:
    if tune != []:
        print(tune)
