from openai import OpenAI
import sounddevice as sd
import scipy.io.wavfile as wav
import tempfile
import webrtcvad
import collections
import numpy as np
import os

class STT:
    def __init__(self, openai_api_key):
        self.client = OpenAI(api_key=openai_api_key)
        self.samplerate = 16000  # VAD는 8k, 16k, 32k, 48k만 지원
        self.vad = webrtcvad.Vad(2)  # 민감도 0~3
        self.frame_duration_ms = 30  # 10, 20, 30ms 중 선택
        self.frame_size = int(self.samplerate * self.frame_duration_ms / 1000)

    def speech2text(self):
        print("음성 인식 중... (말을 마치면 자동으로 종료됩니다)")
        
        audio_buffer = []
        # 최근 1초(약 33프레임) 동안 소리가 있었는지 판단하기 위한 큐
        ring_buffer = collections.deque(maxlen=30) 
        triggered = False
        voiced_frames = []

        # 스트림 시작
        with sd.InputStream(samplerate=self.samplerate, channels=1, dtype='int16') as stream:
            while True:
                # 30ms 분량의 데이터를 읽음
                frame, overflowed = stream.read(self.frame_size)
                frame_bytes = frame.tobytes()

                # VAD로 음성 여부 판단
                is_speech = self.vad.is_speech(frame_bytes, self.samplerate)

                if not triggered:
                    # 음성 시작 감지 단계
                    ring_buffer.append((frame, is_speech))
                    num_voiced = len([f for f, speech in ring_buffer if speech])
                    if num_voiced > 0.9 * ring_buffer.maxlen:
                        triggered = True
                        for f, s in ring_buffer:
                            voiced_frames.append(f)
                        ring_buffer.clear()
                else:
                    # 음성 종료 감지 단계
                    voiced_frames.append(frame)
                    ring_buffer.append((frame, is_speech))
                    num_unvoiced = len([f for f, speech in ring_buffer if not speech])
                    
                    # 약 0.9초간 침묵이 유지되면 종료
                    if num_unvoiced > 0.4 * ring_buffer.maxlen:
                        print("음성 종료 감지.")
                        break
                
                # 최대 녹음 시간 제한 (안전장치: 7초)
                if len(voiced_frames) > (7000 / self.frame_duration_ms):
                    break

        if not voiced_frames:
            return ""

        # 녹음된 데이터를 하나로 합침
        audio_data = np.concatenate(voiced_frames, axis=0)

        # 임시 파일 저장 및 Whisper 전송
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
                temp_path = temp_wav.name
                wav.write(temp_path, self.samplerate, audio_data)

            with open(temp_path, "rb") as f:
                transcript = self.client.audio.transcriptions.create(
                    model="whisper-1", file=f)
            
            print("STT 결과: ", transcript.text)
            return transcript.text
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)