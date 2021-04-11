import glob
import os
import sys
import subprocess
import pdb

"""
Used to convert .m4a files to .wav.
Point at the directory where the speaker directories reside:
path
|-- speaker directories
    |-- speach session directories (each of these contains multiple .m4a files, one for each utterance

Example usage: python m4a_to_wav.py /work/users/idhillon/Speaker-Verification-Capstone/test_data/aac/
"""

def convert(path):
    speaker_dirs = os.listdir(path)
    for speaker in speaker_dirs:
        print('Converting {sp} from AAC to WAV'.format(sp=speaker))
        speaker_dir = os.path.join(path, speaker)
        speech_session_dirs = os.listdir(speaker_dir)

        for speech_session in speech_session_dirs:
            speech_session_dir = os.path.join(speaker_dir, speech_session)
            utterances = os.listdir(speech_session_dir)
            for utterance in utterances:
                utterance_path = os.path.join(speech_session_dir, utterance)
                out_path = utterance_path.replace('.m4a','.wav')
                call_str = 'ffmpeg -y -i %s -ac 1 -vn -acodec pcm_s16le -ar 16000 %s >/dev/null 2>/dev/null' %(utterance_path,out_path)
                out = subprocess.call(call_str, shell=True)

if __name__ == '__main__':
    convert(sys.argv[1])
