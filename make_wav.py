import os
from pydub import AudioSegment

dataset_path = '7100_voiceConversion/VCTK-Corpus-0.92/wav48_silence_trimmed'
targetDir = '7100_voiceConversion/VCTK-Corpus-0.92/wav'
if not os.path.isdir(targetDir):
    os.makedirs(targetDir)
    
all_items = os.listdir(dataset_path)
all_items = sorted(all_items, reverse=False)
count = 0

for s in all_items:
    if count == 4:
        break
    
    print(f'current speaker: {s}')
    s_path = os.path.join(dataset_path, s)
    if os.path.isdir(s_path):

        speaker_recordings = os.listdir(s_path)
        # count_record = 0
        speaker_recordings = sorted(speaker_recordings, reverse=False)
        for r in speaker_recordings:
            # if count_record == 4:
            #     break

            save_path = os.path.join(targetDir, s)
            if not os.path.isdir(save_path):
                os.makedirs(save_path)

            if 'mic1' in r:
                r_rename = r.replace(".flac", ".wav")
                export_file = os.path.join(save_path, r_rename)
                read_file = os.path.join(s_path, r)
                flac_file = AudioSegment.from_file(read_file, format="flac")
                flac_file.export(export_file, format='wav')

            # count_record += 1
    count += 1
