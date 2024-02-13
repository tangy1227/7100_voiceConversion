import os
from sklearn.model_selection import train_test_split
import shutil

# Set the path to your main directory
main_directory = '/home/ytang363/7100_voiceConversion/VCTK-Corpus-0.92/spmel-16k-split' # wav-16k-split, spmel-16k-copy
wav_dir = '/home/ytang363/7100_voiceConversion/VCTK-Corpus-0.92/wav-16k-split'

# Get a list of all subdirectories
all_directories = os.listdir(main_directory)

# Split the data into train, test using scikit-learn
train_set, test_set = train_test_split(all_directories, test_size=0.1, random_state=42)
validation_set, test_set = train_test_split(test_set, test_size=0.5, random_state=42)

# Define the paths for the train, validation, and test sets
train_path = os.path.join(main_directory, 'train')
validation_path = os.path.join(main_directory, 'validation')
test_path = os.path.join(main_directory, 'test')

train_path_wav = os.path.join(wav_dir, 'train')
validation_path_wav = os.path.join(wav_dir, 'validation')
test_path_wav = os.path.join(wav_dir, 'test')

# Create the train, validation, and test directories if they don't exist
os.makedirs(train_path, exist_ok=True)
os.makedirs(validation_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)

os.makedirs(train_path_wav, exist_ok=True)
os.makedirs(validation_path_wav, exist_ok=True)
os.makedirs(test_path_wav, exist_ok=True)

# Move the directories to their respective sets
for directory in train_set:
    shutil.move(os.path.join(main_directory, directory), os.path.join(train_path, directory))
    shutil.move(os.path.join(wav_dir, directory), os.path.join(train_path_wav, directory))

for directory in validation_set:
    shutil.move(os.path.join(main_directory, directory), os.path.join(validation_path, directory))
    shutil.move(os.path.join(wav_dir, directory), os.path.join(validation_path_wav, directory))

for directory in test_set:
    shutil.move(os.path.join(main_directory, directory), os.path.join(test_path, directory))
    shutil.move(os.path.join(wav_dir, directory), os.path.join(test_path_wav, directory))