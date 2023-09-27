import pickle

with open('7100_voiceConversion/autovc-master/results.pkl', 'rb') as file:
    data = pickle.load(file)

print(data)