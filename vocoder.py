import torch
import soundfile as sf
import pickle
from synthesis import build_model
from synthesis import wavegen

if __name__ == '__main__':
    spect_vc = pickle.load(open('/home/ytang363/7100_voiceConversion/11-02_results.pkl', 'rb'))
    device = torch.device("cuda")
    model = build_model().to(device)
    pretrain_vocoder = '/home/ytang363/7100_voiceConversion/pretrain/checkpoint_step001000000_ema.pth'
    checkpoint = torch.load(pretrain_vocoder)
    model.load_state_dict(checkpoint["state_dict"])

    count = 0
    for spect in spect_vc:
        if count == 11:
            break

        name = spect[0]
        c = spect[1]
        print(name)
        waveform = wavegen(model, c=c)
        sf.write('oct-16_'+name+'.wav', waveform, samplerate=16000)
        count += 1