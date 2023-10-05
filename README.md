# 7100_voiceConversion

## AutoVC Baseline
* Vanilla autoencoder serves as a measure of how well the network is reconstructing the input data

## Possible Modification
Add ASR for linuistic content feature \
Use an extra pitch extractor to remove any pitch information for the encode embedding

## Filename Notes
`3000000-BL.ckpt` pretrained speech encoder Wan et al. [2018] \
`g_03280000` pretrained vocoder "HiFi-GAN" \
`checkpoint_step001000000_ema.pth` pretrained vocoder Wavenet

## Inferencing
1. `make_spect.py` to convert audio into mel-spec and output to `spmel/`
2. `make_metadata.py` 
3. `conversion.ipynb` use trained autoVC model to generate `results.pkl`

## Progress
* data cleaning for training

## Resources
https://github.com/KnurpsBram/AutoVC_WavenetVocoder_GriffinLim_experiments/blob/master/AutoVC_WavenetVocoder_GriffinLim_experiments_17jun2020.ipynb


