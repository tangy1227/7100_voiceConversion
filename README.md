# 7100_voiceConversion

## AutoVC Baseline
* vanilla autoencoder serves as a measure of how well the network is reconstructing the input data

## Possible Modification
Add ASR for linuistic content feature
Use an extra pitch extractor to remove any pitch information for the encode embedding

## Filename Notes
`3000000-BL.ckpt` pretrained speech encoder Wan et al. [2018] <br>
`g_03280000` pretrained vocoder "HiFi-GAN"

## Inferencing
1. `make_spect.py` to convert audio into mel-spec
2. `make_metadata.py` 

## Resources
https://github.com/KnurpsBram/AutoVC_WavenetVocoder_GriffinLim_experiments/blob/master/AutoVC_WavenetVocoder_GriffinLim_experiments_17jun2020.ipynb


