# Study of Encoder Embeddings of Any-to-Any Voice Conversion

## AutoVC Baseline
* Vanilla autoencoder serves as a measure of how well the network is reconstructing the input data

## Proposal
Research about different embedding for the VC system \
Speaker encoder: D-Vector \
Linguistic Content: Conformer (ASR), Wav2Vec \
Possible Prosodic Encoder: ? \
Decoder: AutoVC decoder \
Vocoder: Wavenet, HiFi-GAN, Parallel WaveGAN 

## Filename Notes
`model_bl` D-Vector Model

## Model Filename Notes
`3000000-BL.ckpt` pretrained speech encoder Wan et al. [2018] \
`g_03280000` pretrained vocoder "HiFi-GAN" \
`checkpoint_step001000000_ema.pth` pretrained vocoder Wavenet

## Inferencing
* `make_spect.py` to convert audio into mel-spec and output to `spmel/`
* `make_wav.py` use to convert flac to wav for the dataset
* `make_metadata.py` use to generate `train.pkl`
* `make_metadata4test.py` use to generate `metadata.pkl`
* `conversion.ipynb` use trained autoVC model to generate `results.pkl`
* `vocoder.ipynb` load in the `results.pkl` and the pretrained vocder to output the coversion

## To-Do 
* Evaluation metric: compare with the baseline
    * get the trained model to test on more audio files
    * possible to switch to a different vocoder
    * test the difference shape for the `metadata.pkl`, especially the speaker embedding, and spectrogram
* Validation loss implementation

## Resources
https://github.com/KnurpsBram/AutoVC_WavenetVocoder_GriffinLim_experiments/blob/master/AutoVC_WavenetVocoder_GriffinLim_experiments_17jun2020.ipynb


