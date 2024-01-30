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
* `make_metadata4test.py` use to generate `metadata.pkl`, change the variable `process_uttr`
* `conversion.ipynb` use trained autoVC model to generate `results.pkl`
* `vocoder.ipynb` load in the `results.pkl` and the pretrained vocder to output the coversion

## To-Do 
* Explore different speaker encoders
* New train.pkl bug

## Note (Update: 11/02)
The current `metadata.pkl` has these speaker and utterance\
file name: p225_298_mic1.npy, shape: (194, 80)\
file name: p226_076_mic1.npy, shape: (180, 80)\
file name: p227_091_mic1.npy, shape: (190, 80)\
file name: p228_157_mic1.npy, shape: (279, 80)\
file name: p229_305_mic1.npy, shape: (146, 80)\
file name: p230_017_mic1.npy, shape: (249, 80)\
file name: p231_361_mic1.npy, shape: (137, 80)\
file name: p232_328_mic1.npy, shape: (189, 80)\
file name: p233_386_mic1.npy, shape: (136, 80)\
file name: p234_245_mic1.npy, shape: (187, 80)\
file name: p236_268_mic1.npy, shape: (195, 80)

`metadata_001` has speaker ['p225', 'p226', 'p227', 'p228', 'p229', 'p230', 'p231', 'p232', 'p233', 'p234', 'p237']

## Resources
https://github.com/KnurpsBram/AutoVC_WavenetVocoder_GriffinLim_experiments/blob/master/AutoVC_WavenetVocoder_GriffinLim_experiments_17jun2020.ipynb


