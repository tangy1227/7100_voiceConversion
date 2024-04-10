# Study of Encoder Embeddings of Any-to-Any Voice Conversion
### [Project Slides](https://docs.google.com/presentation/d/1aN0-b5HOtv_MIFXhAyAO_j3vy228EclpQDngsl8IFBo/edit?usp=sharing)

## [AutoVC](https://github.com/auspicious3000/autovc) Baseline
Zero-shot voice conversion system based on autoencoder

## Proposal
Research and benchmark different speaker embeddings for the voice conversion system \
**Speaker encoder**: D-Vector (baseline), [X-Vector](https://github.com/speechbrain/speechbrain/blob/develop/speechbrain/lobes/models/Xvector.py), [Resemblyzer](https://github.com/resemble-ai/Resemblyzer) \
**Linguistic Content**: No new linguistic encoder, AutoVC encoder contains this info already \
**Prosodic Encoder**: No new prosodic encoder, AutoVC encoder contains the pitch info already \
**Decoder**: AutoVC decoder \
**Vocoder**: Wavenet, HiFi-GAN, Parallel WaveGAN? 

## Filename Notes
`main.py` run training process \
`model_vc.py` model architecture code \
`solver_encoder.py` training code \
`model_bl.py` D-Vector speaker embedding Model \
`synthesis.py` WaveNet vocoder \
`vocoder.py` WaveNet inferencing at background \

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

## Training with different spk embeddings:
* xvec, go to `dataLoader.py` edit the metadata path to `train_xvec.pkl`. change the dim_emb parameter in `main.py` to 512
* res, ... dim_emb parameter in `main.py` is the same with baseline 256

## Testset Note
uttr002: 'Ask her to bring these things with her from the store.'\
uttr010: 'People look, but no one ever finds it.'\

uttr050:
* p231: 'People look, but no one ever finds it.'
* p243: 'Have I really come to this?'
* p272: 'This represents a tough game for us.'
* p279: 'The judge said.'
* p314: 'We have no choice but to shut down.'
* p339: 'It is a hard act to follow, the Winning act.'

uttr150:
* p231: 'We have a clean bill of health.'
* p243: 'Did he trip?'
* p272: 'Labour's Scottish general secretary Alex Rowley was delighted yesterday.'
* p279: 'In each case they were a goal down.'
* p314: 'It was a moment of madness.'
* p339: 'Mind you, all was not lost.'

uttr275:
* p231: 'It does not work that way in Scottish football.'
* p243: 'It is also seeking a national mortgage rescue plan.'
* p272: 'All options are open.'
* p279: 'The script was funny.'
* p314: 'He has lost confidence and weight.'
* p339: 'This time it really could happen.'

uttr390:
* p231: 'It was fit for royalty.'
* p243: 'His record on Government has always been highly effective.'
* p272: 'It's like a basketball.'
* p279: 'However, BBC Scotland was not interested in his work.'
* p314: 'It was early morning.'
* p339: 'He has run a hell of a race.'

## Evaluation Idea
* Find the better model for xvec, `model_16` or `model_64`
* Using `metric-mcd.ipynb` for evaluating the MCD metrics
* For comparing the output quality from the two vocoder, use the `metric-SDR_PESQ.ipynb`
    * WaveNet audio output folder `eval_audio-WaveNet`. or recalculate the output again with the `model_retrained`
* Word Error Rate evaluation using `word_err_rate.ipynb`

## Resources
https://github.com/KnurpsBram/AutoVC_WavenetVocoder_GriffinLim_experiments/blob/master/AutoVC_WavenetVocoder_GriffinLim_experiments_17jun2020.ipynb


