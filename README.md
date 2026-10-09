<h1 align="center"> Official: Signformer is all you need: Towards Edge AI for Sign Language
</h1>

[![arXiv](https://img.shields.io/badge/arXiv%20paper-2410.06940-b31b1b.svg)](https://arxiv.org/abs/2411.12901v1)&nbsp;
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/signformer-is-all-you-need-towards-edge-ai-1/gloss-free-sign-language-translation-on)](https://paperswithcode.com/sota/gloss-free-sign-language-translation-on?p=signformer-is-all-you-need-towards-edge-ai-1)

![scheme](radar.jpg)
## Outline

🚀 In this work, we present the World's 1st LLM-comparable "Small Language Model" as a from-scratch model for Sign Language Translation

🚀 Signformer-Full achieves new 2nd place in Gloss-Free sign language translation leaderboard of 2024 

🥳 Signformer is the least parametric model across all scoreboards, Signformer-Feather of 0.57 Million achieves TOP5

 
## Dataset: RSL-News
The data is read with [rsl-news-loader](https://github.com/radumicea/rsl-news-loader), from `data_path` (default
`../RSL-News`), or downloaded there first from the Hugging Face dataset repo `hf_repo` (only the
files of the splits being used):

    RSL-News/
        manifests/<channel>_manifest.json                     # episodes, each with "split": "train" | "val" | "test"
        dataset/<Channel>/<episode>/segment_<i>.json          # sentences: start, end (seconds), text
        dataset/<Channel>/<episode>/segment_<i>.bsl5k.npy     # feature windows of the segment
        tokenizer/spm_unigram_lowercase_16k.model             # the text is lowercased, then tokenized

A sample is one sentence: the feature windows that lie entirely between its timestamps, i.e. the
windows of the video cropped to the sentence (`window_size` 8, `window_stride` 2, `fps` 25), and its token ids.

* Install required packages using the `requirements.txt` file (it installs rsl-news-loader).
    `pip install -r requirements.txt`

## Usage
### Train
  `python -m main train [CONFIG PATH]`
### Resume an interrupted training (from `model_dir/latest.ckpt`)
  `python -m main train [CONFIG PATH] --resume`
### Test
  `python -m main test [CONFIG PATH] --ckpt [CHECKPOINT PATH]`

## BIBTEX
```bibtex
@article{eta2024signformer,
      title={Signformer is all you need: Towards Edge AI for Sign Language}, 
      author={Eta Yang},
      year={2024},
      journal={arXiv preprint arXiv:2411.12901}, 
}
```
