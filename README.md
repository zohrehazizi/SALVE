# SALVE
## _Self-supervised Adaptive Low-light Video Enhancement_


This repo containes codes for SALVE which is a self-supervised adaptive low-light video enhancement method. Paper can be found on https://arxiv.org/abs/2009.01385.

Please cite the following reference if you use SALVE:
```sh
@article{azizi2022salve,
  title={SALVE: Self-supervised Adaptive Low-light Video Enhancement},
  author={Azizi, Zohreh and Kuo, C-C Jay},
  journal={arXiv preprint arXiv:2212.11484},
  year={2022}
}
```

## Usage

- Clone the repo and `cd` to it.
- Run `conda env create -f salve.yml` to install the dependencies.
- Run `conda activate salve`.
- Place the folder coontaining frames of the dark video into `dark` folder. (If you need to synthesize dark frames from normal-light frames, use `darken.py`.)
- Run `main.py` to enhnace the video.
- Find the enhanced frames in `enhanced` folder.
