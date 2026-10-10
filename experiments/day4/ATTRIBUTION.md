# Real-video diagnostic sources and attribution

## DAVIS footage

The real video uses the **car-roundabout** sequence from DAVIS, downloaded as
individual original 854 x 480 JPEG frames from the official ETH Zurich server.
The optional second sequence is **car-shadow**. Source frame indices, download
URLs and SHA-256 hashes are recorded in `source_manifest.json` for every run.

- Dataset: https://davischallenge.org/
- Official browsing page: https://davischallenge.org/davis2016/browse.html
- Its image URL definition: https://davischallenge.org/json_data/global.js
- Frame source: https://graphics.ethz.ch/Downloads/Data/Davis/files/sequences/
- Official dataset archive reviewed for terms:
  https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip

The archive's `DAVIS/README.md` contains inconsistent license descriptions:
its Credits section refers to Creative Commons Attribution 4.0, while its
Terms of Use section explicitly links **CC BY-NC 4.0**. This research
demonstration conservatively follows **CC BY-NC 4.0** for the footage and derived
preview videos: https://creativecommons.org/licenses/by-nc/4.0/ .
The archive's `DAVIS/SOURCES.md` lists externally sourced exceptions; neither
`car-roundabout` nor `car-shadow` appears in that list. Attribution is therefore
to the DAVIS authors, as provided by the official archive.

Relevant dataset publications:

- Federico Perazzi, Jordi Pont-Tuset, Brian McWilliams, Luc Van Gool,
  Markus Gross and Alexander Sorkine-Hornung. *A Benchmark Dataset and Evaluation
  Methodology for Video Object Segmentation.* CVPR 2016.
- Jordi Pont-Tuset, Federico Perazzi, Sergi Caelles, Pablo Arbeláez,
  Alexander Sorkine-Hornung and Luc Van Gool. *The 2017 DAVIS Challenge on Video
  Object Segmentation.* arXiv:1704.00675, 2017.

Modifications: selected frames are annotated with new, approximate,
assistant-visually-authored **part polygons**. These are not human ground truth.
These labels are ours for this diagnostic; they are
not official DAVIS annotations. Preview videos add labels, predicted masks,
side-by-side comparisons, and a non-generative color edit of one selected part.
No endorsement by DAVIS or its authors is implied.

## SAM 2.1 model and implementation

- Repository: https://github.com/facebookresearch/sam2
- Pinned code revision: `2b90b9f5ceec907a1c18123530e92e794ad901a4`
- Model: SAM 2.1 Hiera Tiny
- Configuration: `configs/sam2.1/sam2.1_hiera_t.yaml`
- Checkpoint:
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt
- Upstream code and checkpoint license: **Apache 2.0**,
  https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/LICENSE
- Nikhila Ravi and collaborators. *SAM 2: Segment Anything in Images and Videos.*
  arXiv:2408.00714, 2024.

The source experiment downloads these dependencies rather than redistributing
model weights. The footprint and speed must be measured on the actual runtime;
the upstream A100 benchmark numbers do not describe this T4 experiment.

SAM2 may have encountered DAVIS in training or fine-tuning. These videos serve
as a practical pipeline demonstration, not an unseen-data generalization test.
