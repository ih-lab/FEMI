# FEMI

This is the official repository for FEMI, a foundation model for embryology trained on time-lapse images. This repository is written for TensorFlow.

Please contact sur4002@med.cornell.edu or imh2003@med.cornell.edu if you have specific questions.

### Installation
1. Create a new conda environment
```bash
conda create -n femi python=3.9
conda activate femi
```

2. Install dependencies after clioning the repository
```bash
git clone https://github.com/ih-lab/FEMI.git
cd FEMI
pip install -r requirement.txt
```

### Pretrained Model Weights
The pretrained model weights can be accessed through [HuggingFace](https://huggingface.co/ihlab/FEMI).

The weights are compatible with the HuggingFace implementation of the ViT MAE. Information can be found [here](https://huggingface.co/docs/transformers/main/model_doc/vit_mae)

### Pretraining With Your Own Data
You can further pretrain FEMI with your own large-scale IVF image dataset in an SSL setting. The code for pretraining can be found in the `main_pretrain.py` script.

To run pretraining, have your data in the following format:
```
data
└───image1.png
└───image2.png
└───...
```

Then run the following command to pretrain the model. The following command requires GPUs. If you do not have GPUs, you can remove the `--device gpu` flag and the `--GPUs` flag. Instead, use the `--device cpu` flag, which is not recommended for large-scale datasets. `--femi_model_path` should point to the path of the pretrained model weights. All models will be save to the `--output_dir` directory.

```bash
python main_pretrain.py \
--data_path data \
--batch_size 128 \
--epochs 100 \
--output_dir output \
--femi_model_path tf_femi \
--device gpu \
--GPUs '0,1,2,3'
```

If you train your own model from a FEMI checkpoint and want to convert it so that is compatible with Hugging Face, you can use the following:

```python
from transformers import TFViTMAEForPreTraining
import shutil
import os
from huggingface_hub import login

login()

path_to_FEMI = <path_to_FEMI_model>
path_new_tf_model_weights = <path_to_new_SSL_model_weights>
path_new_model_hf = <location_to_save_new_model>
model = TFViTMAEForPreTraining.from_pretrained(path_to_FEMI)
model.load_weights(new_model_weights)
print("Load tensorflow weights successfully")
model.save_pretrained(path_new_model_hf)

# Copying the preprocessor config file
shutil.copyfile(os.path.join(path_to_FEMI, 'preprocessor_config.json'), os.path.join(path_new_model, 'preprocessor_config.json'))
```

### Fine-Tuning With Your Own Data
You can fine-tune FEMI with your own IVF image dataset in a supervised setting. The code for fine-tuning can be found in the `main_finetune.py` script. You can perform either classification or regression with the model. The model takes in either image or videos.

To perform fine-tuning, have your image and video data in the following format, where embryo1, embryo2, etc. are the embryo folders (unique ID numbers) and image1, image2, etc. are the image files:
```
data
└───embryo1
│   └───image1.png
│   └───image2.png
│   └───...
└───embryo2
│   └───image1.png
│   └───image2.png
│   └───...
└───...
```

If performing tasks with an image input, each embryo folder should only contain 1 image. If performing tasks with a video input, all embryo folders should contain a set number of images.

Also needed is a CSV file with the following format:

```
SUBJECT_NO, LABEL, AGE
embryo1, 1, 32
embryo2, 0, 41
...
```

The table should have a header row with the column names `SUBJECT_NO`, `LABEL`, and `AGE`. `SUBJECT_NO` should be the embryo folder name, `LABEL` should be the label for the embryo. If classification, the label should be either 0 or 1. If regression, the labels should be a value between 0 - 1. `AGE` only needs to be included if you want models to inlcude maternal age as a predictor `--use_age 1`.

Then run the following command to fine-tune the model. The following command requires GPUs. If you do not have GPUs, you can remove the `--device gpu` flag and the `--GPUs` flag. Instead, use the `--device cpu` flag, which is not recommended for large-scale datasets. `--femi_model_path` should point to the path of the pretrained model weights. All models will be save to the `--output_dir` directory.

```bash
python main_finetune.py \
--data_path data \
--csv_path data.csv \
--batch_size 32 \
--epochs 100 \
--output_dir output \
--femi_model_path tf_femi \
--device gpu \
--GPUs '0' \
--do_binary_classification 1
```

Example of fine tuning for a regression task with video input and age as a predictor:

```bash
python main_finetune.py \
--data_path data \
--csv_path data.csv \
--batch_size 32 \
--epochs 100 \
--output_dir output \
--femi_model_path tf_femi \
--device gpu \
--GPUs '0' \
--use_age 1 \
--video_input 1 \
--num_frames_in_vid 18 \
--do_regression 1
```

### Preprocessing Data for Foundation Model Training
To preprocess your data for training with FEMI, you can use the `preprocess_data.py` script. This script will preprocess your data and save it in a format that can be used for training. The script adjusts the brightness and crops the images around the embryo.

The script uses a segmentation model that requires PyTorch. If you choose not to preprocess images, you do not need PyTorch.

#### Running just the segmenter on a few images (no TensorFlow required)

If you only want to test the embryo segmenter itself (e.g. to inspect masks/crops on a handful of problematic images) without installing TensorFlow/transformers, install the lighter dependency set and use `run_segmenter_inspect.py`:

```bash
cd preprocessing
python -m venv .venv-segmenter && source .venv-segmenter/bin/activate
pip install -r EmbryoSegmenter/requirements-segmenter.txt --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple

python run_segmenter_inspect.py path/to/image1.png path/to/image2.png
# or process every image in a folder:
python run_segmenter_inspect.py --input-dir path/to/raw_images --output-dir out/
```

For each image, this saves the raw model mask (`_3_raw_mask.png`), a bounding-box overlay (`_4_bbox_overlay.png`), and the aspect-ratio-corrected 224x224 crop (`_6_cropped_224_fixed.png`), plus prints diagnostics (fallback triggers, crop-area fraction, border-touching, possible background false positives) useful for auditing segmentation quality. See `SEGMENTER_ACCURACY_ROADMAP.md` for what these diagnostics mean and known failure modes.

Images flagged for manual review have those three files copied into `<output-dir>/needs_review/`, along with a `needs_review.csv` listing each flagged image and why. By default only images where the full-image fallback triggered are flagged; use `--review-on` to also flag on other diagnostics (`background_fp`, `border`, `distortion`), e.g. `--review-on fallback background_fp`.

#### Improving the segmenter with new/corrected labeled data

If you've found raw images where the segmenter's crop is wrong and want to
retrain it on corrected labels:

1. **Label new masks.** Draw a polygon around the embryo for each
   problem image (e.g. using [Labelme](https://github.com/wkentaro/labelme)),
   producing one `.json` file per image.
2. **Convert JSON annotations to binary mask PNGs:**
   ```bash
   python labelme_json_to_mask.py --input-dir path/to/jsons --output-dir masks/
   ```
3. **Sanity-check image/mask alignment before training** (catches
   mismatched stems, flipped masks, wrong region, etc.):
   ```bash
   python train_segmenter.py \
     --images-dir path/to/raw_images --masks-dir masks/ \
     --preview-only --preview-dir preview/
   ```
   Then look at a few `preview/*_preview.png` files (raw image next to the
   same image with the mask highlighted in green).
4. **Fine-tune** (starts from the current production weights by default —
   pass `--from-scratch` to train from random initialization instead):
   ```bash
   python train_segmenter.py \
     --images-dir path/to/raw_images --masks-dir masks/ \
     --epochs 50 --output-dir EmbryoSegmenter/weights_finetuned
   ```
   If some of your *existing* training masks are known to be inaccurate,
   pass multiple `--images-dir`/`--masks-dir` pairs — a later pair
   overrides an earlier one on filename-stem collisions, so put your
   corrected masks' directory last:
   ```bash
   python train_segmenter.py \
     --images-dir old/raw_images --masks-dir old/masks \
     --images-dir new/raw_images --masks-dir masks/ \
     --epochs 50 --output-dir EmbryoSegmenter/weights_finetuned
   ```
   If you only have a small number of new/corrected examples, also pass
   `--freeze-encoder` to fine-tune only the decoder and reduce overfitting
   risk.
5. **Compare against production before promoting** the new checkpoint —
   don't overwrite `EmbryoSegmenter/weights/` directly:
   ```bash
   python run_segmenter_inspect.py \
     --weights EmbryoSegmenter/weights_finetuned/<best_checkpoint>.pt \
     --input-dir problematic_raw_images/
   ```
   Only copy the new `.pt` file over the production weights once you've
   confirmed it performs at least as well (ideally better) on your known
   problem images.

No new dependencies are needed for training beyond
`EmbryoSegmenter/requirements-segmenter.txt` (torch, numpy, pillow).

##### Using the original paper dataset (e.g. `mask/` + `mask_valid/` folders)

If you have the original ~30-pair dataset the segmenter was first trained
on, it's typically laid out differently from `labelme_json_to_mask.py`'s
output: masks are plain JPEGs named identically to the raw image (not
`<stem>_mask.png`), split across separate `Embryo/` (whole-embryo mask —
use this one), `no_zp/`, and `pn/` subfolders, with the matching raw
images in a `psd/` subfolder (alongside the original `.psd` files, which
are ignored by `train_segmenter.py`). A pre-existing train/validation
split is usually already reflected as two parallel folder trees (e.g.
`mask/` for train, `mask_valid/` for validation). Use `--mask-suffix
"" --mask-ext .jpg` to match this naming, and
`--val-images-dir`/`--val-masks-dir` to preserve the original split
instead of re-shuffling with `--val-frac`:

```bash
python train_segmenter.py \
  --images-dir mask/psd --masks-dir mask/Embryo --mask-suffix "" --mask-ext .jpg \
  --val-images-dir mask_valid/psd --val-masks-dir mask_valid/Embryo \
  --preview-only --preview-dir preview/   # sanity-check alignment first

python train_segmenter.py \
  --images-dir mask/psd --masks-dir mask/Embryo --mask-suffix "" --mask-ext .jpg \
  --val-images-dir mask_valid/psd --val-masks-dir mask_valid/Embryo \
  --epochs 50 --output-dir EmbryoSegmenter/weights_finetuned
```

The masks' mild JPEG compression artifacts (not perfectly 0/255) are not
a problem: masks are resized with nearest-neighbor and thresholded at 127
before use, which removes any intermediate values regardless of source
format.

To combine original + new Labelme data in one run, pass **one
`--mask-suffix` / `--mask-ext` per `--images-dir` pair** (they must align
in order). Do **not** pass a single `--mask-suffix "" --mask-ext .jpg` at
the top — that applies to *every* pair and will cause new Labelme masks
(`foo_mask.png`) to be skipped:

```bash
python train_segmenter.py \
  --images-dir mask/psd --masks-dir mask/Embryo \
  --images-dir new_labels/raw_images --masks-dir new_labels/masks \
  --mask-suffix "" --mask-suffix _mask \
  --mask-ext .jpg --mask-ext .png \
  --val-images-dir mask_valid/psd --val-masks-dir mask_valid/Embryo \
  --freeze-encoder \
  --epochs 50 --output-dir EmbryoSegmenter/weights_finetuned
```

Validation defaults to `--val-mask-suffix "" --val-mask-ext .jpg` when
`--val-images-dir` is used (matching `mask_valid/`).
