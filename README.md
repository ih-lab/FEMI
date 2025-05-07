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
You can fine-tune FEMI with your own IVF image dataset in a supervised setting. The code for fine-tuning can be found in the `main_finetune.py` script. You can perform either classification or regression with the model. The model takes in wither image or videos.

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
