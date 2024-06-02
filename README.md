# **A Generative Approach to Full Sequence Synthetic Micro-Expressions using the Neutral Expression**

A GAN model created based on [GANimation](https://github.com/albertpumarola/GANimation) to remove facial expression and generate synthetic micro-expressions.

## **Setup**
If you already have `conda` installed, I suggest you install `mamba` for faster download of libraries and dependencies.
```
# install mamba
$ conda install -c conda-forge mamba

# create sandbox conda environment and install dependencies
$ mamba env create --file environment.yml
```

## **Data Directory**
Please refer to the following folder structure for the data directory used in training.
```
data_dir
    - imgs
    - aus.pkl
    - train_ids.txt
    - test_ids.txt
```
>`imgs` - directory containing preprocessed images\
>`aus.pkl` - a pickle file contaning the aus extracted from images using OpenFace\
>`train_ids.txt` - list of image names for training\
>`test_ids.txt` - list of image names of validation during training

## **Preparing AUs**
```
python3 data/prepare_aus.py \
--aus_dir "path to directory" \
--output_dir "path to directory" \
```
>`--aus_dir` - path to directory containing CSV files of AUs extracted from OpenFace\
>`--output_dir` - path to save directory


## **Training Model**
```
python3 train.py \
--data_dir "path to directory"  \
--imgs_dir "path to directory"  \
--train_ids_file "path to file" \
--test_ids_file "path to file" \
--aus_file "path to file" \
--expr_name "experiment name" \
--batch_size 25 \
--load_epoch -1 \
```
>`--data_dir` - path to directory containing the necessary files for training\
>`--imgs_dir` - path to directory containing images that have been aligned and cropped with OpenFace\
>`--train_ids_file` - path to TXT file containing the ids for training images\
>`--test_ids_file` - path to TXT file containing the ids for validation images\
>`--aus_file` - path to PKL file containing the AUs for training and test images\
>`--expr_name` - experiment name\
>`--batch_size` - training batch size\
>`--load_epoch` - load model at specified epoch (default = -1)

## **Testing Model**
Please ensure that you have OpenFace installed on your device, as it is needed to extract action units from the input and target images. The code to extract action units is already provided, so only the path to the installed OpenFace needed to be changed in the source code.

For more information on how to install OpenFace, please follow the [guidelines](https://github.com/TadasBaltrusaitis/OpenFace/wiki#installation) provided.

### **Testing Directory**
To test the model, please refer to the following example for the testing directory structure.
```
test_dir
    - ckpts
        - me
            - expr_1
                - net_epoch_30_G.pth
        - neutral
            - expr_2
                - net_epoch_30_G.pth
    - testing
        - input_imgs
            - img001.jpg
            - img002.jpg
```

### **Generating Neutral Expression**
This configuration will only generate the neutral expression from a given input image.
```
python3 test.py \
--ckpts_dir "path to directory" \
--test_dir "path to directory" \
--input_dir "directory name" \
--expr_name "experiment name" \
--neutral_model "neutral model name" \
--use_18_aus 1 \
--test_mode 'neutral' \
--load_epoch -1 \
```
>`--ckpts_dir` - path to directory containing the saved models\
>`--test_dir` - path to test directory\
>`--input_dir` - path to directory containing input images\
>`--expr_name` - experiment name\
>`--neutral_model` - model name for neutral expression\
>`--use_18_aus` - 0 (17 AUs) and 1 (18 AUs)\
>`--test_mode` - mode to determine whether to generate neutral expression only, or with micro-expressions\
>`--load_epoch` - load model at specified epoch (default = -1)

### **Generating Micro-Expression**
This configuration will generate micro-expression from a given input image. The model will first generate the neutral expression using a pretrained neutral expression generator, and proceed to generate the micro-expressions. Please take note that you need to have the pretrained neutral-expression model downloaded as well.
```
python3 test.py \
--ckpts_dir "path to directory" \
--test_dir "path to directory" \
--input_dir "directory name" \
--expr_name "me experiment name" \
--test_mode 'me' \
--load_epoch -1 \
--neutral_expr_name "neutral experiment name" \
--samm_aus_dir "path to directory" \
--mmew_aus_dir "path to directory" \
--casme_ii_aus_dir "path to directory" \
--me_model "me model name" \
--me_summary "filename" \
--me_emotions "all" \
--me_samples -1 \
```
>`--ckpts_dir` - path to directory containing the saved models\
>`--test_dir` - path to test directory\
>`--input_dir` - path to directory containing input images\
>`--expr_name` - experiment name\
>`--test_mode` - mode to determine whether to generate neutral expression only, or with micro-expressions\
>`--load_epoch` - load model at specified epoch (default = -1)
>`--neutral_expr_name` - neutral experiment name\
>`--samm_aus_dir` - path to directory containing aus from SAMM database\
>`--mmew_aus_dir` - path to directory containing aus from MMEW database\
>`--casme_ii_aus_dir` - path to directory containing aus from CASME II database\
>`--me_model` - model name for micro-expression\
>`--me_summary` - JSON file containing the list of me emotions and corresponding files\
>`--me_emotions` - micro-expressions to generate (generate everything if set to 'all')\
>`--me_samples` - number of samples to generate for each micro-expression (generate all samples if set to -1; default = 10)