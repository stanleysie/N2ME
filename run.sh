echo "N2ME"
echo "=========="
echo "1. Training Neutral Expression Generator"
echo "2. Training Micro-Expression Generator"
echo "3. Testing Neutral Expressio Generator"
echo "4. Testing Micro-Expression Generator"

echo -n "Enter your choice: "

read choice

if [ $choice -eq 1 ]
then
    python3 train.py \
    --data_dir "path to processed CelebA images" \
    --imgs_dir "imgs" \
    --ckpts_dir "checkpoints for neutral model" \
    --samples_dir "samples" \
    --train_ids_file "train_ids.txt" \
    --test_ids_file "test_ids.txt" \
    --aus_file "aus_18.pkl" \
    --expr_name "expr name" \
    --model_name "ganimation_modified" \
    --train_mode "neutral" \
    --use_18_aus 1 \
    --load_epoch -1 \
    --end_epoch -1 \
    --batch_size 4 \
    --show_params 1 \
    --use_tensorboard 1 \
    --num_epochs 30 \
    --num_epochs_decay 20 \

elif [ $choice -eq 2 ]
then
    python3 train.py \
    --data_dir "path to processed CelebA images" \
    --imgs_dir "imgs" \
    --ckpts_dir "checkpoints for me model" \
    --samples_dir "samples" \
    --train_ids_file "train_ids.txt" \
    --test_ids_file "test_ids.txt" \
    --aus_file "aus.pkl" \
    --expr_name "expr name" \
    --model_name "ganimation_modified" \
    --train_mode "me" \
    --use_18_aus 0 \
    --load_epoch -1 \
    --end_epoch -1 \
    --batch_size 25 \
    --show_params 1 \
    --use_tensorboard 1 \
    --num_epochs 30 \
    --num_epochs_decay 20 \

elif [ $choice -eq 3 ]
then
    # TESTING NEUTRAL
    python3 test.py \
    --ckpts_dir "checkpoints for neutral model" \
    --test_dir "testing dir for neutral" \
    --input_dir "input images folder" \
    --expr_name "expr name" \
    --neutral_model "ganimation" \
    --use_18_aus 1 \
    --test_mode 'neutral' \
    --load_epoch -1 \

elif [ $choice -eq 4 ]
then
    # TESTING ME
    python3 test.py \
    --ckpts_dir "checkpoints for me model" \
    --test_dir "testing dir for me" \
    --input_dir "input images folder" \
    --expr_name "expr name" \
    --test_mode 'me' \
    --load_epoch -1 \
    --neutral_expr_name "expr name" \
    --neutral_model "ganimation" \
    --use_18_aus 1 \
    --samm_aus_dir "path to SAMM extracted aus" \
    --mmew_aus_dir "path to MMEW extracted aus" \
    --casme_ii_aus_dir "path to CASME II extracted aus" \
    --me_model "ganimation_modified" \
    --me_summary "me_summary.json" \
    --me_emotions "all" \
    --me_samples -1 \

fi