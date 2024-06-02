echo "EVALUATION"
echo "=========="
echo "1. Extract AUs"
echo "2. Pearson Correlation"
echo "3. AUs Comparison"
echo "4. Optical Flow Analysis"
echo "5. Prepare S3S-CNN data"

echo -n "Enter your choice: "

read choice

if [ $choice -eq 1 ]
then
    # EXTRACT AUS
    python3 evaluation/extract_aus.py \
    --root_dir "path to testing folder" \
    --expr "expr name" \
    --subject "all" \

elif [ $choice -eq 2 ]
then
    # PEARSON CORRELATION
    python3 evaluation/pearson_correlation.py \
    --root_dir "path to testing folder" \
    --expr "expr name" \
    --subject "all" \
    --samm_aus "path to SAMM extracted aus" \
    --mmew_aus "path to MMEW extracted aus" \
    --casme_ii_aus "path to CASME II extracted aus" \
    --use_zero 0 \

    python3 evaluation/pearson_correlation.py \
    --root_dir "path to testing folder" \
    --expr "expr name" \
    --subject "all" \
    --samm_aus "path to SAMM extracted aus" \
    --mmew_aus "path to MMEW extracted aus" \
    --casme_ii_aus "path to CASME II extracted aus" \
    --use_zero 1 \

elif [ $choice -eq 3 ] 
then
    # AUS COMPARISON
    python3 evaluation/aus_comparison.py \
    --root_dir "path to testing folder" \
    --expr "expr name" \
    --subject "Arnold_Schwarzenegger_0013" \
    --samm_aus "path to SAMM extracted aus" \
    --mmew_aus "path to MMEW extracted aus" \
    --casme_ii_aus "path to CASME II extracted aus" \

elif [ $choice -eq 4 ]
then
    # OPTICAL FLOW ANALYSIS
    python3 evaluation/optical_flow_analysis.py \
    --root_dir "path to testing folder" \
    --expr "expr name" \
    --subject "Mary_Bono_0001" \
    --samm_dir "path to SAMM dataset" \
    --mmew_dir "path to MMEW dataset" \
    --casme_ii_dir "path to CASME II dataset" \

elif [ $choice -eq 5 ]
then
    # PREPARE DATA FOR S3S-CNN
    python3 evaluation/s3s-cnn/preparation.py \
    --expr "expr name" \
    --data_dir "path to testing folder" \
    --out_dir "path to S3S-CNN folder" \
    --n_samples 100 \

fi