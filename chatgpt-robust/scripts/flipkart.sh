
for model in "xlm-roberta-large" "facebook/bart-large" "bert-large-uncased" "google/electra-large-discriminator" "microsoft/deberta-v3-large" "gpt2" "bigscience/bloomz-7b1" "facebook/opt-66b"
do
    echo $model
    python infer.py --task flipkart --model $model --service hug --dataset flipkart
done