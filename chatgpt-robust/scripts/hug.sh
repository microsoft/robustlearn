# sweep over all models for sst2
for model in "cardiffnlp/twitter-roberta-base-sentiment" "cardiffnlp/twitter-roberta-base-sentiment-latest" "Seethal/sentiment_analysis_generic_dataset"
do
    echo $model $task
    python infer.py --dataset advglue --task sst2 --model $model --service hug
done

# sweep over all models for mnli
# for model in "facebook/bart-large-mnli" "microsoft/deberta-xlarge-mnli" "roberta-large-mnli"
# do
#     echo $model $task
#     python infer.py --dataset advglue --task mnli --model $model --service hug
# done

# sweep over all models for qqp 
# for model in  "textattack/bert-base-uncased-QQP" "yoshitomo-matsubara/bert-base-uncased-qqp" "gchhablani/bert-base-cased-finetuned-qqp"
# do
#     echo $model $task
#     python infer.py --dataset advglue --task qqp --model $model --service hug
# done

# sweep over all models for qnli
# for model in "cross-encoder/qnli-electra-base" "ModelTC/bart-base-qnli" "textattack/roberta-base-QNLI"
# do
#     echo $model $task
#     python infer.py --dataset advglue --task qnli --model $model --service hug
# done

# sweep over all models for rte
# for model in "textattack/roberta-base-RTE" "yoshitomo-matsubara/bert-large-uncased-rte" "yoshitomo-matsubara/bert-base-uncased-rte"
# do
#     echo $model $task
#     python infer.py --dataset advglue --task rte --model $model --service hug
# done
