for model in "text-davinci-002"
do
    for task in "sst2" "mnli" "qqp" "qnli" "rte"
    do
        echo $model $task
        python main.py --dataset advglue --task $task --model $model --service gpt --eval
    done
    echo $model "flipkart"
    python main.py --dataset flipkart --task sst2 --model $model --service gpt --eval
done
