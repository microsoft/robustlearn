for task in "sst2" "mnli" "qqp" "qnli" "rte"
do
    echo $task
    python main.py --dataset advglue --task $task --service chat --eval
done

echo "flipkart"
python main.py --dataset flipkart --task sst2 --service chat --eval
