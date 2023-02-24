
from config import LABEL_SET, PROMPT_SET, LABEL_TO_ID, DATA_PATH, MODEL_SET, MODEL_SET_TRANS
from dataload import Dataset
from inference import Inference
import argparse
import pandas as pd
from tqdm import tqdm
import numpy as np


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,
                        default='facebook/opt-66b')
    parser.add_argument(
        '--data_path', type=str, default='data/advglue/dev.json')
    parser.add_argument('--task', type=str, default='sst2')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_file', type=str, default='result/result.csv')
    parser.add_argument('--service', type=str, default='hug')
    parser.add_argument('--dataset', type=str, default='advglue')
    parser.add_argument('--eval', action='store_true', default=False)
    args = parser.parse_args()
    return args


def merge_res(args):
    df = pd.DataFrame()
    dataset = args.dataset
    task = args.task
    if task.__contains__('translation'):
        model_list = MODEL_SET_TRANS[args.service]
    else:
        model_list = MODEL_SET[args.service]
    for model in model_list:
        res = pd.read_csv('result/' + dataset + '_' + args.task +
                          '_' + args.service + '_' + model.replace('/', '_') + '.csv')
        df['idx'] = res['idx']
        df['content'] = res['content']
        df['true_label'] = res['true_label']
        df['pred-'+model.replace('/', '_')] = res['pred_label']
    df.to_csv(
        f'result/merge_{dataset}_{task}_{args.service}.csv', index=False)


def compute_metric(pred_label, true_label, task):
    if task.__contains__('translation'):
        import jieba
        import nltk.translate.bleu_score as bleu
        import nltk.translate.gleu_score as gleu
        import nltk.translate.meteor_score as meteor
        # import nltk
        # nltk.download('wordnet')
        # jieba.enable_paddle()

        ref_list = [[list(jieba.cut(item.strip(), use_paddle=True, cut_all=False))]
                    for item in true_label]
        hyp_list = [list(jieba.cut(item.strip(), use_paddle=True,
                         cut_all=False)) for item in pred_label]
        bleu_score = []
        for r, h in zip(ref_list, hyp_list):
            s = bleu.sentence_bleu(r, h)
            bleu_score.append(s)
        bleu_score = np.mean(bleu_score)
        gleu_score = []
        for r, h in zip(ref_list, hyp_list):
            s = gleu.sentence_gleu(r, h)
            gleu_score.append(s)
        gleu_score = np.mean(gleu_score)
        meteor_score = []
        for r, h in zip(ref_list, hyp_list):
            s = meteor.meteor_score(r, h)
            meteor_score.append(s)
        meteor_score = np.mean(meteor_score)
        return {'bleu': bleu_score * 100.0, 'gleu': gleu_score * 100.0, 'meteor_score': meteor_score * 100.0}
    else:
        return {'num_examples': len(pred_label), 'acc': np.mean(pred_label == true_label) * 100.0, 'asr': 100.0 - np.mean(pred_label == true_label) * 100.0}


def stat(args):
    df = pd.read_csv(
        f'result/merge_{args.dataset}_{args.task}_{args.service}.csv')
    labels = {}
    labels['true_label'] = df['true_label'].to_numpy()
    if args.task.__contains__('translation'):
        model_list = MODEL_SET_TRANS[args.service]
    else:
        model_list = MODEL_SET[args.service]
    for model in model_list:
        labels['pred-'+model.replace('/', '_')] = df['pred-' +
                                                     model.replace('/', '_')].to_numpy()
    for key in labels.keys():
        if key != 'true_label':
            if args.service in ['gpt', 'chat'] and args.dataset != 'advglue-t':
                pred_label = []
                for label in labels[key]:
                    orig = label
                    label = label.strip()
                    if "not_entail" in label or "not_ent" in label:
                        label = "not_entailment"
                    elif "entails" in label or "is_entailment" in label \
                        or "two sentences are entailment" in label or "entailment." in label \
                            or "entailment relation holds" in label or "\"entailment\"" in label:
                        label = "entailment"
                    elif "the two sentences are neutral" in label or "neutral." in label or "these two sentences are neutral to each other" in label:
                        label = "neutral"
                    elif "the two sentences are a contradiction" in label or "contradiction." in label \
                        or "the first two sentences are a contradiction" in label or "the two sentences are contradictory" in label:
                        label = "contradiction"
                    elif "two questions are equivalent" in label or "therefore equivalent" in label:
                        label = "equivalent"
                    elif "two questions are not equivalent" in label or "not equivalent." in label or "they are not exactly equivalent" in label:
                        label = "not_equivalent"
                    elif "the sentence is negative" in label or "the sentence as negative" in label or "the answer is \"negative\"" in label:
                        label = "negative"
                    elif "the second sentence is not entailment" in label or "do not entail each other" in label \
                        or "the question and the sentence are not entailed" in label or "the given question and sentence are not related" in label:
                        label = "not_entailment"
                    elif "the classification would be \"positive\"." in label or "the answer would be \"positive\"." in label:
                        label = "positive"
                    if '_' in label:
                        label = label.split('_')[-1] if label not in LABEL_TO_ID[args.task] else label
                    if '.' in label:
                        label = label.strip('.')
                        
                    try:
                        pred_label.append(LABEL_TO_ID[args.task][label])
                    except:
                        print(orig)
                        pred_label.append(-1)
                pred_label = np.array(pred_label)

                if args.dataset == 'flipkart':
                    true_label = []
                    for label in labels['true_label']:
                        true_label.append(LABEL_TO_ID[args.task][label])
                    true_label = np.array(true_label)
                    # acc = np.mean(pred_label == true_label)
                    metric_dict = compute_metric(
                        pred_label, true_label, args.task)
                else:
                    # acc = np.mean(pred_label == labels['true_label'])
                    metric_dict = compute_metric(
                        pred_label, labels['true_label'], args.task)
            elif args.dataset == 'flipkart':
                true_label = []
                for label in labels['true_label']:
                    true_label.append(LABEL_TO_ID['sst2'][label])
                true_label = np.array(true_label)
                # acc = np.mean(labels[key] == true_label)
                metric_dict = compute_metric(
                    labels[key], true_label, args.task)
            elif args.dataset == 'advglue':
                pred_label = []
                for label in labels[key]:
                    pred_label.append(LABEL_TO_ID[args.task][label])
                pred_label = np.array(pred_label)
                # acc = np.mean(labels[key] == true_label)
                metric_dict = compute_metric(
                    pred_label, labels['true_label'], args.task)
            elif args.dataset == 'anli':
                true_label = []
                map_dict = {'e': 'entailment',
                            'c': 'contradiction', 'n': 'neutral'}
                for label in labels['true_label']:
                    true_label.append(map_dict[label])
                true_label = np.array(true_label)
                metric_dict = compute_metric(
                    labels[key], true_label, args.task)
            elif args.dataset == 'ddxplus':
                true_label = []
                for label in labels['true_label']:
                    true_label.append(label.lower())
                true_label = np.array(true_label)
                metric_dict = compute_metric(
                    labels[key], true_label, args.task)
            else:
                # acc = np.mean(labels[key] == labels['true_label'])
                metric_dict = compute_metric(
                    labels[key], labels['true_label'], args.task)

            metric_string = ', '.join(
                ['{:s}:{:.2f}'.format(k, v) for k, v in metric_dict.items()])
            print("{:s} - {:s}".format(key, metric_string))


def run(args):
    data = Dataset(args.dataset, DATA_PATH[args.dataset], args.task).dataclass
    infer = Inference(args.task, args.service, LABEL_SET, MODEL_SET, LABEL_TO_ID, args.model, args.gpu)
    data_len = len(data.get_data_by_task(args.task))
    args.save_file = 'result/' + args.dataset + '_' + args.task + \
        '_' + args.service + '_' + args.model.replace('/', '_') + '.csv'
    lst = []
    for idx in tqdm(range(data_len)):
        res_dict = {}
        content, label = data.get_content_by_idx(idx, args.task)
        pred_label = infer.predict(
            content, prompt=PROMPT_SET[args.task][-1])
        res_dict['idx'] = idx
        res_dict['content'] = content
        res_dict['true_label'] = label
        res_dict['pred_label'] = pred_label
        lst.append(res_dict)
        pd.DataFrame(lst).to_csv(args.save_file, index=False)
        # Note that if you are using OpenAI api, you can only send 1000 requests per day.


if __name__ == '__main__':
    args = get_args()
    if not args.eval:
        run(args)
    else:
        merge_res(args)
        stat(args)
