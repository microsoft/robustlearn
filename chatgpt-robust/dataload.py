from config import LABEL_SET, PROMPT_SET, OPENAI_KEYS
import json
import pandas as pd
import openai
openai.api_key = OPENAI_KEYS['api_key']

class DataANLI(object):

    def __init__(self, data_path, task):
        self.task = task
        with open(data_path, 'r') as f:
            self.data = f.readlines()

    def get_data_by_task(self, task):
        self.data_task = self.data#[task]
        return self.data_task

    def get_content_by_idx(self, idx, task=None):
        if task is None:
            task = self.task
        self.data_task = self.get_data_by_task(task)
        data= json.loads(self.data_task[idx])
        content = data['context'] + \
                ' ' + data['hypothesis']
        label = data['label']
        return content, label

    def get_prompt(self):
        return PROMPT_SET[self.task]

    def get_label(self):
        return LABEL_SET[self.task]

class DataDDXPlus(object):

    def __init__(self, data_path, task):
        self.task = task
        self.data = pd.read_csv(data_path)

    def get_data_by_task(self, task):
        self.data_task = self.data
        return self.data_task

    def get_content_by_idx(self, idx, task=None):
        if task is None:
            task = self.task
        self.data_task = self.get_data_by_task(task)
        content = self.data_task.iloc[idx]['Information']
        label = self.data_task.iloc[idx]['Diagnosis']
        return content, label

    def get_prompt(self):
        return PROMPT_SET[self.task]

    def get_label(self):
        return LABEL_SET[self.task]


class DataAdvGLUE(object):

    def __init__(self, data_path, task):
        self.task = task
        self.data = json.load(open(data_path, 'r'))

    def get_data_by_task(self, task):
        self.data_task = self.data[task]
        return self.data_task

    def get_content_by_idx(self, idx, task=None):
        if task is None:
            task = self.task
        self.data_task = self.get_data_by_task(task)
        if task == 'sst2':
            content = self.data_task[idx]['sentence']
        elif task == 'qqp':
            content = self.data_task[idx]['question1'] + \
                ' ' + self.data_task[idx]['question2']
        elif task == 'mnli':
            content = self.data_task[idx]['premise'] + \
                ' ' + self.data_task[idx]['hypothesis']
        elif task == 'qnli':
            content = self.data_task[idx]['question'] + \
                ' ' + self.data_task[idx]['sentence']
        elif task == 'rte':
            content = self.data_task[idx]['sentence1'] + \
                ' ' + self.data_task[idx]['sentence2']
        elif task == 'mnli-mm':
            content = self.data_task[idx]['premise'] + \
                ' ' + self.data_task[idx]['hypothesis']
        label = self.data_task[idx]['label']
        return content, label

    def get_prompt(self):
        return PROMPT_SET[self.task]

    def get_label(self):
        return LABEL_SET[self.task]


class DataFlipkart(object):

    # note that task is fixed to sst2 for flipkart since it is sentiment analysis
    def __init__(self, data_path, task='sst2') -> None:
        self.data = pd.read_csv(data_path)
        self.task = task

    def get_prompt(self):
        return PROMPT_SET[self.task]

    def get_label(self):
        return LABEL_SET[self.task]

    def get_data_by_task(self, task):
        return self.data

    def get_content_by_idx(self, idx, task=None):
        if task is None:
            task = self.task
        content = self.data.iloc[idx]['Summary']
        label = self.data.iloc[idx]['Sentiment']
        return content, label


class DataGLUETranslation(object):

    def __init__(self, data_path, task='translation_en_zh') -> None:

        # self.data = pd.read_csv(data_path)
        import json
        with open(data_path, 'r') as f:
            data = json.load(f)[task]
        self.data = dict()
        translation = task[:-4]
        for d in data:
            self.data[d['idx']] = {'source':d['source'], 'target':d['target']}
        self.task = task

    def get_prompt(self):
        return PROMPT_SET[self.task]

    def get_content_by_idx(self, idx, task=None):
        if task is None:
            task = self.task
        content = self.data[idx]['source']
        label = self.data[idx]['target']
        return content, label

    def get_data_by_task(self, task):
        return self.data


class Dataset(object):

    def __init__(self, data_name, data_path, task) -> None:  # datatype: advglue, flipkart
        self.data = data_path
        self.task = task
        self.name = data_name
        self.dataclass = self.get_dataclass()

    def get_dataclass(self):
        if self.name == 'advglue':
            return DataAdvGLUE(self.data, self.task)
        elif self.name == 'flipkart':
            return DataFlipkart(self.data, self.task)
        elif self.name == 'ddxplus':
            return DataDDXPlus(self.data, self.task)
        elif self.name == 'advglue-t':
            return DataGLUETranslation(self.data, self.task)
        elif self.name == 'anli':
            return DataANLI(self.data, self.task)
        else:
            raise NotImplementedError
