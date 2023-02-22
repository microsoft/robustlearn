LABEL_SET = {
    'sst2': ['positive', 'negative'],
    'mnli': ['entailment', 'neutral', 'contradiction'],
    'anli': ['entailment', 'neutral', 'contradiction'],
    'qqp': ['equivalent', 'not_equivalent'],
    'qnli': ['entailment', 'not_entailment'],
    'mnli-mm': ['entailment', 'neutral', 'contradiction'],
    'rte': ['entailment', 'not_entailment'],
    'ddxplus': ['spontaneous pneumothorax', 'cluster headache', 'boerhaave', 'spontaneous rib fracture', 'gerd', 'hiv (initial infection)', 'anemia', 'viral pharyngitis', 'inguinal hernia', 'myasthenia gravis', 'whooping cough', 'anaphylaxis', 'epiglottitis', 'guillain-barré syndrome', 'acute laryngitis', 'croup', 'psvt', 'atrial fibrillation', 'bronchiectasis', 'allergic sinusitis', 'chagas', 'scombroid food poisoning', 'myocarditis', 'larygospasm', 'acute dystonic reactions', 'localized edema', 'sle', 'tuberculosis', 'unstable angina', 'stable angina', 'ebola', 'acute otitis media', 'panic attack', 'bronchospasm / acute asthma exacerbation', 'bronchitis', 'acute copd exacerbation / infection', 'pulmonary embolism', 'urti', 'influenza', 'pneumonia', 'acute rhinosinusitis', 'chronic rhinosinusitis', 'bronchiolitis', 'pulmonary neoplasm', 'possible nstemi / stemi', 'sarcoidosis', 'pancreatic neoplasm', 'acute pulmonary edema', 'pericarditis', 'cannot decide'],
    'flipkart': ['positive', 'negative', 'neutral'],

}

MODEL_SET = {
    'hug_zs': [ # zero-shot classification using fine-tuned models
        'cross-encoder/nli-deberta-v3-large',
        # 'sentence-transformers/nli-roberta-large',
        'facebook/bart-large-mnli'
    ], 
    'hug_gen': [   # generative big models using text-generation
        'google/flan-t5-large',
        'facebook/opt-66b',
        # 'bigscience/bloomz-7b1',
        # 'bigscience/T0pp',
        'bigscience/bloom',
        'EleutherAI/gpt-j-6B',
        'EleutherAI/gpt-neox-20b',
        # 'BAAI/glm-10b'

    ],
    'gpt': [
        # 'text-ada-001',
        'text-davinci-002',
        'text-davinci-003',
    ],
    'chat': ['bert-large-uncased']
}

PROMPT_SET = {
    'sst2': [
        'Is the following sentence positive or negative? Answer me with "positive" or "negative", just one word. ',
        'Please classify the following sentence into either positive or negative. Answer me with "positive" or "negative", just one word. ',
    ],
    'qqp': [
        'Are the following two questions equivalent or not? Answer me with "equivalent" or "not_equivalent". ',
    ],
    'mnli': [
        'Are the following two sentences entailment, neutral or contradiction? Answer me with "entailment", "neutral" or "contradiction". ',
    ],
    'anli': [
        'Are the following paragraph entailment, neutral or contradiction? Answer me with "entailment", "neutral" or "contradiction". The answer should be a single word. The answer is: ',
    ],
    'qnli': [
        'Are the following question and sentence entailment or not_entailment? Answer me with "entailment" or "not_entailment". ',
    ],
    'mnli-mm': [
        'Are the following two sentences entailment, neutral or contradiction? Answer me with "entailment", "neutral" or "contradiction". ',
    ],
    'flipkart': [
        'Is the following sentence positive, neutral, or negative? Answer me with "positive", "neutral", or "negative", just one word. ',
    ],
    'rte': [
        'Are the following two sentences entailment or not_entailment? Answer me with "entailment" or "not_entailment". ',
    ],
    'ddxplus': [
        "Imagine you are an intern doctor. Based on the previous dialogue, what is the diagnosis? Select one answer among the diseases: spontaneous pneumothorax, cluster headache, boerhaave, spontaneous rib fracture, gerd, hiv (initial infection), anemia, viral pharyngitis, inguinal hernia, myasthenia gravis, whooping cough, anaphylaxis, epiglottitis, guillain-barré syndrome, acute laryngitis, croup, psvt, atrial fibrillation, bronchiectasis, allergic sinusitis, chagas, scombroid food poisoning, myocarditis, larygospasm, acute dystonic reactions, localized edema, sle, tuberculosis, unstable angina, stable angina, ebola, acute otitis media, panic attack, bronchospasm / acute asthma exacerbation, bronchitis, acute copd exacerbation / infection, pulmonary embolism, urti, influenza, pneumonia, acute rhinosinusitis, chronic rhinosinusitis, bronchiolitis, pulmonary neoplasm, possible nstemi / stemi, sarcoidosis, pancreatic neoplasm, acute pulmonary edema, pericarditis. The answer should be one disease. The answer is:",
    ],
    'translation_en_to_zh': [
        'Translate the following sentence from Engilish to Chinese. '
    ],
    'translation_zh_to_en': [
        'Translate the following sentence from Chinese to English. '
    ]
}

LABEL_TO_ID = {
    'sst2': {'negative': 0, 'positive': 1, 'neutral': 2},
    'mnli': {'entailment': 0, 'neutral': 1, 'contradiction': 2},
    'qqp': {'equivalent': 1, 'not_equivalent': 0},
    'qnli': {'entailment': 0, 'not_entailment': 1},
    'flipkart': {'negative': 0, 'positive': 1, 'neutral': 2},
    # 'mnli-mm': {'entailment': 0, 'neutral': 1, 'contradiction': 2},
    'rte': {'entailment': 0, 'not_entailment': 1},
    'ddxplus': {'spontaneous pneumothorax': 0, 'cluster headache': 1, 'boerhaave': 2, 'spontaneous rib fracture': 3, 'gerd': 4, 'hiv (initial infection)': 5, 'anemia': 6, 'viral pharyngitis': 7, 'inguinal hernia': 8, 'myasthenia gravis': 9, 'whooping cough': 10, 'anaphylaxis': 11, 'epiglottitis': 12, 'guillain-barré syndrome': 13, 'acute laryngitis': 14, 'croup': 15, 'psvt': 16, 'atrial fibrillation': 17, 'bronchiectasis': 18, 'allergic sinusitis': 19, 'chagas': 20, 'scombroid food poisoning': 21, 'myocarditis': 22, 'larygospasm': 23, 'acute dystonic reactions': 24, 'localized edema': 25, 'sle': 26, 'tuberculosis': 27, 'unstable angina': 28, 'stable angina': 29, 'ebola': 30, 'acute otitis media': 31, 'panic attack': 32, 'bronchospasm / acute asthma exacerbation': 33, 'bronchitis': 34, 'acute copd exacerbation / infection': 35, 'pulmonary embolism': 36, 'urti': 37, 'influenza': 38, 'pneumonia': 39, 'acute rhinosinusitis': 40, 'chronic rhinosinusitis': 41, 'bronchiolitis': 42, 'pulmonary neoplasm': 43, 'possible nstemi / stemi': 44, 'sarcoidosis': 45, 'pancreatic neoplasm': 46, 'acute pulmonary edema': 47, 'pericarditis': 48, 'cannot decide': 49},
    'anli': {'entailment': 0, 'neutral': 1, 'contradiction': 2},

}

ID_TO_LABEL = {
    'sst2': {0: 'negative', 1: 'positive', 2: 'neutral'},
    'mnli': {0: 'entailment', 1: 'neutral', 2: 'contradiction'},
    'qqp': {1: 'equivalent', 0: 'not_equivalent'},
    'qnli': {0: 'entailment', 1: 'not_entailment'},
    'flipkart': {0: 'negative', 1: 'positive', 2: 'neutral'},
    # 'mnli-mm': {0: 'entailment', 1: 'neutral', 2: 'contradiction'},
    'rte': {0: 'entailment', 1: 'not_entailment'},
    'ddxplus': {0: 'spontaneous pneumothorax', 1: 'cluster headache', 2: 'boerhaave', 3: 'spontaneous rib fracture', 4: 'gerd', 5: 'hiv (initial infection)', 6: 'anemia', 7: 'viral pharyngitis', 8: 'inguinal hernia', 9: 'myasthenia gravis', 10: 'whooping cough', 11: 'anaphylaxis', 12: 'epiglottitis', 13: 'guillain-barré syndrome', 14: 'acute laryngitis', 15: 'croup', 16: 'psvt', 17: 'atrial fibrillation', 18: 'bronchiectasis', 19: 'allergic sinusitis', 20: 'chagas', 21: 'scombroid food poisoning', 22: 'myocarditis', 23: 'larygospasm', 24: 'acute dystonic reactions', 25: 'localized edema', 26: 'sle', 27: 'tuberculosis', 28: 'unstable angina', 29: 'stable angina', 30: 'ebola', 31: 'acute otitis media', 32: 'panic attack', 33: 'bronchospasm / acute asthma exacerbation', 34: 'bronchitis', 35: 'acute copd exacerbation / infection', 36: 'pulmonary embolism', 37: 'urti', 38: 'influenza', 39: 'pneumonia', 40: 'acute rhinosinusitis', 41: 'chronic rhinosinusitis', 42: 'bronchiolitis', 43: 'pulmonary neoplasm', 44: 'possible nstemi / stemi', 45: 'sarcoidosis', 46: 'pancreatic neoplasm', 47: 'acute pulmonary edema', 48: 'pericarditis', 49: 'cannot decide'},
}

DATA_PATH = {
    'advglue': './data/advglue/dev.json',
    'flipkart': './data/flipkart/flipkart_review.csv',
    'ddxplus': './data/ddxplus/ddxplus.csv',
    'anli': './data/anli/test.jsonl',
    'advglue-t': './data/advglue-t/translation.json',
}


MODEL_SET_TRANS = {
    'gpt': [
        # 'text-ada-001',
        'text-davinci-002',
        'Helsinki-NLP/opus-mt-en-zh',
        'liam168/trans-opus-mt-en-zh',
        'text-davinci-003',
    ],
    'chatgpt': [
        'chatgpt'
    ]
}

OPENAI_KEYS = {
    'api_key': "xxxxxxx",
    'api_token': "xxxxxxx"
}

PROMPT_SET2 = {
    'sst2': [
        'Is the following sentence positive or negative? Answer me with "positive" or "negative", just one word. ',
        'Please classify the following sentence into either positive or negative. If it is positive, reply 1, otherwise, reply 0. Just answer me with a single number. ',
    ],
    'qqp': [
        'Are the following two questions equivalent or not? If they are equivalent, answer me with 1, otherwise, answer me with 0. Just answer me with a single number. ',
    ],
    'mnli': [
        'Are the following two sentences entailment, neutral or contradiction? Answer me with 0 if they are entailment, answer me with 2 if they are contradiction, otherwise answer me with 1. ',
    ],
    'anli': [
        'Are the following two sentences entailment, neutral or contradiction? Answer me with 1 if they are entailment, answer me with 0 if they are contradiction, otherwise answer me with 2. ',
    ],
    'qnli': [
        'Are the following question and sentence entailment or not entailment? Answer me with 0 if they are entailment, otherwise 1. ',
    ],
    'mnli-mm': [
        'Are the following two sentences entailment, neutral or contradiction? Answer me with "entailment", "neutral" or "contradiction". ',
    ],
    'flipkart': [
        'Is the following sentence positive, neutral, or negative? Answer me with "positive", "neutral", or "negative", just one word. ',
    ],
    'rte': [
        'Are the following two sentences entailment or not? Answer me with 0 if they are entailment, otherwise answer me with 1. ',
    ],
    'ddxplus': [
        "Imagine you are an intern doctor. Based on the previous dialogue, what is the diagnosis? Select one answer among the following lists: ['spontaneous pneumothorax', 'cluster headache', 'boerhaave', 'spontaneous rib fracture', 'gerd', 'hiv (initial infection)', 'anemia', 'viral pharyngitis', 'inguinal hernia', 'myasthenia gravis', 'whooping cough', 'anaphylaxis', 'epiglottitis', 'guillain-barré syndrome', 'acute laryngitis', 'croup', 'psvt', 'atrial fibrillation', 'bronchiectasis', 'allergic sinusitis', 'chagas', 'scombroid food poisoning', 'myocarditis', 'larygospasm', 'acute dystonic reactions', 'localized edema', 'sle', 'tuberculosis', 'unstable angina', 'stable angina', 'ebola', 'acute otitis media', 'panic attack', 'bronchospasm / acute asthma exacerbation', 'bronchitis', 'acute copd exacerbation / infection', 'pulmonary embolism', 'urti', 'influenza', 'pneumonia', 'acute rhinosinusitis', 'chronic rhinosinusitis', 'bronchiolitis', 'pulmonary neoplasm', 'possible nstemi / stemi', 'sarcoidosis', 'pancreatic neoplasm', 'acute pulmonary edema', 'pericarditis', 'cannot decide']. The answer should be a single word. The answer is: "
    ],
    'translation_en_to_zh': [
        'Translate the following sentence from Engilish to Chinese. '
    ],
    'translation_zh_to_en': [
        'Translate the following sentence from Chinese to English. '
    ]
}