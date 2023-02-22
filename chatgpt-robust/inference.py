
from config import OPENAI_KEYS
import openai
openai.api_key = OPENAI_KEYS['api_key']


class Inference(object):

    def __init__(self,
                 task,
                 service,
                 label_set,
                 model_set,
                 label_to_id,
                 model=None,
                 device=0):  # service: hug, gpt, chat
        self.task = task
        self.service = service
        self.model = model
        self.label_set = label_set
        self.model_set = model_set
        self.label_to_id = label_to_id
        self.bot = None
        self.hug_classifier = None
        self.device = device
        if self.service.__contains__('hug'):
            from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
            if self.task.__contains__('translation'):
                self.hug_translator = pipeline(
                    self.task, model=model, max_length=256, device=device)
            if self.model in self.model_set['hug_zs']:
                self.hug_classifier = pipeline("zero-shot-classification",
                                               model=model,
                                               device=device)
            elif self.model in self.model_set['hug_gen']:
                if self.model.lower().__contains__('t0'):
                    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        model, device=device)
                    self.pipe = AutoModelForSeq2SeqLM.from_pretrained(model)
                    self.pipe = self.pipe.to(f"cuda:{device}")
                elif self.model.lower() in ['facebook/opt-66b']:
                    max_memory = {k: '80GB' for k in range(device+1)}
                    model = AutoModelForCausalLM.from_pretrained(
                        model, device_map="auto", load_in_8bit=True, max_memory=max_memory)
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        "facebook/opt-66b")
                    self.pipe = pipeline(
                        "text-generation", model=model, tokenizer=self.tokenizer, device_map='auto')
                # elif self.model.lower().__contains__('t5'):
                #     from transformers import T5Tokenizer, T5ForConditionalGeneration

                #     self.tokenizer = T5Tokenizer.from_pretrained(model, device=device)
                #     self.pipe = T5ForConditionalGeneration.from_pretrained(model)
                #     self.pipe = self.pipe.to(f"cuda:{device}")
                elif self.model in ['EleutherAI/gpt-j-6B', 'google/flan-t5-large', 'bigscience/bloom']:
                    pass
                elif self.model in ['BAAI/glm-10b']:
                    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
                    self.tokenizer = AutoTokenizer.from_pretrained("BAAI/glm-10b", trust_remote_code=True)
                    self.pipe = AutoModelForSeq2SeqLM.from_pretrained("BAAI/glm-10b", trust_remote_code=True)
                    self.pipe = self.pipe.half().cuda(self.device)
                    self.pipe.eval()
                else:
                    self.pipe = pipeline("text-generation",
                                         model=model,
                                         device=device)
        elif self.service == 'gpt':
            pass
        elif self.service.__contains__('chat'):
            from chatgpt_wrapper import ChatGPT
            self.bot = ChatGPT()

    def predict(self, sentence, prompt=None):
        if self.task.__contains__('translation'):
            return self.predict_trans(sentence, prompt)
        else:
            return self.predict_cls(sentence, prompt)

    def predict_cls(self, sentence, prompt=None):
        if self.service.__contains__('hug'):
            # use huggingface models
            if self.model in self.model_set['hug_gen']:
                # use generation models
                res = self.pred_by_generation(
                    prompt, self.model, sentence, self.label_set[self.task])
                if self.model in ['facebook/opt-66b', 'bigscience/T0pp']:
                    pred = res
                else:
                    pred = self.res_to_label(res)
            else:
                res = self.hug_classifier(sentence, self.label_set[self.task])
            # res = self.hug_classifier(sentence, candidate_labels=LABEL_SET[self.task])#他会把label set放进来，所以我在prompt里面不用写
                pred = self.res_to_label(res)
        elif self.service == 'gpt':
            # use gpt models
            if self.model == 'text-davinci-002':
                input = sentence+prompt
            else:
                input = prompt+sentence
            response = openai.Completion.create(
                engine=self.model,
                prompt=input,
                temperature=0,
                max_tokens=10,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
            pred = self.res_to_label(response)

        elif self.service.__contains__('chat'):
            # use chatgpt models
            response = self.bot.ask(prompt + sentence)
            pred = self.res_to_label(response)
        return pred

    def predict_trans(self, sentence, prompt=None):
        if self.service.__contains__('hug'):
            pred = self.hug_translator(sentence)[0]['translation_text']
        elif self.service.__contains__('gpt'):
            response = openai.Completion.create(
                model=self.model,
                prompt=prompt + f'"{sentence}"',
                temperature=0.3,
                max_tokens=256,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
            pred = response['choices'][0]['text'].lower().strip()
        elif self.service.__contains__('chat'):
            # use chatgpt models
            res = self.bot.ask(prompt + sentence)
            pred = res.lower().strip()
        return pred

    def res_to_label(self, res):
        # return "positive or negative" for generative models while 0-1 for others
        if self.model in self.model_set['hug_gen']:
            pred = res
        elif self.service.__contains__('hug') and not self.model in self.model_set['hug_gen']:
            pred = res['labels'][0]
            pred = self.label_to_id[self.task][pred.lower().strip()]
        elif self.service == 'gpt':
            pred = res['choices'][0]['text'].lower().strip().replace(
                ' ', '_').replace('\n', '_')
        elif self.service.__contains__('chat'):
            pred = res.lower().strip()
        return pred

    def pred_by_generation(self, prompt, model, sentence, label_set):
        def process_label(pred_label, label_set):
            for item in label_set:
                if item.lower() in pred_label.lower():
                    return item
            return pred_label
        out = 'error!'
        input_text = prompt + sentence + ' Answer: '
        if model.lower() in ['bigscience/bloomz-7b1', 'facebook/opt-66b']:
            inputs_ids = self.tokenizer(input_text)['input_ids']
            out = self.pipe(input_text, top_k=1,
                            max_length=len(inputs_ids) + 10)
            out = out[0]['generated_text']  # [{'generated_text': xxx}]
            out = out.split(':')[-1].strip()
        elif model.lower().__contains__('t5') or model.lower() == 'bigscience/bloom' or model.lower().__contains__('gpt-j-6b'):
            # using transformers api
            # input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(f'cuda:{self.device}')
            # outputs = model.generate(input_ids, max_length=20, early_stopping=True)
            # out = self.tokenizer.decode(outputs[0])

            # using huggingface api by default
            out = self.pred_by_inference_api(model, input_text)
        elif model.lower().__contains__('t0'):
            inputs = self.tokenizer.encode(
                input_text, return_tensors="pt").to(f'cuda:{self.device}')
            outputs = self.pipe.generate(inputs)
            out = self.tokenizer.decode(outputs[0])   # <pad> Positive</s>
        elif model.lower().__contains__('glm'):
            answer_mask = ' Answer: [MASK].'
            input_list = prompt + sentence + answer_mask
            inputs = self.tokenizer(input_list, return_tensors="pt")
            inputs = self.tokenizer.build_inputs_for_generation(inputs, max_gen_length=10)
            inputs = inputs.to('cuda')
            outputs = self.pipe.generate(**inputs, max_length=10, eos_token_id=self.tokenizer.eop_token_id)
            # print(outputs)
            out = self.tokenizer.decode(outputs[0].tolist())
            # out = out.split('_')[-1].strip()

        if model.lower() == 'facebook/opt-66b':
            out_processed = out.strip().lower().replace('\n', ',')
        else:
            out_processed = process_label(out, label_set)
        return out_processed

    def pred_by_inference_api(self, model, input_text):
        API_TOKEN = OPENAI_KEYS['api_token']
        import json
        import requests
        API_URL = f"https://api-inference.huggingface.co/models/{model}"
        headers = {"Authorization": f"Bearer {API_TOKEN}"}

        def query(payload):
            data = json.dumps(payload)
            response = requests.request(
                "POST", API_URL, headers=headers, data=data)
            return json.loads(response.content.decode("utf-8"))
        out = query(input_text)
        # print(out)
        out = out[0]['generated_text']
        return out

    def pred_by_mask(self, prompt, model, sentence):
        # this function supports roberta
        from transformers import pipeline
        if model in ['xlm-roberta-large']:
            pipe = pipeline(task="fill-mask", model=model)
            input_text = prompt + sentence + ' Answer: <mask>'
            out = pipe(input_text, top_k=5)
            out_list = [item['token_str'] for item in out]
            answer = 'error:' + '_'.join(out_list)
            for item in out_list:
                if item.lower() in ['positive', 'yes', 'true']:
                    answer = 'positive'
                    return answer
                elif item.lower() in ['negative', 'no', 'false']:
                    answer = 'negative'
                    return answer
            return answer


