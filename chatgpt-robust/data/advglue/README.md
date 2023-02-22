# Adversarial GLUE dataset

You need to download the dev set from the website of AdvGLUE:https://adversarialglue.github.io/.

Then, put it in this folder.

The following content is copied from AdvGLUE webpage.

## Data Format

File ```dev.json``` contains the dev data of AdvGLUE dataset. Each task forms a key-value pair inside the json object. The structure of the file should look like:

```
{
  "sst2": sst2_item_list,
  "qqp": qqp_item_list,
  "mnli": mnli_item_list,
  "mnli-mm": mnli-mm_item_list,
  "qnli": qnli_item_list,
  "rte": rte_item_list
}
```

Items in different tasks have different formats. The format of each task is listed below:

  - **SST-2:** ```{'idx': index, 'label': label, 'sentence': text}```
  - **QQP:** ```{'idx': index, 'label': label, 'question1': text, 'question2': text}```
  - **MNLI:** ```{'idx': index, 'label': label, 'premise': text, 'hypothesis': text}```
  - **QNLI:** ```{'idx': index, 'label': label, 'question': text, 'sentence': text}```
  - **RTE:** ```{'idx': index, 'label': label, 'sentence1': text, 'sentence2': text}```
