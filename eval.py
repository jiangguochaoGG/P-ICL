import json

from argparse import ArgumentParser

def NER_eval(label_path, pred_path):
    with open(pred_path, 'r') as f:
        raw_pred = json.load(f)
    with open(label_path, 'r') as f:
        raw_label = json.load(f)
    preds, labels = {}, {}

    for js in raw_pred:
        key = js['text']
        value = []
        try:
            output = js['result']
            output = output[output.index('{'):output.index('}')+1]
            output = json.loads(output)
            for type, entities in output.items():
                if entities:
                    for entity in entities:
                        value.append((type, entity))
        except:
            pass
    for js in raw_label:
        key = js['text']
        value = []
        for type, entities in js['label'].items():
            if entities:
                for entity in entities:
                    value.append((type, entity))
        labels[key] = value

    nb_true, nb_pred, nb_label = 0, 0, 0
    for text, pred in preds.items():
        label = labels[text]
        nb_pred += len(set(pred))
        nb_label += len(set(label))
        nb_true += len(set(pred) & set(label))
    
    precision = nb_true / nb_pred
    recall = nb_true / nb_label
    f1 = 2 * precision * recall / (precision + recall)    

    return f1

parser = ArgumentParser()
parser.add_argument('--label_path', type=str)
parser.add_argument('--pred_path', type=str)
args = parser.parse_args()

print(NER_eval(
    args.label_path, args.pred_path
))