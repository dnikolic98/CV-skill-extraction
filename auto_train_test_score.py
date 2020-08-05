from preprocessor import Preprocessor
from inputExtractor import InputExtractor
from textFormater import TextFormater
import pandas as pd
import numpy as np
import pickle
from keras.models import model_from_json
from skillExtractNN import SkillsExtractorNN
import re

in_extractor = InputExtractor()
pp = Preprocessor()
tf = TextFormater()

word_features_dim, dense_features_dim = pp.getDim()

df = pd.read_excel("dataset/training/resumes.xlsx", sheet_name=0)
df = df.replace(np.nan, '', regex=True)

every_phrase_vec = []
every_context_vec = []
every_phr_cox_vec = []
every_y = []


for index, row in df.iterrows():
    try:
        if index == 119:
            break
        cv = tf.format(row[0])
        # print(index)
        # print(cv)
        # print()
        phrases, context, np_tags, context_tags = in_extractor.extract(cv)
        phr_vec, cox_vec, phr_cox_vec, y = pp.preprocess(
            phrases, context, np_tags, context_tags, row[1].strip().split("|"))
        every_phrase_vec += phr_vec
        every_context_vec += cox_vec
        every_phr_cox_vec += phr_cox_vec
        every_y += y
    except:
        # print("ERROR")
        continue

# with open('results/final/list_pickles/every_phrase_vec.pkl', 'wb') as f:
#     pickle.dump(every_phrase_vec, f)

# with open('results/final/list_pickles/every_context_vec.pkl', 'wb') as f:
#     pickle.dump(every_context_vec, f)

# with open('results/final/list_pickles/every_phr_cox_vec.pkl', 'wb') as f:
#     pickle.dump(every_phr_cox_vec, f)

# with open('results/final/list_pickles/every_y.pkl', 'wb') as f:
#     pickle.dump(every_y, f)


clf = SkillsExtractorNN(word_features_dim, dense_features_dim)

hist = clf.fit(np.array(every_phrase_vec), np.array(
    every_context_vec), np.array(every_phr_cox_vec), np.array(every_y))
acc = hist.history['accuracy'][-1]

# save model
model_json = clf.model.to_json()
# with open("results/final/model(" + str(acc).replace(".", "_") + ").json", "w") as json_file:
#     json_file.write(model_json)
# clf.model.save_weights(
#     "saved/final/model(" + str(acc).replace(".", "_") + ").h5")
# print("Saved model to disk")


accuracy_my = []
accuracy = []
precision = []
recall = []
f1 = []

for index, row in df.iterrows():
    try:
        if index < 119:
            continue
        cv = tf.format(row[0])
        phrases, context, np_tags, context_tags = in_extractor.extract(cv)
        phr_vec, cox_vec, phr_cox_vec = pp.preprocess(
            phrases, context, np_tags, context_tags)

        predicted = clf.predict(np.array(phr_vec), np.array(
            cox_vec), np.array(phr_cox_vec))

        y = row[1].strip().split("|")
        y = [(re.sub("\d+", "", elem)).strip() for elem in y]
        curr = []
        curr_neg = []
        correct_pos = 0
        false_pos = 0
        true_neg = 0
        false_neg = 0

        correct_pos_my = 0
        false_pos_my = 0
        true_neg_my = 0
        false_neg_my = 0

        print()
        print(index)
        for i in range(len(predicted)):
            if(np.argmax(predicted[i]) == 1):
                curr.append(phrases[i])
            else:
                curr_neg.append(phrases[i])
        print("SKILLS:", " ".join(y))
        print("PREDICTED POSITIVE:", " ".join(curr))
        print("PREDICTED NEGATIVE:", " ".join(curr_neg))

        for elem in curr:
            if elem in y:
                correct_pos += 1
            else:
                false_pos += 1

        for elem in curr_neg:
            if elem in y:
                false_neg += 1
            else:
                true_neg += 1

        for elem in set(curr):
            if elem in y:
                correct_pos_my += 1
            else:
                false_pos_my += 1

        for elem in set(curr_neg):
            if elem in y:
                false_neg_my += 1
            else:
                true_neg_my += 1

        print("---------------------------------")
        print("CORRECT POSITIVE:", correct_pos)
        print("FALSE POSITIVE:", false_pos)
        print("CORRECT NEGATIVE:", true_neg)
        print("FALSE NEGATIVE:", false_neg)
        print("---------------------------------")

        if len(set(y)) != 0:
            acc_my = correct_pos_my / len(set(y))
        else:
            acc_my = 0

        if (len(set(y)) + true_neg + false_pos) != 0:
            acc = (correct_pos_my + true_neg) / \
                (len(set(y)) + true_neg + false_pos)
        else:
            acc = 0

        if (correct_pos + false_neg) != 0:
            rec = correct_pos / (correct_pos + false_neg)
        else:
            rec = 0

        if (correct_pos + false_pos) != 0:
            prec = correct_pos / (correct_pos + false_pos)
        else:
            prec = 0

        if (prec + rec) != 0:
            f1_s = 2 * (prec * rec) / (prec + rec)
        else:
            f1_s = 0

        print("ACCURACY (moja bez true neg):", acc_my)
        print("ACCURACY:", acc)
        print("PRECISION:", prec)
        print("RECALL:", rec)
        print("F1:", f1_s)

        accuracy_my.append(acc_my)
        accuracy.append(acc)
        precision.append(prec)
        recall.append(rec)
        f1.append(f1_s)

    except:
        continue

print()

if len(accuracy_my) == 0:
    print("OVERALL ACC (od ukupno vještina koliko je izvukao/bez true negative): 0")
else:
    print("OVERALL ACC (od ukupno vještina koliko je izvukao/bez true negative):",
          sum(accuracy_my)/len(accuracy_my))

if len(accuracy) == 0:
    print("OVERALL ACC: 0")
else:
    print("OVERALL ACC:", sum(accuracy)/len(accuracy))

if len(precision) == 0:
    print("OVERALL PRECISION: 0")
else:
    print("OVERALL PRECISION:", sum(precision)/len(precision))

if len(recall) == 0:
    print("OVERALL RECALL: 0")
else:
    print("OVERALL RECALL:", sum(recall)/len(recall))

if len(f1) == 0:
    print("OVERALL F1: 0")
else:
    print("OVERALL F1:", sum(f1)/len(f1))


# with open('results/final/list_pickles_results/acc_my.pkl', 'wb') as f:
#     pickle.dump(accuracy_my, f)

# with open('results/final/list_pickles_results/acc.pkl', 'wb') as f:
#     pickle.dump(accuracy, f)

# with open('results/final/list_pickles_results/recall.pkl', 'wb') as f:
#     pickle.dump(precision, f)

# with open('results/final/list_pickles_results/precision.pkl', 'wb') as f:
#     pickle.dump(recall, f)

# with open('results/final/list_pickles_results/f1.pkl', 'wb') as f:
#     pickle.dump(f1, f)
