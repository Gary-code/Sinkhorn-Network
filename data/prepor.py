import csv
import json
import os

"""
split train and test set
"""

def main():
    out = []
    outtest = []
    outval = []
    with open('./datasets/sentence_pairs.json', 'r', encoding='utf-8') as fd:
        tsvin = json.load(fd)
        count1 = 0
        row_count = 208999
        for row in tsvin:
            count1 = count1 + 1
            if 1 < count1 < int(0.75 * row_count) + 2:  # taking the starting 1 lakh pairs as train set. Change this to 50002 for taking staring 50 k examples as train set
                # get the question and unique id from the tsv file
                out.append(row)
            elif int(0.75 * row_count) + 1 < count1 < int(0.75 * row_count) + 2 + int(0.22 * row_count):  # next 30k as the test set acc to https://arxiv.org/pdf/1711.00279.pdf
                outtest.append(row)
            else:  # rest as val
                outval.append(row)

    # write the json files for train test and val
    print(len(out))
    json.dump(out, open('./datasets/sentences_train.json', 'w'))
    print(len(outtest))
    json.dump(outtest, open('./datasets/sentences_test.json', 'w'))
    print(len(outval))
    json.dump(outval, open('./datasets/sentences_val.json', 'w'))


if __name__ == "__main__":
    main()