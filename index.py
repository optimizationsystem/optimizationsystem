"""
"""
import csv
import numpy as np


if __name__ == '__main__':
    # create a mapping from input to 4 values of output
    # notice:
    predicted_data_path = 'indexing.csv'
    indexing = dict()
    with open(predicted_data_path, 'rt') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            row = list(map(float, row))
            row[-1] = round(row[-1], 2)
            if row[-1] not in indexing.keys():
                indexing[row[-1]] = []

            indexing[row[-1]].append(row[:-1])

    input_ = round(float(input('Enter your input: ')), 2)

    keys = np.asarray(list(indexing.keys()))
    idx = np.argmin(np.abs(keys - input_))
    min_key = round(float(keys[idx]), 2)

    print('Filter 1')
    for value in indexing[min_key]:
        print('Input: %.2f -> Output: %.2f %.2f %.2f %.2f' % (input_, *value))

    print('')
    print('Filter 2')
    max_idx = 0
    max_product = 0.
    for i, value in enumerate(indexing[min_key]):
        product = value[0] * value[1] * value[-1]
        if product > max_product:
            max_idx = i
            max_product = product

    print('Input: %.2f -> Output: %.2f %.2f %.2f %.2f' % (input_, *indexing[min_key][max_idx]))
