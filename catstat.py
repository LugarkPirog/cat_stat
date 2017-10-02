import numpy as np
from collections import defaultdict


class CatToNum(object):
    def __init__(self):
        self.dct = {}

    def cat_to_num(self, data, columns, test=False):
        """
        Transform dataframes' categorical columns to numeric

        :params:
            data - a pd.DataFrame to process
            columns - cols to process
            test - bool. Whether, use cache or not

        :return:
            transformed data
        """
        if not test:
            for col in columns:
                data[col] = data[col].astype('str')
                data, self.dct[col] = self.col_to_num(data, col)
        else:
            for col in columns:
                data[col] = data[col].astype('str')
                data[col] = data[col].apply(lambda x: self.dct[col][x])
        return data

    def col_to_num(self, data, col):
        reg = 20  # smoothing coeff
        y = len(data['target'])
        mean = data['target'].mean()
        d = defaultdict(lambda: mean)
        for uniq in np.unique(data[col]):
            k = len(data[data[col] == uniq])  # uniq count in col
            y_ = len(data[data[col] == uniq][data['target'] == 1])  # sum of y for uniq value in col
            a = (k * y_ / y + data['target'].mean() * reg) / (reg + k)
            d[uniq] = a
        data[col] = data[col].apply(lambda x: d[x])
        return data, d
