import numpy as np
from collections import defaultdict


class CatToNum(object):
    """
    Categorical feature constructor
    :params:
        target - name of the target column
        reg_ - regularization strength
    """
    def __init__(self, target, reg_):
        self.dct = {}
        self.target = target
        self.reg = reg_
    
    def cat_to_num(self, data_, columns, test=False):
        """
        Transform dataframes' categorical columns to numeric
        :params:
            data_ - a pd.DataFrame to process
            columns - list of column names to process
            test - bool. Whether, use cache or not
        :return:
            transformed pd.DataFrame
        """
        data = data_.copy()
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

        reg = self.reg #smoothing coeff
        y = len(data[self.target])
        mean = data[self.target].mean()
        d = defaultdict(lambda: mean)
        for uniq in np.unique(data[col]):
            k = len(data[data[col] == uniq])  # uniq count in col
            y_ = len(data[data[col] == uniq][data[self.target] == 1])  # sum of y for uniq value in col
            a = (y_ / y * k + data[self.target].mean() * reg) / (reg + k)
            d[uniq] = a
        data[col] = data[col].apply(lambda x: d[x])
        return data, d