# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pandas as pd

if __name__ == '__main__':
    data_path = ''
    df = pd.read_csv(data_path)
    df2 = df[df['Summary'].str.len() > 150]
    df3 = df2[df2['Summary'].str.len() < 160]
    df3.to_csv('flipkart_review1.csv', index=False)