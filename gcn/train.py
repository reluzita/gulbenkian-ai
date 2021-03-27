import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch.nn as nn
import matplotlib.pyplot as plt
from model import Model

if __name__ == '__main__':
    df = pd.read_pickle('C:\\Users\\ineso\\FEUP-3ano\\gulbenkian-ai\\gcn\\user_cat_bool_df.pickle')
    df = df.astype({'user_id':'category'})
    df = df.sample(frac=0.1, random_state=1)
    categorical_columns = list(df.columns)
    categorical_columns.remove('stars')

    categorical_data = np.stack([df[c].cat.codes.values for c in categorical_columns], 1)
    categorical_data = torch.tensor(categorical_data, dtype=torch.int64)

    outputs = torch.tensor(df['stars'].values).flatten()

    train_data, test_data, train_output, test_output = train_test_split(categorical_data, outputs, test_size=0.2, random_state=42)

    categorical_column_sizes = [len(df[column].cat.categories) for column in categorical_columns]
    categorical_embedding_sizes = [(col_size, min(50, (col_size+1)//2)) for col_size in categorical_column_sizes]

    model = Model(categorical_embedding_sizes, 5, [200,100,50], p=0.4)

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 300
    aggregated_losses = []

    for i in range(epochs):
        i += 1
        y_pred = model(train_data)
        single_loss = loss_function(y_pred, train_output-1)
        aggregated_losses.append(single_loss)

        if i%25 == 1:
            print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

        optimizer.zero_grad()
        single_loss.backward()
        optimizer.step()

    print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')

    plt.plot(range(epochs), aggregated_losses)
    plt.ylabel('Loss')
    plt.xlabel('epoch')