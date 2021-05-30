import csv
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
from sklearn.linear_model import LogisticRegression
import numpy as np


def convert():
    with open('dots_dist_data.json', 'r') as openfile:
        existing_info = json.load(openfile)

    csv_data = []
    for row in existing_info["rows"]:
        for dot in row["dots"]:
            csv_data.append([row["k"], dot["dist"], dot["is_come_together"]])

    with open('dots_dist_data.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["k", "dist", "is_come_together"])
        writer.writerows(csv_data)


def visualize():
    df = pd.read_csv("dots_dist_data.csv")

    seaborn.scatterplot(data=df, x='k', y='dist', hue='is_come_together', palette="tab10")
    plt.show()


def regression():
    # todo visualize doesn't work
    with open('dots_dist_data.json', 'r') as openfile:
        existing_info = json.load(openfile)

    csv_data = []
    for row in existing_info["rows"]:
        for dot in row["dots"]:
            csv_data.append([row["k"], dot["dist"], dot["is_come_together"]])

    data = {
        "x": [x[0] * 1000 for x in csv_data],
        "y": [x[1] for x in csv_data],
        "t": [int(x[2]) for x in csv_data]
    }

    df = pd.DataFrame(data, columns=['x', 'y', 't'])

    X = df[['x', 'y']]
    Y = df['t']

    clf = LogisticRegression(random_state=0).fit(X, Y)

    # Retrieve the model parameters.
    b = clf.intercept_[0]
    w1, w2 = clf.coef_.T
    # Calculate the intercept and gradient of the decision boundary.
    c = -b / w2
    m = -w1 / w2

    print(f"y = {m} * x + {c}")

    # Plot the data and the classification with the decision boundary.
    xmin, xmax = 0, 30
    ymin, ymax = 0, 3
    xd = np.array([xmin, xmax])
    yd = m * xd + c
    plt.plot(xd, yd, 'k', lw=1, ls='--')
    plt.fill_between(xd, yd, ymin, color='tab:blue', alpha=0.2)
    plt.fill_between(xd, yd, ymax, color='tab:orange', alpha=0.2)

    plt.scatter(x=[x[0] * 1000 for x in csv_data if x[2]], y=[x[1] for x in csv_data if x[2]], s=8, alpha=0.5)
    plt.scatter(x=[x[0] * 1000 for x in csv_data if not x[2]], y=[x[1] for x in csv_data if not x[2]], s=8, alpha=0.5)
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.ylabel(r'$x_2$')
    plt.xlabel(r'$x_1$')

    plt.show()


if __name__ == '__main__':
    convert()
    visualize()
    regression()
