import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin_tnc

stats = []
stats_test = []
repeat_names = []
repeat_names_test = []
past_title = False


with open('Seasons_Stats_1982-Present.csv', newline='') as csvfile:
    # Data preprocessing
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in spamreader:
        player = str(row).split(",")

        if not past_title:
            player[0] = player[0][2:]
            player[-1] = player[-1][:-2]
            stats.append(player)
        else:
            player[2] = player[2][:-1]
            player[3] = player[3][2:]
            name = player[2] + " " + player[3]
            player[2] = name
            player.pop(3)

            for i in range(len(player)):
                if player[i] == "": player[i] = "0"
                if player[i][-2:] == "']": player[i] = player[i][:-2]
                if player[i][-2:] == "\"]": player[i] = player[i][:-2]
            if player[2][-1:] == "*": player[2] = player[2][:-1]
            if player[0][:2] == "['": player[0] = player[0][2:]
            if name not in repeat_names:
                stats.append(player)
            if player[5] == "TOT":
                repeat_names.append(name)
        past_title = True
past_title = False


with open('Seasons_Stats_Test.csv', newline='') as csvfile:
    # Data preprocessing
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in spamreader:
        player = str(row).split(",")

        if not past_title:
            player[0] = player[0][2:]
            player[-1] = player[-1][:-2]
            stats_test.append(player)
        else:
            player[2] = player[2][:-1]
            player[3] = player[3][2:]
            name = player[2] + " " + player[3]
            player[2] = name
            player.pop(3)

            for i in range(len(player)):
                if player[i] == "": player[i] = "0"
                if player[i][-2:] == "']": player[i] = player[i][:-2]
                if player[i][-2:] == "\"]": player[i] = player[i][:-2]
            if player[2][-1:] == "*": player[2] = player[2][:-1]
            if player[0][:2] == "['": player[0] = player[0][2:]
            if name not in repeat_names_test:
                stats_test.append(player)
            if player[5] == "TOT":
                repeat_names_test.append(name)
        past_title = True


with open('Processed_Stats.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(stats)


with open('Processed_Stats_Test.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(stats_test)


def load_data(path, header):
    cols = [i for i in range(1, 52)]
    #Remove some redundant statistics
    cols.remove(31) #FG%
    cols.remove(39) #FT
    cols.remove(40) #FTA
    cols.remove(44) #TRB
    cols.remove(49) #PF
    marks_df = pd.read_csv(path, header=header, usecols=cols, skiprows=1)
    return marks_df


if __name__ == "__main__":
    # load the data from the file
    data = load_data("Processed_Stats.csv", None)
    data.fillna(0, inplace=True)

    test_data = load_data("Processed_Stats_Test.csv", None)
    test_data.fillna(0, inplace=True)

    # X = feature values, all the columns except the last column
    X = data.iloc[:, 8:-1]
    X_test = test_data.iloc[:, 8:-1]

    # y = target values, last column of the data frame
    y = data.iloc[:, -1].astype(int)

    MVPs = data.loc[y == 1]

    X = np.c_[np.ones((X.shape[0], 1)), X]
    X_test = np.c_[np.ones((X_test.shape[0], 1)), X_test]
    y = y[:, np.newaxis]
    theta = np.zeros((X.shape[1], 1))


def sigmoid(x):
    # Sigmoid activation function
    return 1 / (1 + np.exp(-x))


def prob(theta, x):
    # Returns probability after put through sigmoid
    return sigmoid(np.dot(x, theta))


def cost(theta, x, y):
    # Computes cost function
    m = x.shape[0]
    total_cost = -(1 / m) * np.sum(
        y * np.log(prob(theta, x)) + (1 - y) * np.log(
            1 - prob(theta, x)))
    return total_cost


def grad(theta, x, y):
    # Computes cost function gradient
    m = x.shape[0]
    return (1 / m) * np.dot(x.T, sigmoid(np.dot(x, theta)) - y)


def fit(x, y, theta):
    opt_weights = fmin_tnc(func=cost, x0=theta, fprime=grad, args=(x, y.flatten()))
    return opt_weights[0]


parameters = fit(X, y, theta)

def predict(x):
    theta = parameters[:, np.newaxis]
    return prob(theta, x)


def accuracy(x, actual_classes, prob_threshold=0.9):
    predicted_classes = (predict(x) >= prob_threshold).astype(int)
    predicted_classes = predicted_classes.flatten()
    accuracy = np.mean(predicted_classes == actual_classes)
    return accuracy * 100


prediction = 1 - predict(X_test)
names_with_probs = np.c_[test_data.iloc[:, 1], prediction]

print(names_with_probs[np.argsort(names_with_probs[:, 1])])
print("MVP Prediction: " + names_with_probs[np.argmax(prediction)][0])

print("Accuracy on training data: " + str(accuracy(X, y.flatten())))