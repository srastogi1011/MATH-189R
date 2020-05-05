import csv
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


seasons = [1982, 1989, 1996, 2003, 2010, 2017]

for s in seasons:
    stats = []
    stats_test = []
    repeat_names = []
    repeat_names_test = []
    past_title = False

    with open("Seasons/Season_" + str(s) + ".csv", newline='') as csvfile:
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


    with open("Processed_" + str(s) + ".csv", 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(stats_test)


if __name__ == "__main__":
    indexed_features = ["Index", "Year", "Player", "Pos", "Age", "Tm", "G", "GS", "MP", "MPG", "PER", "TS%", "3PAr", "FTr", "ORB%",
                        "DRB%", "TRB%", "AST%", "STL%", "BLK%", "TOV%", "USG%", "OWS", "DWS", "WS", "WS/48", "OBPM",
                        "DBPM", "BPM", "VORP", "FG", "FGA", "FG%", "3P", "3PA", "3P%", "2P", "2PA", "2P%", "eFG%", "FT",
                        "FTA", "FT%", "ORB", "DRB", "TRB", "AST", "STL", "BLK", "TOV", "PF", "PTS"]

    features = ["Player", "GS", "MP", "MPG", "PER", "TS%", "3PAr", "FTr", "ORB%", "DRB%", "TRB%", "AST%", "STL%", "BLK%", "TOV%",
                "USG%", "OWS", "DWS", "WS", "WS/48", "OBPM", "DBPM", "BPM", "VORP", "FG", "FGA", "FG%", "3P", "3PA",
                "3P%", "2P", "2PA", "2P%", "eFG%", "FT", "FTA", "FT%", "ORB", "DRB", "TRB", "AST", "STL", "BLK", "TOV",
                "PF", "PTS"]

    offensive_features = ["Player", "GS", "MP", "MPG", "PER", "TS%", "3PAr", "FTr", "AST%", "TOV%", "USG%", "OWS", "OBPM",
                          "VORP", "FG", "FGA", "FG%", "3P", "3PA", "3P%", "2P", "2PA", "2P%", "eFG%", "FT", "FTA",
                          "FT%", "AST", "TOV", "PTS"]

    defensive_features = ["Player", "GS", "MP", "MPG", "STL%", "BLK%", "DWS", "DBPM", "VORP", "STL", "BLK", "PF"]

    for s in seasons:
        # load the data from the file
        data = pd.read_csv("Processed_" + str(s) + ".csv", names=indexed_features).iloc[1:, :]
        data.fillna(0, inplace=True)

        # X = feature values, all the columns
        X = data.loc[:, features].values
        off_X = data.loc[:, offensive_features].values
        def_X = data.loc[:, defensive_features].values

        X_df_named = pd.DataFrame(data=X, columns=features)
        X_df = X_df_named.iloc[:, 1:].astype(float)

        off_X_df_named = pd.DataFrame(data=off_X, columns=offensive_features)
        off_X_df = off_X_df_named.iloc[:, 1:].astype(float)

        def_X_df_named = pd.DataFrame(data=def_X, columns=defensive_features)
        def_X_df = def_X_df_named.iloc[:, 1:].astype(float)

        # Get names of indexes for which players play < 30 min per game
        indexNames = X_df[X_df["MPG"] < 30].index
        # Delete these row indexes from dataFrame
        X_df.drop(indexNames, inplace=True)
        X_df_named.drop(indexNames, inplace=True)
        off_X_df.drop(indexNames, inplace=True)
        off_X_df_named.drop(indexNames, inplace=True)
        def_X_df.drop(indexNames, inplace=True)
        def_X_df_named.drop(indexNames, inplace=True)

        # Get names of indexes for which players start < 40 games
        indexNames = X_df[X_df["GS"] < 40].index
        # Delete these row indexes from dataFrame
        X_df.drop(indexNames, inplace=True)
        X_df_named.drop(indexNames, inplace=True)
        off_X_df.drop(indexNames, inplace=True)
        off_X_df_named.drop(indexNames, inplace=True)
        def_X_df.drop(indexNames, inplace=True)
        def_X_df_named.drop(indexNames, inplace=True)

        # Get names of indexes for which players have < 20 PER
        indexNames = X_df[X_df["PER"] < 20].index
        # Delete these row indexes from dataFrame
        X_df.drop(indexNames, inplace=True)
        X_df_named.drop(indexNames, inplace=True)
        off_X_df.drop(indexNames, inplace=True)
        off_X_df_named.drop(indexNames, inplace=True)
        def_X_df.drop(indexNames, inplace=True)
        def_X_df_named.drop(indexNames, inplace=True)

        X = X_df.to_numpy()
        off_X = off_X_df.to_numpy()
        def_X = def_X_df.to_numpy()

        X = StandardScaler().fit_transform(X)
        off_X = StandardScaler().fit_transform(off_X)
        def_X = StandardScaler().fit_transform(def_X)

        pca = PCA(n_components=2)

        # Overall
        comps = pca.fit_transform(X)
        comps_df = pd.DataFrame(data=comps, columns=["Principle Component 1", "Principle Component 2"])

        kMeans = KMeans(n_clusters=5)
        y_kMeans = kMeans.fit_predict(X_df)

        plt.scatter(comps_df.iloc[:, 0], comps_df.iloc[:, 1], c=["C" + str(col) for col in y_kMeans])
        plt.xlabel("Principle Component 1")
        plt.ylabel("Principle Component 2")
        plt.title(str(s) + " Overall Player Impact (" + str(len(X_df.index)) + " players)")
        for i, txt in enumerate(X_df_named.iloc[:, 0]):
            plt.annotate(txt, (comps_df.iloc[:, 0][i], comps_df.iloc[:, 1][i]), fontsize=6)
        plt.savefig(str(s) + "_Graphs/PCA_Overall_" + str(s) + ".png", format="png")
        plt.close()

        if s == 2017:
            Error = []
            for k in range(1, 11):
                kmeans = KMeans(n_clusters=k).fit(X_df)
                kmeans.fit(X_df)
                Error.append(kmeans.inertia_)

            plt.plot(range(1, 11), Error)
            plt.title('Elbow method')
            plt.xlabel('No of clusters')
            plt.ylabel('Error')
            plt.savefig("2017_Graphs/K_Means_Error_2017.png", format="png")
            plt.close()

    # Offensive breakdown - 2017
    comps = pca.fit_transform(off_X)
    comps_df = pd.DataFrame(data=comps, columns=["Principle Component 1", "Principle Component 2"])

    kMeans = KMeans(n_clusters=5)
    y_kMeans = kMeans.fit_predict(off_X_df)

    plt.scatter(comps_df.iloc[:, 0], comps_df.iloc[:, 1], c=["C" + str(col) for col in y_kMeans])
    plt.xlabel("Principle Component 1")
    plt.ylabel("Principle Component 2")
    plt.title("2017 Offensive Impact (" + str(len(X_df.index)) + " players)")
    for i, txt in enumerate(off_X_df_named.iloc[:, 0]):
        plt.annotate(txt, (comps_df.iloc[:, 0][i], comps_df.iloc[:, 1][i]), fontsize=6)
    plt.savefig("2017_Graphs/PCA_2017_Offensive.png", format="png")
    plt.close()

    # Defensive breakdown - 2017
    comps = pca.fit_transform(def_X)
    comps_df = pd.DataFrame(data=comps, columns=["Principle Component 1", "Principle Component 2"])

    for j in range(4, 9):
        kMeans = KMeans(n_clusters=j)
        y_kMeans = kMeans.fit_predict(def_X_df)

        plt.scatter(comps_df.iloc[:, 0], comps_df.iloc[:, 1], c=["C" + str(col) for col in y_kMeans])
        plt.xlabel("Principle Component 1")
        plt.ylabel("Principle Component 2")
        plt.title("2017 Defensive Impact (" + str(len(X_df.index)) + " players)")
        for i, txt in enumerate(def_X_df_named.iloc[:, 0]):
            plt.annotate(txt, (comps_df.iloc[:, 0][i], comps_df.iloc[:, 1][i]), fontsize=6)
        plt.savefig("2017_Graphs/PCA_2017_Defensive_" + str(j) + ".png", format="png")
        plt.close()