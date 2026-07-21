import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def display_points(X, Y, xlabel='X', ylabel='Y'):
    plt.figure(figsize=(10,8))
    for x, y in zip(X, Y):
        plt.scatter(x, y, color='red')
    plt.plot(X, Y, color='blue')
    plt.title("Early Verification", fontsize=20)
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel(ylabel, fontsize=15)
    plt.show()



def costs_VS_iters(costs, iters, color='b'):
    plt.figure(figsize=(15, 13))
    xItr = np.arange(iters)
    plt.plot(xItr, costs, linestyle='-', color=color, label="House Size")
    plt.xlabel("iterations", fontsize=15)
    plt.ylabel("cost", fontsize=15)
    plt.title("Cost Vs Iterations", fontsize=20)
    plt.legend(fontsize=20)
    plt.grid(True)
    plt.show()

def featurs_Vs_target(df1,df2,df3):
    # Create figure of three plots
    fig, axis = plt.subplots(1, 3,figsize=(24,8))

    # Feat1
    df1.sort_values(by="Feat1", inplace=True)
    axis[0].set_title("Feat1 VS Target", fontsize=20)
    axis[0].set_xlabel("Feat1", fontsize=15)
    axis[0].set_ylabel("Target", fontsize=15)
    axis[0].scatter(df1['Feat1'],df1['Target'], color="g")

    # Feat2
    df2.sort_values(by="Feat2", inplace=True)
    axis[1].scatter(df2['Feat2'],df2['Target'],linestyle='-', color="r")
    axis[1].set_title("Feat2 VS Target", fontsize=20)
    axis[1].set_xlabel("Feat2", fontsize=15)
    axis[1].set_ylabel("Target", fontsize=15)

    # Feat3
    df3.sort_values(by="Feat3", inplace=True)
    axis[2].scatter(df3['Feat3'],df3['Target'],linestyle='-', color="b")
    axis[2].set_title("Feat3 VS Target", fontsize=20)
    axis[2].set_xlabel("Feat3", fontsize=15)
    axis[2].set_ylabel("Target", fontsize=15)
    plt.tight_layout()
    plt.show()