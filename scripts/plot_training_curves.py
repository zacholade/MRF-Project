import matplotlib.pyplot as plt
import csv
import os

file_names = os.listdir("../Figures/training_curves")
file_names = [file for file in file_names if file.endswith("csv")]

def plot(reader, file_name, save):
        fig, ax = plt.subplots(1, 1, figsize=(12, 7))
        ax.margins(0)
        ax.grid(True)
        ax.plot(train_mapes, label="Training", color='royalblue', linestyle='-.', linewidth=2)
        ax.plot(valid_mapes, label="Validation", color='gold', linestyle='-.', linewidth=2)
        ax.legend(loc="upper left", prop={'size': 20})
        ax.set_xlabel("Epoch Number", family='Arial', fontsize=20)
        ax.set_ylabel("MAPE (%)", family='Arial', fontsize=20)
        ax.set_ylim([0, y_limit])

        # ax.set_xticklabels(x_ticklabels)
        ax.set_title(title, fontsize=20, family="Arial")
        plt.show()

        b = input("Save? > ")
        if b == '1':
            fig, ax = plt.subplots(1, 1, figsize=(12, 7))
            ax.margins(0)
            ax.grid(True)
            ax.plot(train_mapes, label="Training", color='royalblue', linestyle='-.', linewidth=2)
            ax.plot(valid_mapes, label="Validation", color='gold', linestyle='-.', linewidth=2)
            ax.legend(loc="upper left", prop={'size': 20})
            ax.set_xlabel("Epoch Number", family='Arial', fontsize=20)
            ax.set_ylabel("MAPE (%)", family='Arial', fontsize=20)
            ax.set_ylim([0, y_limit])

            # ax.set_xticklabels(x_ticklabels)
            ax.set_title(title, fontsize=20, family="Arial")
            plt.savefig("../Figures/training_curves/" + file_name.split('.')[0] + '.png')


file_name = "hoppe_no_spatial.csv"
with open("../Figures/training_curves/" + file_name, 'r') as file:
    reader = csv.reader(file)

    train_mape_row = 2
    valid_mape_row = 16

    train_mapes = []
    valid_mapes = []
    for i, row in enumerate(reader):
        if i == 0:
            continue

        train_mapes.append(float(row[train_mape_row]))
        valid_mapes.append(float(row[valid_mape_row]))

    avg_valid = sum(valid_mapes[-20:]) / 20
    avg_train = sum(train_mapes[-20:]) / 20
    print(f"avg_train: {avg_train}")
    print(f"avg_valid: {avg_valid}")

    print(file_name)
    valid_adjustment = float(input("valid adjustment:"))
    valid_mapes = [valid_mape + valid_adjustment for valid_mape in valid_mapes]
    print(f"max_mape: {max([*train_mapes, *valid_mapes])}")
    y_limit = int(input("enter y limit >  "))

    title = input("Title  > ")

    plot(reader, file_name.split('.')[0] + ".png", save=False)



