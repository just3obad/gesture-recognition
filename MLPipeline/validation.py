import matplotlib.pyplot as plt
import pandas as pd


def plot_loss(loss_csv, save_fig=False):
    loss_df = pd.read_csv(loss_csv)
    print(loss_df["loss"])
    fig = plt.figure(figsize=(6, 6))
    plt.plot(loss_df["loss"], label="loss")
    plt.show()
    if save_fig:
        fig.savefig("loss.jpg")


if __name__ == '__main__':
    lr_loss_path = "models/loss.csv"
    plot_loss(lr_loss_path, save_fig=True)
