from pathlib import Path

import matplotlib.pyplot as plt

DIR = Path(__file__).parent


def plot_model(model, x_label_param: tuple, y_label_param: tuple, title: str):
    x_label, x_param = x_label_param
    y_label, y_param = y_label_param
    plot(model.train_summary[x_param], model.train_summary[x_param], title=title, xlabel=x_label, ylabel=y_label)


def prepare_plot(x, y, title: str, xlabel: str, ylabel: str):
    plt.grid()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(x, y)


def plot_and_save(x, y, title: str, xlabel: str, ylabel: str, base_dir: str = "figures"):
    path = DIR / base_dir / title.lower().replace(" ", "_")
    prepare_plot(x, y, title, xlabel, ylabel)
    plt.savefig(f'{path}.png')
    plt.show()


def plot(x, y, title: str, xlabel: str, ylabel: str):
    prepare_plot(x, y, title, xlabel, ylabel)
    plt.show()
