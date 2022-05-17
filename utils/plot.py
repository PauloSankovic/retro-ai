from pathlib import Path

import matplotlib.pyplot as plt

DIR = Path(__file__).parent.parent


def create_path(base_dir: str, model_name: str, parts: dict) -> Path:
    name_parts = [f"{key}={value}" for key, value in parts.items()]
    name_parts.insert(0, model_name)
    return DIR / base_dir / "_".join(name_parts)


def plot_model(model):
    x_param_label = model.params[0]
    for i in range(1, len(model.params)):
        y_param_label = model.params[i]
        plot_model_param(model, x_param_label, y_param_label, f'{x_param_label[1]}/{y_param_label[1]}')


def plot_model_param(model, x_param_label: tuple, y_param_label: tuple, title: str = ""):
    x_param, x_label = x_param_label
    y_param, y_label = y_param_label
    plot_and_save(model.train_summary[x_param], model.train_summary[y_param], title=title, xlabel=x_label,
                  ylabel=y_label)


def prepare_plot(x, y, title: str, xlabel: str, ylabel: str):
    plt.grid()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(x, y)


def plot_and_save(x, y, title: str, xlabel: str, ylabel: str, base_dir: str = "figures"):
    path = DIR / base_dir / title.lower().replace(" ", "_").replace("/", "_")
    prepare_plot(x, y, title, xlabel, ylabel)
    plt.savefig(f'{path}.png')
    plt.show()


def plot(x, y, title: str, xlabel: str, ylabel: str):
    prepare_plot(x, y, title, xlabel, ylabel)
    plt.show()
