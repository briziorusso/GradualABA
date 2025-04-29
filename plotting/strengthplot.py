from matplotlib import pyplot as plt


def strengthplot(model, delta, epsilon, title="Untitled graph"):

    if (type(delta) != float and type(delta) != int):
        raise TypeError("delta must be a float or integer")

    if (type(epsilon) != float and type(epsilon) != int):
        raise TypeError("epsilon must be a float or integer")

    fig, ax = plt.subplots()
    graph_data = model.approximator.graph_data if hasattr(model, 'approximator') else model.graph_data
    if not graph_data:
        raise ValueError("No graph data found in the model. Please run the model first.")
    for x in graph_data:
        x_values = [x[0] for x in graph_data[x]]
        y_values = [y[1] for y in graph_data[x]]
        ax.plot(x_values, y_values, label=x)

    ax.set_xlabel('Time (t)')  # Add an x-label to the axes.
    ax.set_ylabel('Strength (t)')  # Add a y-label to the axes.
    ax.set_title(title)  # Add a title to the axes.
    ax.legend()  # Add a legend.

    return plt
