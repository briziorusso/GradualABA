from matplotlib import pyplot as plt

def strengthplot(
    model,
    delta,
    epsilon,
    title="Untitled graph",
    wide=6,
    tall=4
):
    """
    Plots strength over time for each assumption in `model.graph_data`.

    model          -- a DiscreteModular (or similar) with graph_data populated
    delta (int)    -- if ≥2, cluster on the last `delta` points; 
                      if ==1, cluster on the *entire* series instead
    epsilon (float)-- max‐diff tolerance for grouping two series into the same color
    title (str)    -- plot title
    wide (float)   -- width of the figure in inches
    tall (float)   -- height of the figure in inches
    """
    # 1) parameter checks
    if not isinstance(delta, int) or delta < 1:
        raise TypeError("delta must be an integer ≥ 1")
    if not isinstance(epsilon, (int, float)) or epsilon < 0:
        raise TypeError("epsilon must be a non-negative number")
    if not (isinstance(wide, (int, float)) and wide > 0):
        raise TypeError("wide must be a positive number")
    if not (isinstance(tall, (int, float)) and tall > 0):
        raise TypeError("tall must be a positive number")

    # 2) grab the time-series data
    graph_data = getattr(model, "graph_data",
                       getattr(model, "approximator", {}).get("graph_data", None))
    if not graph_data:
        raise ValueError("No graph_data found—run solve(..., generate_plot=True) first")

    # 3) build a dict of the series we’ll compare
    series_to_compare = {}
    if delta == 1:
        for name, seq in graph_data.items():
            series_to_compare[name] = [v for (_, v) in seq]
    else:
        for name, seq in graph_data.items():
            if len(seq) < delta:
                raise ValueError(f"Not enough points for '{name}' to take last {delta}")
            series_to_compare[name] = [v for (_, v) in seq[-delta:]]

    # 4) cluster by last‐delta or full series
    clusters = []
    for name, vals in series_to_compare.items():
        placed = False
        for cluster in clusters:
            rep = series_to_compare[cluster[0]]
            max_diff = max(abs(a - b) for a, b in zip(vals, rep))
            if max_diff <= epsilon:
                cluster.append(name)
                placed = True
                break
        if not placed:
            clusters.append([name])

    # 5) assign colors
    cmap = plt.get_cmap("tab10")
    name2color = {}
    for idx, cluster in enumerate(clusters):
        col = cmap(idx % 10)
        for nm in cluster:
            name2color[nm] = col

    # 6) plot with adjustable size
    fig, ax = plt.subplots(figsize=(wide, tall))
    for name, seq in graph_data.items():
        xs = [i for (i, _) in seq]
        ys = [v for (_, v) in seq]
        ax.plot(xs, ys, label=name, color=name2color[name])

    ax.set_xlabel("Time (t)")
    ax.set_ylabel("Strength (t)")
    ax.set_title(title)

    # 7) alphabetical legend on the right
    handles, labels = ax.get_legend_handles_labels()
    sorted_pairs = sorted(zip(labels, handles), key=lambda x: x[0])
    sorted_labels, sorted_handles = zip(*sorted_pairs)
    ax.legend(
        sorted_handles,
        sorted_labels,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5)
    )

    # 8) leave room on the right
    fig.tight_layout(rect=(0, 0, 0.8, 1))

    return plt
