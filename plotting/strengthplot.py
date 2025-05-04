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

    # 3) get convergence map if available
    try:
        conv_map = model.has_converged(epsilon=epsilon, last_n=delta)
    except Exception:
        raise ValueError("No convergence data found—run solve(..., generate_plot=True) first")

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
    # 4) make the axes
    fig, ax = plt.subplots(figsize=(wide, tall))
    # hide top & right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    texts = []

    for name, seq in graph_data.items():
        xs = [t for (t, _) in seq]
        ys = [v for (_, v) in seq]
        (line,) = ax.plot(xs, ys, label=name, color=name2color[name])
        # decide convergence
        if name in conv_map:
            did_conv = conv_map[name]
        else:
            last_vals = ys[-delta:]
            did_conv = max(last_vals) - min(last_vals) <= epsilon

        # annotate at end
        x_last, y_last = xs[-1], ys[-1]
        marker = "✓" if did_conv else "✗"
        text = ax.text(
            x_last + (xs[-1] - xs[0]) * 0.01,
            y_last,
            f"{name} {marker}",
            va="center",
            fontsize="small",
            color=line.get_color()
        )
        texts.append(text)

    ax.set_xlabel("Time (t)")
    ax.set_ylabel("Strength (t)")
    ax.set_title(title)

    # 5) gently separate overlapping labels
    ymin, ymax = ax.get_ylim()
    sep = (ymax - ymin) * 0.02  # 2% of y-span
    # sort from bottom to top
    texts.sort(key=lambda txt: txt.get_position()[1])
    for i in range(1, len(texts)):
        x, y = texts[i].get_position()
        _, y_prev = texts[i-1].get_position()
        if y < y_prev + sep:
            texts[i].set_position((x, y_prev + sep))

    # now if any label exceeds the top, push *it* down to max and
    # then cascade a downward bump to any that collide below it
    top_limit = ymax - sep/2
    for i in range(len(texts)-1, -1, -1):
        x, y = texts[i].get_position()
        if y > top_limit:
            # clamp this one
            texts[i].set_position((x, top_limit))
            # now push any preceding ones below it
            for j in range(i-1, -1, -1):
                xj, yj = texts[j].get_position()
                _, y_above = texts[j+1].get_position()
                if yj > y_above - sep:
                    texts[j].set_position((xj, y_above - sep))

    fig.tight_layout(rect=(0,0,0.85,1))
    return plt
