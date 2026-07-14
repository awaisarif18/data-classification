"""
============================================================================
DECISION TREE CLASSIFIER FROM SCRATCH  (no scikit-learn)
Dataset : Iris  (150 flowers, 4 measurements, 3 species)
Method  : CART-style classification tree using GINI IMPURITY
Outputs : 1) iris_tree_growth.png  -> how the tree expands, node by node
          2) iris_tree_final.png   -> the finished, fully-labelled tree
============================================================================
The only "library" doing real ML work is YOUR code below. numpy handles the
math, pandas only reads the CSV, and matplotlib only draws the picture.
"""

import numpy as np                              # numerical arrays + fast math
import pandas as pd                             # only used to read the CSV cleanly
import matplotlib.pyplot as plt                 # to draw the tree ourselves
from matplotlib.patches import FancyBboxPatch   # rounded rectangles for the nodes


# ===========================================================================
# 1. LOAD THE DATA
# ===========================================================================
df = pd.read_csv("Iris.csv")                 # read the file into a table
df = df.drop(columns=["Id"])                 # 'Id' is a row counter, not a feature

feature_names = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"]

# X = the 4 measurement columns as one numeric numpy array, shape (150, 4).
X = df[["SepalLengthCm", "SepalWidthCm",
        "PetalLengthCm", "PetalWidthCm"]].to_numpy()

# Turn the 3 species strings into integer codes 0/1/2 so numpy can count fast.
# np.unique(..., return_inverse=True) returns BOTH the sorted unique names AND,
# for every row, the index of its name -> that index array IS our label y.
class_names, y = np.unique(df["Species"].to_numpy(), return_inverse=True)
n_classes = len(class_names)                  # = 3

short_names = ["setosa", "versicolor", "virginica"]
class_colors = [(0.99, 0.55, 0.24),           # orange  -> setosa
                (0.30, 0.69, 0.31),           # green   -> versicolor
                (0.61, 0.35, 0.71)]           # purple  -> virginica


# ===========================================================================
# 2. GINI IMPURITY  -- "how mixed is this group of flowers?"
# ===========================================================================
def gini(labels):
    """Gini impurity of integer labels. 0.0 = pure, higher = more mixed."""
    if len(labels) == 0:
        return 0.0
    counts = np.bincount(labels, minlength=n_classes)   # e.g. [50, 10, 0]
    probs = counts / len(labels)                        # counts -> fractions p_k
    return 1.0 - np.sum(probs ** 2)                     # formula: 1 - Σ p_k²


# ===========================================================================
# 3. BEST SPLIT  -- the greedy search: try every question, keep the best
# ===========================================================================
def best_split(X_node, y_node):
    """
    Scan EVERY feature and EVERY candidate threshold. Return the split that
    lowers the weighted child impurity the most (== highest information gain).
    Returns (feature_index, threshold, gain).  gain == 0 -> no useful split.
    """
    n = len(y_node)
    parent_gini = gini(y_node)
    best_gain, best_feature, best_threshold = 0.0, None, None

    for f in range(X_node.shape[1]):          # loop over the 4 feature columns
        values = np.unique(X_node[:, f])      # sorted, de-duplicated column values
        if len(values) == 1:                  # all identical -> cannot split
            continue
        # candidate thresholds = midpoints between consecutive distinct values.
        thresholds = (values[:-1] + values[1:]) / 2.0

        for t in thresholds:
            left_mask = X_node[:, f] <= t     # boolean array: True -> goes left
            right_mask = ~left_mask           # '~' flips every True/False
            gL = gini(y_node[left_mask])      # y_node[mask] keeps only True rows
            gR = gini(y_node[right_mask])
            nL, nR = left_mask.sum(), right_mask.sum()
            weighted = (nL * gL + nR * gR) / n
            gain = parent_gini - weighted

            if gain > best_gain:              # remember the champion so far
                best_gain, best_feature, best_threshold = gain, f, t

    return best_feature, best_threshold, best_gain


# ===========================================================================
# 4. THE NODE  -- one box in the tree
# ===========================================================================
class Node:
    def __init__(self, labels, depth):
        self.samples = len(labels)
        self.counts = np.bincount(labels, minlength=n_classes)  # e.g. [50,10,0]
        self.gini = gini(labels)
        self.prediction = int(np.argmax(self.counts))  # majority class -> leaf guess
        self.depth = depth
        self.feature = None       # which column this node tests (if it splits)
        self.threshold = None     # the cut value
        self.left = None          # child for  <= threshold
        self.right = None         # child for  >  threshold
        self.order = None         # construction step at which we split it

    @property
    def is_leaf(self):
        return self.left is None


# ===========================================================================
# 5. BUILD THE TREE  -- breadth-first, so we can watch it grow
# ===========================================================================
def build_tree(X, y, max_depth=None, min_samples=2):
    root = Node(y, depth=0)
    root.X, root.y = X, y             # stash this node's data on the node itself
    frontier = [root]                # a queue of nodes still waiting to be split
    step = 0

    while frontier:
        node = frontier.pop(0)       # pop(0) = take OLDEST node -> breadth-first

        # STOPPING RULES -> if any is true, the node stays a leaf.
        too_pure = node.gini == 0.0
        too_small = node.samples < min_samples
        too_deep = max_depth is not None and node.depth >= max_depth
        if too_pure or too_small or too_deep:
            continue

        f, t, gain = best_split(node.X, node.y)
        if f is None or gain <= 0:   # no split improves purity -> leaf
            continue

        node.feature, node.threshold, node.order = f, t, step
        step += 1

        left_mask = node.X[:, f] <= t                 # partition the data
        node.left = Node(node.y[left_mask], node.depth + 1)
        node.left.X, node.left.y = node.X[left_mask], node.y[left_mask]
        node.right = Node(node.y[~left_mask], node.depth + 1)
        node.right.X, node.right.y = node.X[~left_mask], node.y[~left_mask]

        frontier.append(node.left)   # new children join the back of the queue
        frontier.append(node.right)

    return root, step


# ===========================================================================
# 6. (BONUS) PREDICT  -- walk one flower down the finished tree
# ===========================================================================
def predict_one(node, x):
    while not node.is_leaf:
        node = node.left if x[node.feature] <= node.threshold else node.right
    return node.prediction


# ===========================================================================
# 7. LAYOUT  -- give every node an (x, y) position for drawing
# ===========================================================================
DX = 1.7            # horizontal gap between neighbouring leaves (spread them out)

def layout(node, coords, depth=0, counter=[0.0]):
    """Leaves get evenly spaced x's; each parent is centred above its children."""
    if node.is_leaf:
        x = counter[0]
        counter[0] += DX
    else:
        layout(node.left, coords, depth + 1, counter)
        layout(node.right, coords, depth + 1, counter)
        x = (coords[node.left][0] + coords[node.right][0]) / 2
    coords[node] = (x, -depth)
    return coords


# ===========================================================================
# 8. DRAW ONE NODE (a labelled, coloured box)
# ===========================================================================
BW, BH = 0.75, 0.34     # box half-width / half-height in data units

def draw_node(ax, node, pos, pending=False, fs=7):
    x, y = pos
    frac = node.counts[node.prediction] / node.samples      # purity of majority
    base = np.array(class_colors[node.prediction])
    fill = tuple(1 - frac * (1 - base))                     # blend toward white

    lines = []
    if not node.is_leaf and not pending:                    # already-split node
        lines.append(f"{feature_names[node.feature]} <= {node.threshold:.2f}")
    lines.append(f"gini = {node.gini:.3f}")
    lines.append(f"samples = {node.samples}")
    lines.append(f"[{', '.join(map(str, node.counts))}]")
    lines.append(f"-> {short_names[node.prediction]}")

    style = "--" if pending else "-"            # dashed = still in the queue
    box = FancyBboxPatch((x - BW, y - BH), 2 * BW, 2 * BH,
                         boxstyle="round,pad=0.02,rounding_size=0.10",
                         linewidth=1.1, edgecolor="0.35",
                         facecolor=fill, linestyle=style)
    ax.add_patch(box)
    ax.text(x, y, "\n".join(lines), ha="center", va="center",
            fontsize=fs, family="monospace")


# ===========================================================================
# 9. DRAW A WHOLE TREE (optionally only the part revealed by a given step)
# ===========================================================================
def draw_tree(ax, root, coords, up_to_step=None, title="", lims=None, fs=7):
    ax.set_title(title, fontsize=11)
    ax.axis("off")
    if lims:
        ax.set_xlim(lims[0]); ax.set_ylim(lims[1])   # same frame every panel

    # Collect the nodes visible in this frame (descend only through expanded ones).
    visible, stack = [], [root]
    while stack:
        nd = stack.pop()
        visible.append(nd)
        if (not nd.is_leaf) and (up_to_step is None or nd.order <= up_to_step):
            stack += [nd.left, nd.right]

    def expanded(nd):
        return (not nd.is_leaf) and (up_to_step is None or nd.order <= up_to_step)

    # Edges first (so boxes sit on top of the lines).
    for nd in visible:
        if expanded(nd):
            x0, y0 = coords[nd]
            for child, label in ((nd.left, "True"), (nd.right, "False")):
                x1, y1 = coords[child]
                ax.plot([x0, x1], [y0 - BH, y1 + BH], color="0.55", lw=1, zorder=1)
                lx, ly = x0 + 0.30 * (x1 - x0), (y0 - BH) + 0.30 * ((y1 + BH) - (y0 - BH))
                ax.text(lx, ly, label, fontsize=6, color="0.3", ha="center",
                        bbox=dict(boxstyle="round", fc="white", ec="none", pad=0.1),
                        zorder=2)

    # Then the node boxes. Internal-but-not-yet-expanded nodes are drawn 'pending'.
    for nd in visible:
        draw_node(ax, nd, coords[nd], pending=(not nd.is_leaf) and not expanded(nd), fs=fs)


# ===========================================================================
# 10. MAIN  -- run everything and save the two figures
# ===========================================================================
def main():
    root, n_splits = build_tree(X, y, max_depth=None, min_samples=2)
    print(f"Tree built. Total splits made: {n_splits}")

    coords = layout(root, {})
    xs = [p[0] for p in coords.values()]
    ys = [p[1] for p in coords.values()]
    lims = ((min(xs) - BW - 0.3, max(xs) + BW + 0.3),
            (min(ys) - BH - 0.4, BH + 0.4))
    span_x = (max(xs) - min(xs)) + 2 * BW + 1
    span_y = (max(ys) - min(ys)) + 2 * BH + 1

    # ---- FIGURE 1: growth sequence (one panel per split, then the final tree) ----
    panels = n_splits + 1
    ncols = 3
    nrows = int(np.ceil(panels / ncols))
    fig1, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * span_x * 0.62, nrows * span_y * 0.62))
    axes = np.array(axes).reshape(-1)
    for k in range(n_splits):
        draw_tree(axes[k], root, coords, up_to_step=k,
                  title=f"Step {k+1}: after split #{k+1}", lims=lims, fs=6)
    draw_tree(axes[n_splits], root, coords, up_to_step=None,
              title="Final tree", lims=lims, fs=6)
    for j in range(panels, len(axes)):
        axes[j].axis("off")
    fig1.suptitle("How the decision tree expands, split by split", fontsize=14)
    fig1.tight_layout(rect=[0, 0, 1, 0.98])
    fig1.savefig("iris_tree_growth.png", dpi=130, bbox_inches="tight")

    # ---- FIGURE 2: the final tree, large and detailed ----
    fig2, ax = plt.subplots(figsize=(span_x * 0.85, span_y * 1.25))
    draw_tree(ax, root, coords, up_to_step=None, lims=lims, fs=8,
              title="Final Iris decision tree  (box colour = predicted species)")
    fig2.tight_layout()
    fig2.savefig("iris_tree_final.png", dpi=150, bbox_inches="tight")

    preds = np.array([predict_one(root, xi) for xi in X])
    print(f"Training accuracy: {(preds == y).mean():.3f}")


if __name__ == "__main__":
    main()
