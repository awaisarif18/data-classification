# Today's Learnings — Decision Tree Classification on Iris

A log of the concepts, skills, and discoveries from today's session — written
to actually stick, not just to summarize. Today was a hard pivot from
*regression* (predicting a number) to *classification* (predicting a category),
and building a decision tree from scratch, no sklearn.

---

## 1. Classification is a different animal from regression

Yesterday everything predicted a **number** on a continuous scale, and error
was something like squared distance. Today the output is a **category** — one
of three species — and "distance" stops meaning anything (there's no
meaningful gap between setosa and virginica; you're just right or wrong). Two
concrete consequences: the scoreboard changes from R² to **accuracy**, and the
thing we *minimise while building the model* changes from squared error to
**impurity**. Same overall move as before (find the model that's best on
average across all the data), totally different machinery.

## 2. A decision tree is a flowchart, not a weighted sum

The linear/weighted model was democratic — every feature spoke at once, each
scaled by its weight, all summed. A decision tree is the opposite: **one
yes/no question at a time**, nested. Geometrically that means axis-aligned
rectangular cuts instead of one tilted boundary. The payoff that made it click:
a finished tree is **readable as plain-English rules** ("if petal length ≤ 2.45
→ setosa; else if petal width ≤ 1.75 → versicolor; else virginica"), which you
simply cannot do with a bag of weights. Interpretability is a real, practical
advantage of trees.

## 3. Gini impurity — the number that replaces "loss"

Gini = **1 − Σ pₖ²**, where pₖ is the fraction of the node that is class k.
Intuition: the probability you'd misclassify a random sample if you labelled it
by drawing from the node's own class mix. Proved I owned it by hand:

- Root (50/50/50): 1 − 3×(1/3)² = **0.667** — maximally messy for 3 classes.
- A pure node (50/0/0): 1 − 1² = **0** — done, becomes a leaf.
- 10 setosa + 30 versicolor: 1 − (0.25² + 0.75²) = **0.375** — and I correctly
  read that as *more pure than the root* because one class dominates.

Gini 0 is the base case that stops the whole algorithm. Entropy is the
information-theory alternative (−Σ pₖ log pₖ); gives near-identical trees, Gini
is cheaper.

## 4. Choosing a split = the same "scan candidates, keep the best" move as the weight search

The single biggest connection today: finding a split is *the same skeleton* as
yesterday's weight search — enumerate candidates, score each against **all** the
data, take the best. For a tree node: for every feature, for every candidate
threshold, split into left (≤) and right (>), compute the **size-weighted
average impurity** of the two children `(nL·gL + nR·gR)/n`, and the **gain** =
parent impurity − that weighted child impurity. Pick max gain.

Two honest differences from the weight bowl: (a) it's a **discrete, jagged**
search over candidate cuts, not a smooth minimum — pure brute enumeration, no
calculus; (b) it's **greedy** — best split *now*, never checking if a worse
split now enables a better tree later. Greedy trees are not globally optimal,
and that's fine to say out loud.

## 5. Candidate thresholds are the midpoints between sorted values

You don't try infinitely many cut points — a cut anywhere between two
neighbouring values produces the identical left/right grouping, so you only
need **one representative per gap: the midpoint**. N distinct values → N−1
gaps → N−1 candidate thresholds. In numpy this is a slick one-liner:
`(values[:-1] + values[1:]) / 2` — offset the sorted array against itself by
one and average. Duplicate values collapse (via `np.unique`), so shared values
give fewer candidates. That slicing pattern is worth memorising.

## 6. Built a whole tree by hand before touching code

On a 6-flower toy set (2 each of setosa/versicolor/virginica, petal length +
width), I ran the full algorithm on paper:

- Root Gini 0.667. Scanned all candidates; best weighted Gini = **0.333**,
  and *four* splits tied for it — a real reminder that **ties happen**, and the
  mechanical rule is "take the first best in scan order."
- Took petal length ≤ 2.75 → left child pure setosa (**leaf**), right child
  {2 versicolor, 2 virginica}, Gini 0.5 → **recurse**.
- On the right child, petal length ≤ 4.8 scored weighted Gini **0.000** — a
  perfect split, gain 0.5 — both children pure. Tree done at depth 2.

Seeing the recursion terminate itself (every leaf pure) made the base case
concrete. The tree never even *used* petal width — it decided on its own which
features mattered.

## 7. Real numpy fluency, from writing the algorithm myself

The from-scratch build forced real numpy, not library calls. The idioms I now
actually understand:

- `np.unique(labels, return_inverse=True)` → **label-encode strings to 0/1/2 in
  one line** (returns the sorted names AND each row's index into them).
- `np.bincount(labels, minlength=n_classes)` → **count each class fast**, using
  the integer label as the index; `minlength` guards against missing classes.
- **Boolean masks are the whole game**: `X[:, f] <= t` returns a True/False
  array; `~mask` flips it; `y[mask]` keeps only the True rows; `mask.sum()`
  counts them (True == 1). This replaces every filter loop I'd have written.
- **Vectorization**: `probs ** 2` then `np.sum(...)` computes 1 − Σpₖ² across
  the whole array with no loop.
- `np.argmax(counts)` → the majority class (a node's fallback prediction).

## 8. The full 150-row result — and overfitting made visible

Ran it on the real data: **8 splits, depth 5, 100% training accuracy.** The
setosa split stays perfectly clean (petal length ≤ 2.45 isolates all 50 in one
cut — exactly like the toy). But the versicolor/virginica boundary is genuinely
fuzzy, so the tree keeps splitting down to leaves of `samples = 1` and
`samples = 2` just to chase the few overlapping flowers. That's **overfitting
you can literally see** — the model memorising stragglers. The fix in a real
project is one argument: cap `max_depth`. (This time I dropped train/test per
the mentor's instruction — the task was just to construct the tree — so I let
it grow to purity, but I can now point at exactly *where* it would overfit.)

## 9. Built the visualization from scratch too — and one parameter powers the "growth" view

Drew the tree with matplotlib (a plotting library, **not** an ML shortcut, so
it respects the no-sklearn rule). Two ideas worth keeping: colour **encodes
purity** (blend the class colour toward white by how impure the node is — vivid
= pure leaf, pale = mixed), and the entire "watch it grow" animation comes from
**one flag**: when drawing frame k, only descend into a node's children if that
node was split at step ≤ k; otherwise draw it dashed as a "pending" node still
sitting in the queue. Set the flag to None → the final tree. Same draw
function, both pictures.

## 10. Why build it instead of calling `.fit()`

`sklearn`'s `DecisionTreeClassifier().fit(X, y)` does all of this in one line —
and hides *exactly* the two things I was supposed to learn: the impurity math
and the greedy recursion. Same lesson as yesterday's "verify by execution":
implementing Gini + the split scan by hand turns "the tree just knows" into "I
can see precisely why it asked that question." The whole algorithm is now three
pieces I can rebuild from memory — `gini(labels)`, `best_split(X, y)` (the
scan), and `build_tree(...)` (recurse until pure) — plus a hand-drawn picture.
