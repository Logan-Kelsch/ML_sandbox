# Vector 01: from gradients to neural-guided program search

A visual, from-scratch learning sequence. Each numbered prompt corresponds to one
notebook and a separate pull request. Lessons 00 and 01 are available.

| Prompt | Lesson | Build or investigate | Status |
|---|---|---|---|
| 00 | Computation and learning | Computation graphs, derivatives, backpropagation, NumPy neural network, optimization and generalization | [Available](00_computation_and_learning.ipynb) |
| 01 | Equation learners (EQL) | Differentiable mathematical operators, sparsity and expression recovery | [Available](01_equation_learners.ipynb) |
| 02 | Programs as hypotheses | Typed grammars, a small interpreter and inverse semantics | Planned |
| 03 | Choosing hypotheses | Bayesian reasoning, minimum description length and uncertainty | Planned |
| 04 | Search and decisions | Bandits, UCT and Monte Carlo tree search | Planned |
| 05 | Neural-guided program search | Policy/value predictions, imitation and a guided search loop | Planned |
| 06 | Structure and execution embeddings | Representing trees, graphs and program behavior | Planned |
| 07 | Learning reusable abstractions | Library learning and reusable program components | Planned |
| 08 | Adaptation and compositional generalization | New task adaptation, evaluation splits and failure analysis | Planned |

The main project is neural-guided program search. A small equation learner comes
first to make differentiable structure concrete; embeddings later become a
component of the search system. These lessons develop research tools, not a claim
that any one architecture solves ARC.

## Run the notebooks

From the repository root, using Python 3.10 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r learn/vector_01/requirements.txt
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1` instead.
Open the notebook in VS Code or an existing Jupyter installation, select this
environment as the Python kernel, and run all cells from the top. JupyterLab users
can install it separately with `python -m pip install jupyterlab`.

Lessons 00 and 01 use NumPy and Matplotlib, generate their own data, and run without
TensorFlow, a GPU, network access, or a particular working directory. Executed
plots are included for reading before running it yourself. Allow 60–90 minutes
for the explanations and experiments.

Every lesson aims to connect visual intuition, equations, transparent code,
numerical checks, and exercises with separate solutions. Use the **Predict**
prompts before running cells; restart and run all after experiments to restore
the baseline.

For a fresh-kernel execution check from the repository root:

```bash
python - <<'PY'
from pathlib import Path
import nbformat
from nbclient import NotebookClient

path = Path('learn/vector_01/00_computation_and_learning.ipynb')
notebook = nbformat.read(path, as_version=4)
NotebookClient(notebook, timeout=120, kernel_name='python3').execute()
print('All cells executed successfully.')
PY
```

The notebook checks scalar and full-network derivatives against central finite
differences, verifies that hidden contributions reconstruct the prediction, and
checks that both training runs remain finite and reduce training loss.

## Lesson 01: equation learners

Build a small EQL-style network with identity, square, sine, and multiplication
operators. The notebook derives and checks its gradients, demonstrates L1 shrinkage
and fixed-mask pruning, exports the learned expression, and expands its polynomial
part without refitting. Eight figures connect the graph, operator shapes, training,
sparsity, recovered coefficients, and extrapolation.

The experiment selects among three predetermined seeds using validation loss, then
compares a frozen model with a quadratic least-squares baseline. It includes six
exercises with solutions, identifiability and grammar limitations, and primary EQL
references. Allow 75–120 minutes including exercises. No additional dependencies
are required. To use the execution command above, change the filename to
`01_equation_learners.ipynb`.
