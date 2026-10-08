# Vector 01: from gradients to neural-guided program search

A visual, from-scratch learning sequence. Each numbered prompt corresponds to one
notebook and a separate pull request. All lessons 00–08 are available.

| Prompt | Lesson | Build or investigate | Status |
|---|---|---|---|
| 00 | Computation and learning | Computation graphs, derivatives, backpropagation, NumPy neural network, optimization and generalization | [Available](00_computation_and_learning.ipynb) |
| 01 | Equation learners (EQL) | Differentiable mathematical operators, sparsity and expression recovery | [Available](01_equation_learners.ipynb) |
| 02 | Programs as hypotheses | Typed grammars, a small interpreter and inverse semantics | [Available](02_programs_as_hypotheses.ipynb) |
| 03 | Choosing hypotheses | Bayesian reasoning, minimum description length and uncertainty | [Available](03_choosing_hypotheses.ipynb) |
| 04 | Search and decisions | Bandits, UCB1, UCT and Monte Carlo tree search | [Available](04_search_and_decisions.ipynb) |
| 05 | Neural-guided program search | Policy/value predictions, imitation and a guided search loop | [Available](05_neural_guided_program_search.ipynb) |
| 06 | Structure and execution embeddings | Representing trees, graphs and program behavior | [Available](06_structure_execution_embeddings.ipynb) |
| 07 | Learning reusable abstractions | Library learning and reusable program components | [Available](07_learning_reusable_abstractions.ipynb) |
| 08 | Adaptation and compositional generalization | New task adaptation, evaluation splits and failure analysis | [Available](08_adaptation_and_compositional_generalization.ipynb) |

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

Lessons 00–03 use NumPy and Matplotlib, generate their own data, and run without
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

## Lesson 02: programs as hypotheses

Build a typed grid language, immutable program trees, a recursive interpreter,
and a bottom-up enumerator. Nine figures show execution traces, syntax trees,
ambiguous rules, an informative query, behavioral compression, and exact inverse
constraints for a shift that loses information.

The notebook keeps competing explanations, separates query labels from a held-out
test, and completes a fixed program sketch by matching forward candidates against
backward constraints. It exhaustively checks the inverse relation for four-cell
rows and cross-checks sketch completion against direct execution. It includes
seven exercises with solutions and primary reading. Allow 75–120 minutes.

No additional dependencies are required. For the execution command above, use
`02_programs_as_hypotheses.ipynb` as the filename.

## Lesson 03: choosing hypotheses

Use exact inference over five small programs to connect Bayesian updates, noisy
likelihoods, prefix-free model codes, and minimum description length. Ten figures
show posterior changes, prior sensitivity, model-averaged predictions, uncertainty
decomposition, informative queries, and confident failure from missing hypotheses.

The notebook verifies direct versus log-space inference, code-length/MAP rankings,
and two equivalent information-gain calculations. A 4,000-task simulation checks
probability calibration under a matched synthetic generator. It includes eight
exercises with solutions and emphasizes the limits of inference over a restricted
candidate set. Allow 75–120 minutes; no additional dependencies are required.
Use `03_choosing_hypotheses.ipynb` in the execution command above.

## Lesson 04: search and decisions

Build an independent multi-armed bandit and compare uniform, greedy and UCB1
allocation across reproducible trials. Visualize the exploration bonus,
pseudo-regret, and sensitivity to its coefficient. Then implement a token
prefix-tree MCTS with selection, expansion, simulation and backpropagation.

The synthetic three-step program grammar is small enough to enumerate
exhaustively for correctness checks. Compare exact, shaped and prefix-based
rewards, display evolving root visits and distinct-program coverage, and
examine unequal execution costs and discounted lineage credit. A final section
separates textbook UCT from persistent cross-task Grammar-UCT and motivates
neural-guided search in lesson 05. The notebook includes nine exercises,
worked solutions and primary references. No new dependencies are required.
Run using `04_search_and_decisions.ipynb` in the execution command above.

## Lesson 05: neural-guided program search

Learn a task-conditioned policy over next grammar actions from synthetic solved
program traces; implement an explicit NumPy neural network and check its
softmax-cross-entropy gradients. Separately train a value network on a precisely
defined random-rollout success-probability target. Hold out whole task identities
before creating policy/value rows.

Integrate learned policy priors into a transparent PUCT tree search. Compare UCT
with uniform rollout, PUCT with uniform rollout, and PUCT with learned rollout
using equal complete-program execution budgets. Diagnose rare-value regression
failures, imitation/selection bias, policy miscalibration, and distribution shift.
The notebook includes 10 exercises with worked solutions, a direct mapping to
Grammar-UCT's operation/source decisions, and a bridge to lesson 06 embeddings.
No new dependencies. Use `05_neural_guided_program_search.ipynb` in the README's
notebook execution example.

## Lesson 06: structure and execution embeddings

Construct a self-contained AST language for seven-cell Boolean grids, visualize
operator trees, and implement their exact typed semantics. Compare bag-of-operators
counts, untrained recursive structural vectors, execution signatures from chosen
probes, and PCA compression. Use exhaustive evaluation over all 128 Boolean
inputs as a toy-domain semantic oracle, showing both true equivalences and
collisions induced by inadequate probe sets.

Generate diverse bounded ASTs, split programs by complete semantic family, and
train a NumPy neural execution emulator on structural-position features plus
input grids. Gradient-check its supervised BCE objective and evaluate separately
on unseen program families and unseen input grids. Inspect exact-grid vs cell
accuracy, learned pooled hidden vectors, retrieval disagreement, and a cached
shared-subtree execution DAG. End with exercises, solutions, research sources,
and an explicit bridge to Grammar-UCT's bundled genes and component paths.

The notebook has no new dependencies and uses only original synthetic tasks.
Run `06_structure_execution_embeddings.ipynb` with the README's Jupyter command.

## Lesson 07: learning reusable abstractions

Build a small, executable library learner from first principles. Match two-level
unary AST contexts with an ARG placeholder, compress programs into macro calls,
and expand calls exactly—including nested invocations—without mistaking them for
cyclic macro definitions. Verify expansion against all 128 possible binary inputs
in the toy width-seven Boolean domain.

Score candidates by a transparent, library-inclusive description-length proxy:
charge for macro definitions, all rewritten program nodes, and fixed entry
overhead. Train a greedy library only on a synthetic solved-program corpus, then
compare primitive-only breadth-first search with a macro-extended grammar on
held-out compositions at equal complete-program execution counts. Plot both
beneficial transfer and branching-factor negative transfer. Includes theoretical
caveats, 10 exercises and solutions, and explicit mappings to Grammar-UCT's
operations, source components and cached representations.

Run `07_learning_reusable_abstractions.ipynb` using the README's normal Jupyter
setup; no new dependencies are required.

## Lesson 08: adaptation and compositional generalization

Capstone for vector_01. Create a small exact-execution grammar of seven-bit
Boolean-grid transformations and group all 1,365 programs by their behavior
across the complete 128-input toy domain. Partition target transformations by
semantic identity and evaluate four distinct task-family shifts: unseen short
functions, a withheld legal operator transition, longer unseen compositions,
and their joint shift.

Train a smoothed Markov grammar prior only on short training solutions. Adapt it
using a **separate labeled support set** of six solved tasks. Compare uniform
candidate enumeration, the trained prior and the adapted prior at the same
complete candidate-execution budget. Record correct function discovery versus
mere demonstration agreement, candidate-execution curves, negative transfer,
support-count sensitivity, and ambiguity from too few input/output examples.

The notebook also shows a grammar-expressivity failure from a missing primitive,
plots descriptive uncertainty intervals, includes 12 exercises with worked
solutions, and proposes an evaluation protocol for a task-conditioned neural
Grammar-UCT search system. This toy benchmark does not establish ARC accuracy.
No new dependencies are required.

**Sequence complete:** lessons 00–08 now form a continuous curriculum from
calculus and neural networks through symbolic hypotheses, search, neural
policies, structural embeddings, reusable libraries, and adaptation. The next
step is to apply the pieces in a reproducible grammar-search experiment.

## Review and next project

Lessons 04–08 now include executed outputs and additional diagnostics. See
[NEXT_PROJECT.md](NEXT_PROJECT.md) for the implementation boundaries, verification
record, and a staged execution-guided synthesis project that connects the lessons.
