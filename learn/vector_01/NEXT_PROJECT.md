# After vector_01: build a small execution-guided synthesizer

The next useful step is one integrated experiment: **does observing intermediate
execution help a learned search policy solve unfamiliar compositions?** Build
enough infrastructure to answer that question, then expand toward your ARC engine.

The notebooks teach components. They do not yet form one trained solver:

| Lesson | What is implemented | Boundary to remember |
|---|---|---|
| 04 | UCB and a tiny MCTS tree | Tokens use a constructed reward table, not grid execution |
| 05 | Task-conditioned policy, separate value model, guided search | New contexts can share training target programs; the value model is not used in search backup |
| 06 | Exact AST interpreter, probe signatures, learned execution emulator | The recursive encoder is random; the trained model uses bounded positional features |
| 07 | Parameterized macros and greedy library compression | Handcrafted corpus, restricted templates, selected demonstration macro |
| 08 | Semantic-family splits and support-set adaptation | A bigram prior, not a neural meta-learner; transition and length shift are partly confounded |

## Build in four milestones

### 1. Turn the Boolean-grid lessons into one small environment

Start with the seven-cell rows and four unary operators from 08. Implement
`execute`, `legal_actions`, task generation, and a search result record as normal
Python modules. Keep a notebook for inspecting examples and plots. Define the
composition convention once: `('R', 'N')` executes R first, while the AST is
`NOT(RIGHT(INPUT))`.

Separate the solver from the evaluator. The solver gets demonstrations, legal
operations, and a budget. Only the evaluator sees the generating program and
hidden outputs. A candidate must pass exact interpretation; a network score is
never a substitute. Start with enumeration and a small bounded grammar where an
oracle can establish which tasks are expressible.

**Completion check:** enumeration recovers a behaviorally correct program for
every expressible target when given a sufficient budget and identifying examples.
With fewer examples, report consistent-but-wrong programs as a separate outcome.

### 2. Learn proposals before introducing a tree

Generate solved training tasks with their input/output examples and intermediate
states. Group behaviorally equivalent programs before splitting. Use a separate
validation pool for model selection and a final pool for evaluation.

Train a small TensorFlow policy to predict the next legal operation. A starting
input contains demonstrations, the current prefix, its depth, and the remaining
budget. Use the exact interpreter to compute intermediate rows. Compare greedy
prediction, policy-only sampling, and beam search with enumeration.

Account for equivalent solutions: the generating trace supplies a valid target,
but another program may be equally correct. Teacher-forced action accuracy is a
diagnostic, not the primary metric. Search will encounter prefixes outside those
training traces; record how the policy behaves there.

**Completion check:** produce a reproducible exact-solve-versus-budget plot with
at least one strong simple baseline. A result showing no gain is still useful.

### 3. Run one controlled representation experiment

Keep the task context, search algorithm, data, training budget, and model capacity
as comparable as possible. Change only the representation of the partial program:

| Variant | Partial-program information |
|---|---|
| Syntax | Operator prefix, depth, and legal-action information |
| Execution | Intermediate rows across demonstrations, depth, and legal actions |
| Combined | Both of the above |

Every variant receives the same input/output demonstrations. Otherwise a gain
could merely come from giving one variant more information about the task.

Use separate tests for new input patterns, new semantic functions, withheld
compositions at **matched lengths**, and longer compositions. A held-out bigram
is a statement about traces, not automatically a held-out semantic concept.
Generate and freeze the splits before fitting anything. For larger domains where
exact semantic grouping is impossible, document the weaker overlap checks.

Report exact hidden-output success, demonstration fit, spurious fits, candidate
executions, primitive operations, neural calls, and wall time. Fix several seeds;
summarize variation by task rather than treating repeated runs of the same task as
independent tasks. Keep training cost separate from per-query cost.

**Completion check:** explain which representation helps, on which split, at what
compute cost—and show a concrete failure where it does not help.

### 4. Add search and abstraction one at a time

Only after the simple baselines are measured, add PUCT using the same learned
policy. Compare against policy-only sampling and beam search. Then, in a separate
experiment, add one training-learned macro. Measure its expanded primitive cost
and its effect on unrelated tasks, not merely the reduction in program depth.

Reserve learned values, learned recursive encoders, reinforcement-learning loops,
and automated library growth for follow-up questions supported by observed
failures. Each changes a different part of the system.

**Completion check:** each additional component has an isolated benefit or a
documented negative result. Then replace the toy interpreter with a small adapter
over your ARC operation registry, preserving typed sources, component identity,
provenance, and exact execution.

## How to study while building

For each lesson: predict one plot, run it, explain the discrepancy, then recreate
one core function without looking. Use the notebook as a reference when stuck.
The useful milestone is being able to diagnose why a solver failed: insufficient
language, poor search allocation, misleading representation, ambiguous examples,
or distribution shift.

For a research bridge, read [ExeDec and its released code](https://github.com/google-deepmind/exedec).
It studies execution-informed synthesis and explicit compositional-generalization
benchmarks. The proposed experiment above is a manageable learning project, not a
claim to reproduce or improve that paper.

For LOGARITHM, a later commercial experiment could replace grid operations with
typed data-cleaning operations and learn repeatable transformations from input/
output examples. Validate an actual recurring customer workflow before extending
the research prototype into a product. This is a possible application, not market
validation or a reason to delay existing customer work.

## Review verification (2026-10-08)

All code cells in lessons 04–08 executed in a fresh IPython process per notebook,
with original training settings and assertions enabled: **77 code cells and 58
rendered figures**. Figures were inspected, and outputs are saved in the notebooks.
Separate Jupyter kernels could not start because this runtime blocks kernel
sockets; execution used in-process IPython and Matplotlib PNG capture instead.
Standard Jupyter Run All remains the intended local workflow.

The review fixed a syntax error in 05, a test-label-derived baseline in 06,
nonterminating identity-template rewriting in 07, and non-nested example-count
comparisons plus inconsistent chart scales/colors in 08. Added experiments make
tree backup, simple search baselines, and crossed generalization visible.

These checks establish that the teaching examples execute; they do not validate
ARC benchmark performance or exhaustive correctness of a production solver.
