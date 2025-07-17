# Solving Combinatorial Problems using Reinforcement Learning
Combinatorial optimization problems are frequently characterized by being intractable for classical algorithms or prone to suboptimal solutions. Such problems naturally arise in logistics (e.g., vehicle routing), scheduling (e.g., job shop), network design (e.g., flow), finance (e.g., portfolio selection), and beyond. Even minor improvements in these domains can yield substantial benefits in cost, efficiency, or strategy. However, designing heuristics and metaheuristics to tackle their complexity is time-consuming.

**Reinforcement Learning** (RL) has excelled at sequential decision-making tasks in fields ranging from autonomous driving and industrial control to robotics, protein folding, theorem proving, and multiagent games such as chess and go, where it has achieved superhuman performance.

In this exemplar, we will focus on learning to use Reinforcement Learning for solving sequential combinatorial problems, where an optimal strategy involves taking specific actions in a sequence while also responding to a probabilistic setting (environment). Notably, Reinforcement Learning is able to learn the state and action space, so it is able to effectively search these spaces for optimal solutions as opposed to exhaustive searches in classical algorithms, without any heuristics that require expert knowledge to correctly derive.

We will start by implementing a foundational algorithm, Tabular Q Learning, then learn how to apply it in a pre-supplied environment, involving the famous **Monty Hall** problem, where we will also explore hyperparameter tuning and visualisation of training. After this, we will learn how you can apply RL to any problem space of interest by creating your own environment, where we will walk through an example implementing an environment from scratch for the seminal News Vendor problem from inventory management.

<!-- ![Scikit Camera Image](docs/assets/readme-img.png) -->

<!-- Author information -->
This exemplar was developed at Imperial College London by Omar Adalat in
collaboration with Dr. Diego Alonso Alvarez from Research Software Engineering and
Dr. Jesús Urtasun Elizari from Research Computing & Data Science at the Early Career
Researcher Institute.

## Learning Outcomes 🎓

After completing this exemplar, students will:

- Explain the core principles of Reinforcement Learning, and be able to identify when and where it is applicable to a problem space, with a particular focus on combinatorial problems for this project.
- Develop an implementation of a foundational algorithm, Tabular Q Learning, starting from basic principles and concepts.
- Gain the ability to perform experimental validation of the trained Reinforcement Learning algorithm, visualising the learning over training episodes.
- Design hyperparameter tuning configurations that can automatically be applied to retrieve the optimal set of hyperparameters.
- Generalise to non-supplied environments by learning how to create your own Reinforcement Learning environment, allowing you to apply Reinforcement Learning to any problem space that you are interested in.


<!-- Audience. Think broadly as to who will benefit. -->
## Target Audience 🎯
This exemplar is broadly applicable to anyone interested in solving sequential decision problems, which comes up ubiquitously across many domains and industries (e.g. protein synthesis, self-driving cars, planning & scheduling, and strategic games). Specifically, although we focus on sequential combinatorial problems which are a specific flavour of sequential decision problems, the underlying concepts are the same between both.

Our exemplar suitable for students, researchers and engineers alike, and academic prerequisite knowledge is not assumed, aside from some confidence in using Python.

<!-- Requirements.
What skills and knowledge will students need before starting?
e.g. ECRI courses, knowledge of a programming language or library...

Is it a prerequisite skill or learning outcome?
e.g. If your project uses a niche library, you could either set it as a
requirement or make it a learning outcome above. If a learning outcome,
you must include a relevant section that helps with learning this library.
-->
## Prerequisites ✅

### Academic 📚

- Basic familiarity with Python & basic programming skills is required to solve the exercises notebooks
- Some math background in the basics of set theory and probability theory is helpful but not required

### System 💻

- [Astral's uv](https://docs.astral.sh/uv/) Python package and project manager
- An integrated development environment (IDE) for developing and running Python, [Visual Studio Code (VS Code)](https://code.visualstudio.com/) with Python & Jupyter Notebook extensions is the easiest to set up and use. VS Code should automatically prompt you to install the required extensions that you need, but you can refer to here for the [Python extension](https://marketplace.visualstudio.com/items?itemName=ms-python.python) and the [Jupyter](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter) extension is available for notebook support.
- 10 GB of disk space

<!-- Quick Start Guide. Tell learners how to engage with the exemplar. -->
## Getting Started 🚀

1. Start by cloning the repository, either using the GitHub interface or Git directly (`git clone https://github.com/ImperialCollegeLondon/ReCoDE-Solving-Combinatorial-Problems-using-Reinforcement-Learning`).
2. Install Astral's `uv` if not already installed from the following URL: https://docs.astral.sh/uv/
3. Run the command `uv sync` in the cloned repository directory. This will install the correct version of Python (scoped to the directory under `uv`) and gather all dependencies needed.
4. Create a virtual environment under which the Jupyter Notebooks will run under, which will be scoped to the project directory. Simply run `uv venv --python 3.12`. When running any notebook, use the virtual environment created for Python 3.12 in the current directory's path, VS Code will give you a list selection of virtual environments to run under (you can also switch this in the top right of a notebook as of the time of writing).
5. Navigate to the four notebooks in the directory `/notebooks/` and complete them in order, running the exercises which will be checked against automated tests and checking the solutions if at any time you are stuck!

## Disciplinary Background 🔬
     
Reinforcement Learning is a powerful learning paradigm in Artificial Intelligence & Computer Science. While Deep Learning and general Machine Learning are very interesting, often the focus is on making a single isolated decision as in the tasks of classification or regression. Reinforcement Learning, which at the state-of-the-art level also uses Deep Learning for effective learning, is important to learn and master for solving more complex tasks that involve sequential decisions.

Specifically, as it solves sequential decision problems, it is incredibly useful in an interdisciplinary manner for various problems that arise such as the aforementioned: scheduling, protein synthesis, finance, autonomous vehicles and beyond.

<!-- Software. What languages, libraries, software you use. -->
## Software Tools 🛠️

- Python, with version and dependencies managed by Astral's uv
- Gymnasium, allowing to define custom RL environments which conform to a standard interface
- Weights & Biases, for experiment tracking best practices and hyperparameter tuning
- Pygame, for visualisation of the environments
- Matplotlib, for visualisation of training results by plotting charts and diagrams
- Pytest, for unit testing
- Jupyter Notebooks, for literate programming & interactive content

<!-- Repository structure. Explain how your code is structured. -->
## Project Structure 🗂️

Overview of code organisation and structure.

```
.
├── notebooks
│ ├── 1-intro-to-rl.ipynb
│ ├── 2-tabular-q-learning.ipynb
│ ├── 3-experiments.ipynb
│ ├── 4-custom-environment-news-vendor.ipynb
├── src
│ ├── environments
│ ├───── monty_hall
│       │          └── env.py
│       │          └── state.py
│       │          └── renderer.py
│       │          └── discrete_wrapper.py
│ ├───── news_vendor
│       │          └── env.py
│       │          └── state.py
│       │          └── renderer.py
│       │          └── discrete_wrapper.py
│ ├── rl
│   │   └── common.py
│   │   └── tabular_q_learning.py
├── docs
└── test
```

Code is organised into logical components:

- `notebooks` for tutorials and exercises
- `src` for core code
    - `monty_hall` provides the full implementation of the Monty Hall Gymnasium environment. This is something you are expected to import in for Notebook 3. However, later on you can explore this directory in terms of how everything is implemented, for example the discrete state space wrapper, action masking, and visualisation. It may be useful as a reference for any environments you create in the future!
    - `news_vendor` is a full reference/target implementation for Notebook 4.
    - ` rl` is a reference implementation for Notebook 2, particularly focused on Tabular Q Learning.
- `docs` for documentation
- `test` for testing scripts

<!-- Best practice notes. -->
## Best Practice Notes 📝

- Package (dependency) management and Python version management is provided by `uv`, which allows a perfectly replicable development environment
- Reference code is entirely documented and commented using [Google's Style of Python Docstrings](https://www.sphinx-doc.org/en/master/usage/extensions/example_google.html#example-google)
- Experiments are stored and tracked using [Weights & Biases](https://wandb.ai), which allows long-term access to results of experiments, accompanied by all necessary information to replicate such experiments such as hyperparameters

## Estimated Time ⏳

| Task                                                    | Time      |
| ------------------------------------------------------- | --------- |
| Notebook 1) Intro to RL                                 | 1.5 hours |
| Notebook 2) Tabular Q Learning                          | 5 hours   |
| Notebook 3) Experiments                                 | 2 hours   |
| Notebook 4) Custom environment: News Vendor             | 4 hours   |


<!-- Any references, or other resources. -->
## Additional Resources 🔗

* For building your Reinforcement Learning knowledge:
    * [Mastering Reinforcement Learning](https://gibberblot.github.io/rl-notes/index.html#), which is a book accompanied by videos, providing an excellent overview of the various Reinforcement Learning methods out there
    * [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html), a seminal book with its latest edition published in 2018, by Richard S. Sutton and Andrew G. Barto. This book is considered foundational, and both authors heavily contributed to Reinforcement Learning research and helped start the field. However, this book is more on the theoretical side.
    * [Spinning Up in Deep RL by OpenAI](https://spinningup.openai.com/en/latest/), which provides a great overview of the state-of-the-art methods (e.g. PPO and actor-critic methods), particularly with deep reinforcement learning.
        * If you are not familiar with Deep Learning, consider looking at:
            * [Dive into Deep Learning](https://d2l.ai/), free online book, with code accompanying each section
            * [fast.ai courses](https://www.fast.ai/), covering advanced deep learning methods from the foundations accompanied by practical implementations
* Additional combinatorial environments are available at:
    * [Jumanji](https://github.com/instadeepai/jumanji)
    * [or-gym](https://github.com/hubbs5/or-gym), or stands for Operations Research 
* Specifically for attaining better performance in combinatorial RL, you may want to investigate:
    * More advanced exploration methods, other than greedy-epsilon, starting with Boltzmann
    *  [Pointer Networks](https://proceedings.neurips.cc/paper_files/paper/2015/file/29921001f2f04bd3baee84a12e98098f-Paper.pdf), used by some methods such as [AlphaStar](https://deepmind.google/discover/blog/alphastar-mastering-the-real-time-strategy-game-starcraft-ii/)
    * [Stochastic Q Learning](https://arxiv.org/abs/2405.10310) for handling large action spaces
    * Abstraction methods for lowering the complexity of the state and action space

<!-- LICENCE.
Imperial prefers BSD-3. Please update the LICENSE.md file with the current year.
-->
## Licence 📄

This project is licensed under the [BSD-3-Clause license](LICENSE.md).
