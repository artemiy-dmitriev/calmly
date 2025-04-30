# calmly: Cavity ALignment with Machine Learning

**Calmly** is a modular Python framework for aligning optical cavities using reinforcement learning and simulation. 

## Features

- Built-in simulator for optical cavity alignment
- Reinforcement learning environment compatible with Stable-Baselines3
- Modular design for swapping simulation and preprocessing components
- Full training pipeline: heuristic algorithms → dataset generation → behavioral cloning → RL fine-tuning
- Command-line interface for all major steps
- YAML-based configuration system
- Tools for evaluation, logging, and visualization

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourname/calmly.git
cd calmly
pip install -e .
```

Dependencies (see `pyproject.toml`) include:
- `numpy`, `scipy`, `matplotlib`
- `stable-baselines3`, `gymnasium`
- `torch`, `tqdm`, `PyYAML`

You can install optional packages for training and evaluation in headless or HPC environments.

## Documentation

Full documentation is available at:

📚 **[https://calmly.readthedocs.io](https://calmly.readthedocs.io)**

## Quickstart

An example project is included in the repository. It can be used as a template for user projects.

For a typical workflow check out the [example Jupyter notebook](example_project/calmly_example.ipynb)

## Contributing

Contributions are welcome! Please open an issue or submit a pull request if you'd like to improve the software or documentation.

## License

This project is licensed under the **GNU General Public License v3.0**. See the [LICENSE](LICENSE) file for details.