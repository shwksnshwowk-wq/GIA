This code includes experiments for paper "Gradient Inversion Attack for Adapter-based Federated Large Language Models"

All experiments are done using Python 3.10 and PyTorch 2.3.1.

### Installation 

Now, let's configure the system:

- Install all dependencies: ``pip install requirements.txt``
  - Create the ``results`` directory to store results.

### Structure

Every directory with model names (BERT, GPT2, Qwen2) includes the main attack files. The ``auxiliary_code`` includes the utilities used in main files. All codes have been thoroughly commented. 