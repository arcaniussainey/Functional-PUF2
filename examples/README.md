# PUF Experiment Examples

These scripts show how to use the immutable operation-spec pipeline for reproducible PUF experiments.

Run from the project root:

```bash
python examples/01_construct_crp_dataset.py
python examples/02_reliability_aging_experiment.py
python examples/04_register_custom_operation.py
```

`03_seeded_xor_attack.py` requires `evosax`, because it exercises the evolutionary attack path.
