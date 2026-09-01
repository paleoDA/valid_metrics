# valid_metrics

A collection of validation metrics for climate field reconstructions currently including:

- KD Test (Harris et al. 2021)
- EOF Analysis w/ xeof
- Superposed Epoch Analysis
- volcanic response


## Installation

From the repository root:

```bash
python -m pip install -e .
```

Then the metric modules can be imported from anywhere in the environment:

```python
from metrics import KD_test
from metrics import EOF
from metrics import SEA
from metrics import volcano
```
