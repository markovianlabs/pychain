# PyChain

A Python implementation of Markov Chain Monte Carlo (MCMC) sampling using the **Metropolis-Hastings** algorithm. PyChain makes it straightforward to perform Bayesian parameter estimation for any model you can express as a chi-square (negative log-likelihood).

---

## Features

- Metropolis-Hastings random-walk sampler with uniform priors
- Adaptive proposal covariance estimated from early good samples
- Full diagnostics: autocorrelation, effective sample size, Gelman-Rubin R-hat
- Burn-in and thinning utilities
- Corner plots of parameter posteriors (via [`corner`](https://corner.readthedocs.io))
- Clean abstract base class — subclass and override one method
- 86 tests, 0 warnings

---

## Installation

```bash
git clone https://github.com/creativeishu/pychain.git
cd pychain
pip install -r requirements.txt
```

Or install as an editable package:

```bash
pip install -e .
```

**Requirements:** Python ≥ 3.10, NumPy ≥ 1.24, Matplotlib ≥ 3.7, corner ≥ 2.2

---

## How it works

PyChain implements the **Metropolis-Hastings** algorithm:

1. Start at a random point in parameter space drawn from the uniform prior.
2. Propose a new point from a multivariate normal centred on the current point.
3. Accept the proposal with probability:

$$\alpha = \min\!\left(1,\ \exp\!\left(-\frac{\chi^2_\text{new} - \chi^2_\text{old}}{2}\right)\right)$$

4. Repeat until the desired number of accepted samples is collected.

The proposal covariance is adaptively estimated from the first good samples (those with $\chi^2 < \texttt{goodchi2}$), which improves mixing without user tuning.

---

## Quick start

Subclass `MCMC` and implement `chisquare()`:

```python
import numpy as np
from pychain import MCMC

class LinearFit(MCMC):
    def __init__(self):
        super().__init__(
            NumberOfParams=2,
            Mins=[0.0, 20.0],
            Maxs=[10.0, 30.0],
            SDs=[1.0, 2.0],
            alpha=0.2,
            TargetAcceptedPoints=5000,
        )
        self.X = np.linspace(-10, 10, 25)
        self.Y = 5.0 * self.X + 25.0 + np.random.normal(0, 3, 25)
        self.noise = 3.0

    def chisquare(self, params):
        y_model = params[0] * self.X + params[1]
        return float(np.sum(((self.Y - y_model) / self.noise) ** 2))

chain = LinearFit()
result = chain.MainChain()

print(f"Acceptance ratio : {result.acceptance_ratio:.3f}")
print(f"Best chi2        : {result.best_chi2:.3f}")
print(f"Best params [m,c]: {result.best_params}")
```

---

## Diagnostics

All diagnostics operate on `result.samples` — the `(N, D)` array of accepted parameter vectors.

```python
from pychain import diagnostics

samples = result.samples

# Discard first 500 samples (burn-in) and thin by 2
samples = diagnostics.burn_in(samples, n_burn=500)
samples = diagnostics.thin(samples, step=2)

# Autocorrelation function
acf = diagnostics.autocorrelation(samples, max_lag=50)

# Effective sample size per parameter
ess = diagnostics.effective_sample_size(samples)
print(f"ESS: {ess}")

# Gelman-Rubin R-hat (requires multiple independent chains)
chain2 = LinearFit()
result2 = chain2.MainChain()
r_hat = diagnostics.gelman_rubin([result.samples, result2.samples])
print(f"R-hat: {r_hat}")  # values near 1.0 indicate convergence

# Corner plot
fig = diagnostics.corner_plot(
    samples,
    param_names=["m", "c"],
    truths=[5.0, 25.0],
    output_path="posteriors.png",
)
```

---

## Running the examples

```bash
# Linear fit (y = mx + c)
python examples/linearfit.py

# Quadratic fit (y = ax^2 + bx + c)
python examples/quadraticfit.py

# Animated real-time visualisation
python animation/anim.py            # default settings
python animation/anim.py 0.5 0.01  # custom alpha and delay
```

---

## Running tests

```bash
pip install pytest pytest-cov
pytest
```

With coverage:

```bash
pytest --cov=pychain --cov-report=term-missing
```

---

## Project structure

```
pychain/
├── pychain/
│   ├── __init__.py       # Package exports
│   ├── mcmc.py           # MCMC base class + ChainResult
│   └── diagnostics.py    # Autocorrelation, ESS, R-hat, corner plots
├── examples/
│   ├── linearfit.py      # Linear model example
│   └── quadraticfit.py   # Quadratic model example
├── animation/
│   └── anim.py           # Animated MCMC visualisation
├── tests/
│   ├── test_mcmc.py
│   ├── test_examples.py
│   └── test_diagnostics.py
├── pyproject.toml
└── requirements.txt
```

---

## Authors

- Irshad Mohammed (creativeishu@gmail.com)
- Janu Verma (j.verma5@gmail.com)

## License

MIT — see [LICENSE.txt](LICENSE.txt).
