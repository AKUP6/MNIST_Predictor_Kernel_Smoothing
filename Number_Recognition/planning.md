# Number Recognition: Planning

Kernel ridge regression on random ReLU features, MNIST.

## Model summary
- Input: 784 pixels, scaled to [0, 1], centered with the training mean
- Hidden layers: 2, ReLU, Gaussian init N(0, 2/fan_in), frozen
- Features: Phi = relu(relu(X W1 + b1) W2 + b2)
- Kernel: RBF on Phi, sigma from the median heuristic
- Solve: alpha = (K + lambda I)^-1 Y  (Y one-hot, n x 10)
- Predict: argmax(K_test @ alpha)

## File layout (your proposal, with my filenames filled in)

```
Number_Recognition/
├── planning.md
├── main.py                      # pulls everything together: load -> features -> fit -> evaluate
├── data/                        # raw MNIST files (not code)
└── helpers/
    ├── __init__.py
    ├── fixed_params.py          # frozen Gaussian W1,b1,W2,b2
    ├── network.py               # forward pass: X -> Phi
    ├── mnist_data.py            # load / flatten / scale / center / one-hot
    └── kernel_ridge/
        ├── __init__.py
        ├── kernel_utils.py      # rbf kernel, median-heuristic sigma, solve
        └── kernel_ridge.py      # fit(Phi, Y) -> alpha ; predict(Phi_test) -> scores
```

### What each file does
| File | Responsibility |
|------|----------------|
| `main.py` | One clean top-level function. Calls the helpers in order, sweeps sigma/lambda on a validation split, reports test accuracy once. |
| `helpers/fixed_params.py` | Draws W1, b1, W2, b2 from N(0, 2/fan_in) with a seeded rng. Never trained. |
| `helpers/network.py` | `features(X, params)`: relu(relu(X W1 + b1) W2 + b2). |
| `helpers/mnist_data.py` | Everything about the digits: load, flatten to 784, /255, subtract the training mean, one-hot labels, subset and split. |
| `helpers/kernel_ridge/kernel_utils.py` | `rbf(A, B, sigma)` with no loops, `median_sigma(Phi)`, and the linear solve. |
| `helpers/kernel_ridge/kernel_ridge.py` | Packages the above: fit builds K, solves for alpha, and stores alpha and the training Phi. Predict builds K_test and returns K_test @ alpha. |

## Feedback

Feedback is in the section at the bottom of this file.

## Open questions / decisions
- [ ] Where does the MNIST data live (and who downloads it)?
- [ ] Subset size n for the first run (suggest 5,000)
- [ ] Hidden sizes h1, h2 (suggest 512, 256)
- [ ] How to tune sigma and lambda (validation split)

## Feedback from Claude

**What's good**
- The split matches the math. Frozen random part (fixed_params + network) and learned part (kernel_ridge) are separate, so you can swap either independently, for example trying backprop later.
- Keeping main.py thin and pushing everything into helpers is the right call.

**Things I interpreted. Correct me if wrong**
1. "Fixed biases (including the last one)": I read this as the frozen parameters, meaning both W and b for both layers. I assumed "including the last one" means the second hidden layer's W2, b2. If you meant a frozen output layer, that doesn't exist here: the output is alpha, which is fit.
2. "Dynamic biases": I read this as alpha (n x 10), the learned part. Strictly, alpha isn't a bias. It's one coefficient per training point per digit. The folder name can still be whatever you like.
3. "The existing number shit": I read this as the MNIST loading and preprocessing, which became `mnist_data.py`.

**Suggested changes**
- Add `data/` for the raw MNIST files, so data stays out of code folders.
- Add `__init__.py` in `helpers/` and `helpers/kernel_ridge/` so the imports work.
- Name the folder and file differently. `kernel_ridge/kernel_ridge.py` makes imports like `from helpers.kernel_ridge.kernel_ridge import ...` confusing. Use the folder name `kernel/` or the file name `model.py`.
- Put evaluation (accuracy, validation sweep) in main.py for now. Move it to its own `helpers/evaluate.py` only if main.py gets long.
- `fixed_params.py` should take a seed and sizes as arguments. Don't hard-code them, or you can't compare runs.
- Predict must use the **same** centering mean and the **same** W/b as training. Return or store them together (a small dict or tuple) so they can't drift apart.

**Still open**
- Where does the data come from? Your earlier "2": did you mean the synthetic fallback instead of downloading MNIST?
