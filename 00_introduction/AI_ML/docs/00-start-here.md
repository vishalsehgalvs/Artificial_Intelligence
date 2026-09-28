# Start here: environment and learning method

## Outcome

Think of a notebook as a worksheet that can run code one small box (a **cell**) at a time. This course lets you try examples on your Windows computer. First read the plain-language story, then predict a cell's output, run it, and compare. No paid service or powerful graphics card is needed for the early lessons. Some later experiments are marked optional.

Before installing anything, skim [the topic map](topic-map.md): it explains what each subject is for using a small everyday problem. The [course index](../README.md) then gives you the reading and notebook order.

## Install on Windows (PowerShell)

Install Python 3.11 or 3.12 from [python.org](https://www.python.org/downloads/) and check `py -V`. From the workspace root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[course]"
python -m ipykernel install --user --name ai-ml-course --display-name "AI ML Course"
python -m pip check
```

If 3.11 is unavailable, replace `-3.11` with `-3.12`. In VS Code choose the `.venv` interpreter and the **AI ML Course** notebook kernel. If script execution is restricted, run `.venv\Scripts\python.exe -m pip install -e ".[course]"` directly without activation. Do not change the system execution policy just for this course. `python -m jupyter notebook` starts a local notebook server if VS Code's notebook editor is unavailable. Optional PyTorch installation is platform-specific; use the [official selector](https://pytorch.org/get-started/locally/) for CPU or your GPU driver rather than assuming a particular CUDA build.
`pip check` checks whether installed packages have compatible requirements; it does **not** run the notebooks. Open the [first notebook](../notebooks/01-foundations/gradient_descent.ipynb) and run its code cells to check the actual lesson. The runnable labs include assertions; use `python -m pytest` after installing the course extra when tests are added to your checkout, and use the lab commands as the current smoke checks.

## Cost and resource labels

| Label       | Meaning                                                                               |
| ----------- | ------------------------------------------------------------------------------------- |
| Core        | No key, no GPU, no new dataset download after dependencies install                    |
| GPU         | Same concepts, faster computation with optional PyTorch accelerator                   |
| Download    | May pull a model, with disk/RAM/license requirements                                  |
| Hosted      | Requires your own provider account/key; may incur charges and send data to a provider |
| Side effect | Tools can affect a machine or account: isolate, approve, and inspect before running   |

The notebooks index lists the labels for each exercise. Dependencies require an initial internet connection unless already cached; _offline_ refers to execution after installation. Never commit credentials, private documents, notebook outputs containing secrets, or model weights. For hosted work, use synthetic examples and environment variables, not keys written in cells. No lesson requires installing OpenClaw or enabling a messaging channel.

## How to work through a lesson

1. Write a guess for the example's output; run it and explain any mismatch.
2. Change one assumption (split seed, feature scale, graph edge, mask); predict the effect and measure it.
3. Complete the exercise before reading the answer or hint.
4. Record a short experiment log: data, metric, baseline, change, result, limitation.

If this is your first time with Python, try this in a notebook code cell: `print(2 * 3)`. It prints `6`. Try `apples = 4` on one line and `print(2 * apples)` on the next; it prints `8`. A variable is just a named place to keep a value. An `assert 2 * apples == 8` line checks that a result is what we expect; a wrong result stops with an error. That is useful feedback, not a sign that you broke the computer.

Use the [source ledger](sources.md) when a product name or API changes. Results from toy data are demonstrations, not promises about the real world. Later lessons introduce symbols only _after_ their everyday meaning: $X$ is a table of inputs, $y$ is the known answer, $\hat y$ is a guessed answer and $\mathcal L$ measures mistakes. You do not need to memorize these now.

**Self-check:** If each apple costs $2, what should `print(2 * 4)` display? **8.** If you can predict that, you are ready for the first lesson. Terms such as "cross-validation" are explained in the ML chapter, not assumed here.

Next: [Foundations](01-foundations.md) and [notebook guide](../notebooks/README.md).
