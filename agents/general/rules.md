# General Rules

This document defines mandatory rules that all AI agents **must** follow when performing any task in this repository. Rules will be added progressively over time.

---

## Dependency Management

### Jupyter Notebooks

When creating or editing a Jupyter notebook (`.ipynb`), **all** packages required by the code—whether imported directly or used as transitive dependencies—must be installed inline using `%pip install` at the top of the notebook before any import statements.

Example pattern (first code cell):

```python
%pip install pandas numpy scikit-learn
```

> Do **not** assume packages are pre-installed. Always include an explicit `%pip install` cell so the notebook is self-contained and reproducible.

### Python Scripts and Modules

When creating any Python file (`.py`) that depends on external packages, a `requirements.txt` file **must** be placed in the same directory as the script (or at the root of the relevant sub-folder if multiple scripts share the same dependencies).

The `requirements.txt` must list every external package required to run the script(s) in that folder.

---

*End of rules — additional rules will be appended below as needed.*
