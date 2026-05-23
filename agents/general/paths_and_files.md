# Repository Paths and File Placement Rules

This document defines the folder structure of this repository and the rules for placing files. Follow these conventions when creating, moving, or organizing any file.

---

## @/agents/

Purpose: Contains authoritative specifications, conventions, datasets definitions, idea descriptions, and general rules that all coding agents **must** read before writing any code.

Structure:
- `agents/datasets/` — dataset schemas, identities, column definitions, access helpers.
- `agents/ideas/` — idea and algorithm specifications (naming, parameters, formulas).
- `agents/general/` — repository-wide conventions and structural rules (this file).

Rule: Any document that defines a **general rule, convention, or specification** that agents must follow belongs here.

---

## @/ai_chats/

Purpose:
1. Store instruction files intended for AI Agents.
2. Share files between multiple agents working on related tasks.
3. Maintain a history of instructions previously assigned to agents.

Rule: When instructed to create an instruction or task description for another agent, create a markdown file in this folder.

---

## @/packages/

Purpose: Contains reusable libraries and modules that may be imported by multiple scripts or notebooks across the repository.

Rule: When instructed to create a reusable library or package, place it here.

---

## @/notebooks/

Purpose: Contains Jupyter notebooks organized by use case.

Sub-folder conventions:
- `notebooks/study/` — notebooks created for learning, research, or study purposes.
- `notebooks/tests/` — notebooks created for testing, experimenting, or validating something.
- Additional sub-folders may be created when a notebook's use case does not fit `study` or `tests`.

Rule: When instructed to create a notebook, determine its use case and place it in the appropriate sub-folder.

---

## @/scripts/

Purpose: Contains standalone scripts that perform specific tasks and should be preserved.

Sub-folder conventions:
- `scripts/test/` — test scripts.
- `scripts/tools/` — utility/tool scripts.
- Additional sub-folders may be added when explicitly mentioned.

Rule: When instructed to create a script, place it in this folder or the relevant sub-folder.

---

## File Placement Decision Logic

When you are instructed to create a file **without** an explicit filename or path, apply the following rules:

| Instruction type | Target location |
|---|---|
| Create a notebook for testing | `@/notebooks/tests/<meaningful_name>.ipynb` |
| Create a notebook for study/research | `@/notebooks/study/<meaningful_name>.ipynb` |
| Create a reusable library/package | `@/packages/<meaningful_name>/` |
| Create a script or tool | `@/scripts/<sub_folder>/<meaningful_name>.py` |
| Create instructions for another agent | `@/ai_chats/<meaningful_name>.md` |
| Define a general rule or specification | `@/agents/<category>/<meaningful_name>.md` |

Always choose a **meaningful, descriptive filename** that reflects the content or purpose of the file.
