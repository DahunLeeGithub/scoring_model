# pytorch scoring models

## Setup and Execution Guide

This guide provides instructions to set up and run the project in a virtual environment on Windows, macOS, and Linux.

---

## Prerequisites

- Python 3.11 or later
- `pip` (Python package manager)

---

## Setup Instructions

### Windows

```sh
# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### macOS/Linux

```sh
# Create a virtual environment
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Running the Project

### Windows

```sh
.venv\Scripts\activate
python main.py
```

### macOS/Linux

```sh
source .venv/bin/activate
python main.py
```

---

## Deactivating the Virtual Environment

```sh
deactivate
```

---

## Additional Notes

- If `venv` is not found, install it using:

  ```sh
  pip install virtualenv
  ```

- If using `bash` or `zsh`, you may need to give execution permissions:

  ```sh
  chmod +x .venv/bin/activate
  ```
