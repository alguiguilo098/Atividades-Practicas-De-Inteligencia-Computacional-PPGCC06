# Template Matching - Four Shapes

This project implements template matching for shape recognition using four different shapes.

## Project Setup

### 1. Prerequisites
- Python 3.x
- virtualenv (install using: `pip install virtualenv`)

### 2. Virtual Environment Setup
1. Create a virtual environment:
    ```bash
    python -m venv .env
    ```

2. Activate the virtual environment:
    - On Windows:
    ```bash
    .env\Scripts\activate
    ```
    - On Unix or MacOS:
    ```bash
    source .env/bin/activate
    ```

3. Install Jupyter plugin (if not already installed) in VSCode:


### 3. Dataset Preparation

#### Decompress the datasets
1. For Leaves dataset:
     ```bash
     cd ./Folhas
     unzip Folhas.zip
     ```

2. For Shapes dataset:
     ```bash
     cd ./Formas
     unzip Formas.zip
     ```

### 4. Directory Structure
After decompressing, your project should have the following structure:

Atv-01-Template-Matching-Four-Shapes
    ├── Folhas
    │   └── Folhas.zip
    ├── Formas
    │   └── fourShapes.tar.gz
    ├── main.ipynb
    ├── Projeto1 - IC-2025-2.pdf
    ├── Readme.md
    └── Utils.py

### 5. Running the Project

1. Open `main.ipynb` in Jupyter Notebook or JupyterLab.
2. Run the notebook cells sequentially to execute the template matching process.
3. Follow the instructions and comments within the notebook for further guidance.
4. Ensure that the paths to the datasets are correctly set in the notebook.
5. Visualize the results as per the notebook instructions.