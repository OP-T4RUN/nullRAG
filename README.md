# NullRAG: Hierarchical Multimodal RAG for Financial Document Understanding

**NullRAG—Hierarchical Multimodal RAG for Financial Document Understanding**

---

## Overview

NullRAG is a specialized Retrieval-Augmented Generation (RAG) framework designed for complex financial documents. It addresses the challenges of tabular data, cross-referencing, and long-range dependencies in financial reports (e.g., 10-Ks, Annual Reports) by introducing:

1.  **Hierarchical Parsing**: Structured decomposition of documents into Sections, Paragraphs, Tables, and Cells.
2.  **Table-Text Graph Neural Network (TTGNN)**: A graph-based module to capture relationships between textual content and tabular data.
3.  **Symbolic-Neural Fusion**: A reasoning engine that combines LLM generation with precise symbolic execution for numerical accuracy.

## Installation

Ensure you have Python 3.13+ installed.

1.  Clone the repository:
    ```bash
    git clone https://github.com/vinhqdang/NullRAG.git
    cd NullRAG
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: We recommend installing PyTorch with CUDA support first if you have a GPU.*

## Usage

### 1. Run the Main Pipeline
To see the end-to-end processing from document parsing to reasoning:

```bash
python run_pipeline.py
```
This script demonstrates:
- Mocking a financial document (JSON structure).
- Parsing sections and tables.
- Building the heterogeneous graph.
- Running the TTGNN and Symbolic-Neural Fusion engine.

### 2. Generate Figures and Results
To reproduce the experimental results and figures mentioned in the paper:

```bash
python run_demo.py
```
This will generate:
- Performance comparison plots (`results/Fig1_Main_Performance.png`, etc.)
- Ablation study tables (`results/Table1_Ablation.md`)
- Qualitative comparison logs (`results/Qualitative_Comparison.md`)

## Repository Structure

- `nullrag/` (also aliased as `hierfinrag/`): Core package containing the implementation.
  - `parsing/`: Document layout analysis and JSON parsing.
  - `graph/`: Graph construction and TTGNN model.
  - `reasoning/`: Symbolic execution and fusion logic.
- `data/`: Directory for input documents and intermediate parsed data.
- `results/`: Output directory for experiments and visualizations.

## License

This project is licensed under the GPL 3.0 License - see the [LICENSE](LICENSE) file for details.
