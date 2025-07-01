# itds420-dsa-assessment
ITDS420 Assessment: Implementation and analysis of a parallel Dijkstra's algorithm for efficient shortest path computation on massive graph datasets.

# Scalable Shortest Path Finding: Parallel Dijkstra's Algorithm

![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Coverage](https://img.shields.io/badge/coverage-59%25-brightgreen) <!-- Placeholder: Replace with actual coverage badge -->
[![Python CI](https://github.com/KaungSettLwin/itds420-dsa-assessment/actions/workflows/ci.yml/badge.svg)](https://github.com/KaungSettLwin/itds420-dsa-assessment/actions/workflows/ci.yml)

## Project Description

This repository contains a Master's level assessment project for the "Advanced Data Structures and Algorithms" course (ITDS420). The core of this project is the design, implementation, and analysis of a **scalable parallel Dijkstra's algorithm** to efficiently find the shortest path from a source node to all other reachable nodes in very large graphs (specifically, graphs with at least 100,000 nodes).

The project demonstrates:
* In-depth understanding of foundational data structures (Graphs, Priority Queues) and algorithmic paradigms.
* Application of software engineering principles for robust and maintainable code.
* Strategies for performance optimization and parallel computing.
* Comprehensive testing and benchmarking methodologies.

## Problem Statement

The primary objective is to **determine the shortest path from a given source node to all other reachable nodes in a graph containing at least 100,000 nodes, leveraging parallel computing principles for optimal performance.** The solution must be computationally efficient and minimize resource consumption, addressing the complexities introduced by graph size.

**Input:**
* A directed or undirected graph $G = (V, E)$ with non-negative edge weights.
* Number of vertices $|V| \ge 100,000$.
* A specified source node $s \in V$.

**Output:**
* The shortest distance $d(s, v)$ from $s$ to $v$ for each reachable node $v \in V$.

## Installation

To set up the project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/KaungSettLwin/itds420-dsa-assessment.git](https://github.com/KaungSettLwin/itds420-dsa-assessment.git)
    cd itds420-dsa-assessment
    ```
    

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    ```

3.  **Activate the virtual environment:**
    * **On macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```
    * **On Windows:**
        ```bash
        .\venv\Scripts\activate
        ```

4.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Create a `requirements.txt` file in your project root with `heapq`, `collections`, `concurrent.futures`, `time`, `random` if you used them explicitly, though most are built-in. For `pytest` and `pytest-cov`, add them to `requirements.txt`)*
    ```
    # requirements.txt
    pytest
    pytest-cov
    matplotlib # If you plan to generate plots
    seaborn    # If you plan to generate plots
    ```

## Usage

To run the parallel Dijkstra's algorithm example:

1.  **Ensure your virtual environment is activated.**
2.  **Execute the main script:**
    ```bash
    python dijkstra_solver.py
    ```


    This will run the example graph and print the shortest distances, as demonstrated in the `if __name__ == "__main__":` block of your code.

## Testing

The project includes unit tests using `pytest` and measures code coverage with `pytest-cov`.

1.  **Ensure your virtual environment is activated.**
2.  **Run tests and generate a coverage report:**
    ```bash
    pytest --cov=dijkstra_solver --cov-report=html
    ```

    This command will:
    * Run all tests in the `test_dijkstra.py` file (assuming you named your test file this way).
    * Generate an HTML coverage report in a new directory named `htmlcov/`.

3.  **View the coverage report:**
    Open `htmlcov/index.html` in your web browser to see a detailed breakdown of code coverage.

## Benchmarking

To generate real performance benchmarks and plots (as discussed in Section 5.2 of the assessment), you would:

1.  **Implement the graph generation and timing logic** within your `dijkstra_solver.py` or a separate benchmarking script. An example snippet is provided in the code comments.
2.  **Run the benchmarking script:**
    ```bash
    python dijkstra_solver.py # If benchmark logic is in main block
    # OR
    python benchmark_script.py # If separate script
    ```
3.  **Analyze the output** and use a plotting library (e.g., `matplotlib`, `seaborn`) to create visual charts.
4.  **Embed the generated plots** (e.g., `speedup_plot.png`, `time_vs_nodes.svg`) directly into your assessment document.



