import time
import csv
import os
import random
import math
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import heapq # Make sure this import is present and at the top
import itertools # Added for PriorityQueue's unique ID generation
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

# --- 1. Graph Data Structure ---
class Graph:
    """
    Represents a graph using an adjacency list.
    Supports adding nodes and weighted edges.
    """
    def __init__(self):
        """
        Initializes an empty graph.
        Adjacency list stores neighbors and their weights: {node: {neighbor: weight}}
        """
        self.adj_list = defaultdict(dict)
        self.nodes = set()

    def add_edge(self, u: int, v: int, weight: float):
        """
        Adds a directed edge from node u to node v with a given weight.
        Raises ValueError if weight is negative.
        """
        if weight < 0:
            raise ValueError(f"Edge weight must be non-negative. Got {weight} for edge ({u}, {v}).")
        self.adj_list[u][v] = weight
        self.nodes.add(u)
        self.nodes.add(v)

    def get_neighbors(self, u: int) -> dict:
        """
        Returns a dictionary of neighbors and their weights for a given node.
        """
        return self.adj_list.get(u, {})

    def get_nodes(self) -> set:
        """
        Returns a set of all nodes in the graph.
        """
        return self.nodes

    def __len__(self) -> int:
        """
        Returns the number of nodes in the graph.
        """
        return len(self.nodes)

    def __contains__(self, node: int) -> bool:
        """
        Checks if a node exists in the graph.
        """
        return node in self.nodes

# --- 2. Priority Queue Data Structure ---
class PriorityQueue:
    """
    A min-priority queue implementation using Python's heapq module.
    Stores (priority, item) tuples.
    This version is *not* thread-safe or process-safe by itself.
    For parallel Dijkstra, a shared/distributed PQ or careful synchronization is needed.
    In this example, it's used sequentially by the main process.
    """
    def __init__(self):
        """Initializes an empty priority queue."""
        self._queue = []
        self._entry_finder = {}  # Maps item to its entry in the queue (priority, entry_id, active_flag, item)
        self._counter = 0        # Unique sequence count for tie-breaking

    def add_task(self, item: int, priority: float):
        """
        Add a new item or update the priority of an existing item.
        If the item exists, its priority is updated. If the new priority
        is higher than the current one, the item's position in the queue
        is effectively updated (a new entry is added, old one marked removed).
        """
        if item in self._entry_finder:
            # Mark old entry as removed by setting its active_flag to False
            self._entry_finder[item][2] = False
        
        count = self._counter
        self._counter += 1
        entry = [priority, count, True, item] # [priority, entry_id, active_flag, item]
        self._entry_finder[item] = entry
        heapq.heappush(self._queue, entry)

    def pop_task(self) -> int:
        """
        Remove and return the lowest priority item.
        Raises KeyError if the queue is empty.
        """
        while self._queue:
            priority, count, active, item = heapq.heappop(self._queue)
            if active: # Check if the entry is still active (not marked as removed)
                del self._entry_finder[item]
                return item
        raise KeyError('pop from an empty priority queue')

    def is_empty(self) -> bool:
        """Checks if the priority queue is empty."""
        return not self._entry_finder # Check if entry_finder (active tasks) is empty

# --- 3. Parallel Dijkstra's Algorithm ---

# This helper function for edge relaxation will run in separate processes
def _relax_edge_worker(args):
    """
    Helper function to relax an edge (u, v) and return potential updates.
    Designed to be run by a multiprocessing worker.
    
    Args:
        args (tuple): A tuple containing:
            (u, v, weight, current_distances_u, current_distances_v)
            - u (int): Source node of the edge.
            - v (int): Destination node of the edge.
            - weight (float): Weight of the edge (u, v).
            - current_distances_u (float): Current shortest distance to node u.
            - current_distances_v (float): Current shortest distance to node v.
            
    Returns:
        tuple: (v, new_distance) if an update occurs, otherwise None.
    """
    u, v, weight, current_distances_u, current_distances_v = args
    new_distance = current_distances_u + weight
    if new_distance < current_distances_v:
        return (v, new_distance)
    return None

def parallel_dijkstra(graph: Graph, start_node: int, num_processors: int = 4) -> dict:
    """
    Finds the shortest path from a start_node to all other nodes
    in a non-negatively weighted graph using a conceptual parallel Dijkstra approach
    with multiprocessing.
    
    Args:
        graph (Graph): The graph object.
        start_node (int): The starting node for shortest path calculation.
        num_processors (int): Number of separate processes to use for parallel edge relaxation.
    
    Returns:
        dict: A dictionary where keys are nodes and values are their shortest distances
              from the start_node. Returns float('inf') for unreachable nodes.
    
    Raises:
        ValueError: If start_node is not in the graph or num_processors is invalid.
    """
    if start_node not in graph:
        raise ValueError(f"Start node {start_node} not found in graph.")
    if num_processors <= 0:
        raise ValueError("Number of processors must be a positive integer.")

    # Distances dictionary is shared conceptually, but updates are managed by the main process
    # based on results from parallel workers.
    distances = {node: float('inf') for node in graph.get_nodes()}
    distances[start_node] = 0
    
    pq = PriorityQueue()
    pq.add_task(start_node, 0)
    
    # Use ProcessPoolExecutor for true CPU-bound parallelism
    with ProcessPoolExecutor(max_workers=num_processors) as executor:
        while not pq.is_empty():
            try:
                current_node = pq.pop_task()
            except KeyError:
                # Should not happen if loop condition is correct, but good for robustness.
                break 

            # If we find a shorter path to an already processed node, skip it
            # This check is implicitly handled by PriorityQueue.add_task
            # which updates existing entries or adds new ones if priority improves.

            # Prepare tasks for parallel edge relaxation
            relaxation_tasks = []
            for neighbor_node, weight in graph.get_neighbors(current_node).items():
                relaxation_tasks.append((
                    current_node, 
                    neighbor_node, 
                    weight, 
                    distances[current_node], 
                    distances[neighbor_node] # Pass current neighbor distance for comparison
                ))
            
            # Submit tasks to the process pool
            # `executor.map` is suitable here as order of results doesn't strictly matter
            # and it handles argument passing for each task.
            # Alternatively, `executor.submit` with futures could be used for more fine-grained control.
            
            # Process results from parallel workers
            # Each worker returns (v, new_distance) or None
            for result in executor.map(_relax_edge_worker, relaxation_tasks):
                if result is not None:
                    v, new_distance = result
                    # Update distances and priority queue in the main process
                    # This ensures thread-safety for 'distances' and 'pq'
                    if new_distance < distances[v]: # Re-check to handle potential concurrent updates
                        distances[v] = new_distance
                        pq.add_task(v, new_distance)

    return distances

# --- 4. Graph Generation Function (For generating graphs for benchmarks) ---
def generate_random_graph(num_nodes: int, num_edges: int, max_weight: int = 100) -> Graph:
    """
    Generates a random graph with specified number of nodes and edges.
    Edges and weights are random. Graph can be disconnected.
    
    Args:
        num_nodes (int): The total number of nodes in the graph.
        num_edges (int): The total number of edges to add to the graph.
        max_weight (int): Maximum possible weight for an edge.
        
    Returns:
        Graph: A newly created Graph object.
    """
    g = Graph()
    # Ensure all nodes exist, even if they have no edges
    for i in range(num_nodes):
        g.nodes.add(i)
        
    # Add random edges
    for _ in range(num_edges):
        u = random.randint(0, num_nodes - 1)
        v = random.randint(0, num_nodes - 1)
        # Ensure no self-loops and avoid duplicate edges if graph is undirected
        if u == v:
            continue
        
        weight = random.uniform(1, max_weight) # Random float weights
        g.add_edge(u, v, weight)
        # If you want an undirected graph, add the reverse edge too:
        # g.add_edge(v, u, weight)  
        
    return g

# --- Main Execution Block (Benchmark Running and CSV/Plot Saving Logic) ---
if __name__ == "__main__":
    print("--- Parallel Dijkstra Implementation Example ---")

    # Create a graph
    g = Graph()
    g.add_edge(0, 1, 4)
    g.add_edge(0, 7, 8)
    g.add_edge(1, 2, 8)
    g.add_edge(1, 7, 11)
    g.add_edge(2, 3, 7)
    g.add_edge(2, 8, 2)
    g.add_edge(2, 5, 4)
    g.add_edge(3, 4, 9)
    g.add_edge(3, 5, 14)
    g.add_edge(4, 5, 10)
    g.add_edge(5, 6, 2)
    g.add_edge(6, 7, 1)
    g.add_edge(6, 8, 6)
    g.add_edge(7, 8, 7)

    start_node = 0
    print(f"\nFinding shortest paths from node {start_node} using 4 processors:")
    try:
        shortest_paths = parallel_dijkstra(g, start_node, num_processors=4)
        for node, distance in sorted(shortest_paths.items()):
            print(f"Distance from {start_node} to {node}: {distance}")
    except ValueError as e:
        print(f"Error: {e}")

    # Example with an empty graph
    print("\n--- Testing with an empty graph ---")
    empty_graph = Graph()
    try:
        shortest_paths_empty = parallel_dijkstra(empty_graph, 0)
    except ValueError as e:
        print(f"Expected error for empty graph: {e}")

    # Example with a disconnected graph (node 9 is isolated)
    print("\n--- Testing with a disconnected graph ---")
    disconnected_graph = Graph()
    disconnected_graph.add_edge(0, 1, 1)
    disconnected_graph.add_edge(1, 2, 1)
    disconnected_graph.add_edge(3, 4, 1) # Disconnected component
    disconnected_graph.nodes.add(9) # Isolated node
    
    start_node_disconn = 0
    try:
        shortest_paths_disconn = parallel_dijkstra(disconnected_graph, start_node_disconn)
        for node, distance in sorted(shortest_paths_disconn.items()):
            print(f"Distance from {start_node_disconn} to {node}: {distance}")
    except ValueError as e:
        print(f"Error: {e}")

    print("\n--- Running Performance Benchmarks ---")

    # Define graph sizes and processor counts for benchmarking
    graph_configs = [
        {"nodes": 1000, "edges_factor": 10},  # 1k nodes, 10k edges
        {"nodes": 5000, "edges_factor": 10},  # 5k nodes, 50k edges
        {"nodes": 10000, "edges_factor": 10}, # 10k nodes, 100k edges
        # {"nodes": 50000, "edges_factor": 10}, # Uncomment for larger tests (may take time)
        # {"nodes": 100000, "edges_factor": 10}, # Uncomment for even larger tests (may take significant time)
    ]
    processor_counts = [1, 2, 4, 8] # Test with 1, 2, 4, 8 processors/cores

    benchmark_results = []

    # Create output directory if it doesn't exist
    output_dir = "benchmark_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    csv_file_path = os.path.join(output_dir, "dijkstra_benchmark_results.csv")
    print(f"Benchmark results will be saved to: {csv_file_path}\n")

    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write CSV header.
        writer.writerow(["Nodes", "Edges", "Processors", "Execution Time (s)", "Speedup vs. P=1"])

        baseline_times = {} # To store baseline times for P=1

        for config in graph_configs:
            num_nodes = config["nodes"]
            num_edges = num_nodes * config["edges_factor"]
            
            print(f"--- Benchmarking for Nodes: {num_nodes}, Edges: {num_edges} ---")
            
            # Generate graph. (It's better to generate a new graph for each benchmark run)
            graph = generate_random_graph(num_nodes, num_edges)
            
            # Select a random start node for consistency
            start_node_for_benchmark = random.randint(0, num_nodes - 1) if num_nodes > 0 else 0

            # Run for each processor count
            for P in processor_counts:
                print(f"  Running with P={P}...")
                try:
                    start_time = time.perf_counter()
                    # Call the parallel Dijkstra function
                    _ = parallel_dijkstra(graph, start_node_for_benchmark, num_processors=P)
                    end_time = time.perf_counter()
                    execution_time = end_time - start_time
                    
                    # Store results temporarily
                    benchmark_results.append({
                        "Nodes": num_nodes,
                        "Edges": num_edges,
                        "Processors": P,
                        "Execution Time (s)": execution_time
                    })
                    print(f"    Completed in {execution_time:.4f} seconds.")
                except Exception as e:
                    print(f"    Error running for V={num_nodes}, P={P}: {e}")
                    benchmark_results.append({
                        "Nodes": num_nodes,
                        "Edges": num_edges,
                        "Processors": P,
                        "Execution Time (s)": float('nan') # Not a number for errors
                    })
            print("-" * 40) # Separator for different graph sizes

        print("\n--- Benchmark Results Summary (before speedup calculation) ---")
        # Convert results to a pandas DataFrame for easier plotting and speedup calculation
        df_results = pd.DataFrame(benchmark_results)
        print(df_results.to_string()) # Print the raw DataFrame

        # Calculate speedup for P > 1 relative to P=1 for the same graph size
        # Ensure 'Speedup vs. P=1' column exists before calculating
        df_results["Speedup vs. P=1"] = float('nan') 

        for num_nodes in df_results["Nodes"].unique():
            subset = df_results[df_results["Nodes"] == num_nodes].copy() # Use .copy() to avoid SettingWithCopyWarning
            baseline_row = subset[subset["Processors"] == 1]
            
            if not baseline_row.empty and baseline_row["Execution Time (s)"].iloc[0] > 0:
                baseline_time = baseline_row["Execution Time (s)"].iloc[0]
                df_results.loc[df_results["Nodes"] == num_nodes, "Speedup vs. P=1"] = \
                    baseline_time / df_results.loc[df_results["Nodes"] == num_nodes, "Execution Time (s)"]
            else:
                # Handle cases where baseline (P=1) is missing or 0
                df_results.loc[df_results["Nodes"] == num_nodes, "Speedup vs. P=1"] = float('nan')


        # Write final results (with speedup) to CSV
        writer.writerow([]) # Add an empty row for separation
        writer.writerow(["Final Results (with Speedup)"])
        writer.writerow(df_results.columns.tolist()) # Write columns as header
        for index, row in df_results.iterrows():
            writer.writerow([
                row["Nodes"],
                row["Edges"],
                row["Processors"],
                f"{row['Execution Time (s)']:.4f}" if not math.isnan(row['Execution Time (s)']) else "N/A",
                f"{row['Speedup vs. P=1']:.2f}x" if not math.isnan(row['Speedup vs. P=1']) else "N/A"
            ])

    print(f"\nAll benchmark results (including speedup) have been successfully saved to: {csv_file_path}")

    # --- Plotting the Results ---
    print("\nGenerating plots...")
    sns.set_style("whitegrid") # Set plot style

    # Plot 1: Execution Time vs. Number of Nodes for different Processors
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_results, x="Nodes", y="Execution Time (s)", hue="Processors", marker="o", palette="viridis")
    plt.xscale("log") # Use log scale for nodes if range is large
    plt.title("Parallel Dijkstra: Execution Time vs. Number of Nodes")
    plt.xlabel("Number of Nodes (V)")
    plt.ylabel("Execution Time (seconds)")
    plt.legend(title="Processors")
    plt.tight_layout()
    execution_time_plot_path = os.path.join(output_dir, "execution_time_vs_nodes.png")
    plt.savefig(execution_time_plot_path) # Save the plot
    print(f"Saved '{execution_time_plot_path}'")
    plt.close() # Close the plot to free memory

    # Plot 2: Speedup vs. Number of Processors for different Graph Sizes
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_results, x="Processors", y="Speedup vs. P=1", hue="Nodes", marker="o", palette="magma")
    plt.title("Parallel Dijkstra: Speedup vs. Number of Processors")
    plt.xlabel("Number of Processors (P)")
    plt.ylabel("Speedup (relative to P=1)")
    plt.xticks(processor_counts) # Ensure x-axis ticks are at processor counts
    plt.legend(title="Number of Nodes")
    plt.tight_layout()
    speedup_plot_path = os.path.join(output_dir, "speedup_vs_processors.png")
    plt.savefig(speedup_plot_path) # Save the plot
    print(f"Saved '{speedup_plot_path}'")
    plt.close() # Close the plot to free memory

    print("\n--- Benchmarking Complete ---")
    print(f"Check the '{output_dir}' folder for the CSV file and plot images.")
