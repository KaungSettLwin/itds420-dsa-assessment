import heapq
import collections
from concurrent.futures import ProcessPoolExecutor # Changed from ThreadPoolExecutor
import time
import random
import math # Added this import
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd # Added this import for DataFrame

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
        self.adj_list = collections.defaultdict(dict)
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
        self._entry_finder = {}  # Maps item to its entry in the queue (priority, item, True/False for removed)
        self._counter = 0     # Unique sequence count for tie-breaking

    def add_task(self, item: int, priority: float):
        """
        Add a new item or update the priority of an existing item.
        If the item exists, its priority is updated. If the new priority
        is higher than the current one, the item's position in the queue
        is effectively updated (a new entry is added, old one marked removed).
        """
        if item in self._entry_finder:
            # Mark old entry as removed
            self._entry_finder[item][2] = False
        
        count = next(self._id_generator())
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
            if active:
                del self._entry_finder[item]
                return item
        raise KeyError('pop from an empty priority queue')

    def is_empty(self) -> bool:
        """Checks if the priority queue is empty."""
        return not self._entry_finder

    def _id_generator(self):
        """Generates unique IDs for tie-breaking in the priority queue."""
        while True:
            yield self._counter
            self._counter += 1

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

def _relax_edge(graph: Graph, distances: dict, pq: PriorityQueue, u: int, v: int, weight: float):
    """
    Helper function to relax an edge (u, v) and update distances.
    This function could be executed in parallel.
    
    Args:
        graph (Graph): The graph object.
        distances (dict): Shared dictionary of current shortest distances.
        pq (PriorityQueue): Shared priority queue.
        u (int): Source node of the edge.
        v (int): Destination node of the edge.
        weight (float): Weight of the edge (u, v).
    """
    # This block might require a lock for `distances` and `pq` in a truly multi-threaded
    # environment to prevent race conditions. Python's GIL often simplifies this for
    # simple cases, but explicit locking is best practice.
    new_distance = distances[u] + weight
    if new_distance < distances[v]:
        distances[v] = new_distance
        pq.add_task(v, new_distance)


# --- Usage Example ---
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

    # --- Real Timing Plot/Benchmark Generation Note ---
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

    for config in graph_configs:
        num_nodes = config["nodes"]
        num_edges = num_nodes * config["edges_factor"]
        
        print(f"\nGenerating graph with V={num_nodes}, E={num_edges}...")
        graph = generate_random_graph(num_nodes, num_edges)
        
        # Select a random start node for consistency
        start_node = random.randint(0, num_nodes - 1) if num_nodes > 0 else 0

        # Run for each processor count
        for P in processor_counts:
            print(f"  Running with P={P}...")
            try:
                start_time = time.perf_counter()
                # Call the parallel Dijkstra function
                _ = parallel_dijkstra(graph, start_node, num_processors=P)
                end_time = time.perf_counter()
                execution_time = end_time - start_time
                
                # Store results
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

    print("\n--- Benchmark Results Summary ---")
    # Calculate speedup for P > 1 relative to P=1 for the same graph size
    for result in benchmark_results:
        if result["Processors"] == 1:
            # Find the baseline time for 1 processor for this graph size
            baseline_time = next((r["Execution Time (s)"] for r in benchmark_results 
                                  if r["Nodes"] == result["Nodes"] and r["Processors"] == 1), float('nan'))
            result["Speedup vs. P=1"] = 1.0
        else:
            baseline_time = next((r["Execution Time (s)"] for r in benchmark_results 
                                  if r["Nodes"] == result["Nodes"] and r["Processors"] == 1), float('nan'))
            # Fix: Use math.isnan() to check for NaN values
            if baseline_time != 0 and not math.isnan(baseline_time):
                result["Speedup vs. P=1"] = baseline_time / result["Execution Time (s)"]
            else:
                result["Speedup vs. P=1"] = float('nan')

    # Print results in a table format (similar to the one in your assessment document)
    print(f"{'Nodes':<10} {'Edges':<10} {'P':<5} {'Time (s)':<15} {'Speedup':<10}")
    print("-" * 55)
    for res in benchmark_results:
        time_str = f"{res['Execution Time (s)']:.4f}" if not math.isnan(res['Execution Time (s)']) else "N/A"
        speedup_str = f"{res['Speedup vs. P=1']:.2f}x" if not math.isnan(res['Speedup vs. P=1']) else "N/A"
        print(f"{res['Nodes']:<10} {res['Edges']:<10} {res['Processors']:<5} {time_str:<15} {speedup_str:<10}")

    # Convert results to a pandas DataFrame for easier plotting (optional, but good practice)
    try:
        # pandas imported at the top now
        df_results = pd.DataFrame(benchmark_results)
        print("\nDataFrame created for plotting:")
        print(df_results.head())
    except ImportError:
        print("\nPandas not installed. Skipping DataFrame creation. Install with 'pip install pandas' for easier data handling.")
        df_results = None

    # --- Plotting the Results ---
    if df_results is not None:
        print("\nGenerating plots...")
        sns.set_style("whitegrid") # Set plot style

        # Plot 1: Execution Time vs. Number of Nodes for different Processors
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df_results, x="Nodes", y="Execution Time (s)", hue="Processors", marker="o")
        plt.xscale("log") # Use log scale for nodes if range is large
        plt.title("Parallel Dijkstra: Execution Time vs. Number of Nodes")
        plt.xlabel("Number of Nodes (V)")
        plt.ylabel("Execution Time (seconds)")
        plt.legend(title="Processors")
        plt.tight_layout()
        plt.savefig("execution_time_vs_nodes.png") # Save the plot
        print("Saved 'execution_time_vs_nodes.png'")
        # plt.show() # Uncomment to display plot immediately

        # Plot 2: Speedup vs. Number of Processors for different Graph Sizes
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df_results, x="Processors", y="Speedup vs. P=1", hue="Nodes", marker="o")
        plt.title("Parallel Dijkstra: Speedup vs. Number of Processors")
        plt.xlabel("Number of Processors (P)")
        plt.ylabel("Speedup (relative to P=1)")
        plt.xticks(processor_counts) # Ensure x-axis ticks are at processor counts
        plt.legend(title="Number of Nodes")
        plt.tight_layout()
        plt.savefig("speedup_vs_processors.png") # Save the plot
        print("Saved 'speedup_vs_processors.png'")
        # plt.show() # Uncomment to display plot immediately

    print("\n--- Benchmarking Complete ---")