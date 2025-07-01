import pytest
from dijkstra_solver import Graph, PriorityQueue, parallel_dijkstra # Assuming your code is in 'your_module.py'

def test_graph_add_edge():
    g = Graph()
    g.add_edge(0, 1, 5)
    assert g.get_neighbors(0) == {1: 5}
    assert 1 in g.get_nodes()
    with pytest.raises(ValueError):
        g.add_edge(0, 2, -1) # Test negative weight edge case

def test_priority_queue_basic():
    pq = PriorityQueue()
    pq.add_task(1, 10)
    pq.add_task(2, 5)
    pq.add_task(3, 15)
    assert pq.pop_task() == 2 # Smallest priority first
    assert pq.pop_task() == 1
    assert not pq.is_empty()
    pq.pop_task()
    assert pq.is_empty()
    with pytest.raises(KeyError):
        pq.pop_task() # Test popping from empty queue

def test_dijkstra_simple_graph():
    g = Graph()
    g.add_edge(0, 1, 1)
    g.add_edge(1, 2, 1)
    g.add_edge(0, 2, 3)
    distances = parallel_dijkstra(g, 0, num_processors=1) # Use single processor for deterministic test
    assert distances[0] == 0
    assert distances[1] == 1
    assert distances[2] == 2 # Path 0 -> 1 -> 2 is shorter than 0 -> 2

def test_dijkstra_disconnected_graph():
    g = Graph()
    g.add_edge(0, 1, 1)
    g.add_edge(2, 3, 1) # Disconnected component
    distances = parallel_dijkstra(g, 0, num_processors=1)
    assert distances[0] == 0
    assert distances[1] == 1
    assert distances[2] == float('inf') # Unreachable
    assert distances[3] == float('inf') # Unreachable

def test_dijkstra_empty_graph():
    empty_graph = Graph()
    with pytest.raises(ValueError, match="Start node .* not found in graph."):
        parallel_dijkstra(empty_graph, 0) # Start node not in empty graph

