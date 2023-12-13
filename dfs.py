# DFS

def dfs(graph, current, destination, visited=None, path=None):
    if visited is None:
        visited = set()
    if path is None:
        path = []
    path = path + [current]
    visited.add(current)
    if current == destination:
        print("Path found:", ' -> '.join(path))
        return
    for neighbor in graph[current]:
        if neighbor not in visited:
            dfs(graph, neighbor, destination, visited, path)
india_city_graph = {
    'Delhi': ['Mumbai', 'Jaipur', 'Lucknow'],
    'Mumbai': ['Delhi', 'Chennai', 'Bangalore'],
    'Chennai': ['Mumbai', 'Bangalore'],
    'Bangalore': ['Mumbai', 'Chennai', 'Hyderabad'],
    'Hyderabad': ['Bangalore', 'Chennai'],
    'Jaipur': ['Delhi', 'Lucknow'],
    'Lucknow': ['Delhi', 'Jaipur']
}
start_city = input("Enter the start city in India: ")
destination_city = input("Enter the destination city in India: ")
print(f"Searching for a path from {start_city} to {destination_city}")
dfs(india_city_graph, start_city, destination_city)
