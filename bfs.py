from collections import deque
def bfs(graph, start, destination):
    queue = deque([(start, [start])])
    visited = set()
    while queue:
        current, path = queue.popleft()
        if current == destination:
            print("Path found:", ' -> '.join(path))
            return
        visited.add(current)
        for neighbor in graph[current]:
            if neighbor not in visited:
                queue.append((neighbor, path + [neighbor]))
                visited.add(neighbor)
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
bfs(india_city_graph, start_city, destination_city)
