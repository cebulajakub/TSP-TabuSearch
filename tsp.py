import random
import numpy as np
import networkx as nx
import cv2
import xml.etree.ElementTree as ET

# Funkcja do obliczania kosztu trasy w TSP
def calculate_cost(G, tour):
    total_cost = 0
    for i in range(len(tour) - 1):
        total_cost += G[tour[i]][tour[i + 1]]['weight']
    total_cost += G[tour[-1]][tour[0]]['weight']  # Dodać koszt powrotu na początek
    return total_cost


# Funkcja do generowania sąsiedztwa (zamiana dwóch wierzchołków w trasie)
def generate_neighborhood(tour):
    neighborhood = []
    for i in range(len(tour)):
        for j in range(i + 1, len(tour)):
            # Zamiana dwóch wierzchołków w trasie
            new_tour = tour[:]
            new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
            neighborhood.append(new_tour)
    return neighborhood


# Tabu Search z graficzną aktualizacją
def tabu_search(G, max_iterations=1000, tabu_tenure=10):
    # Losowa inicjalizacja rozwiązania (losowa permutacja wierzchołków)
    nodes = list(G.nodes)
    if not nodes:
        raise ValueError("The graph is empty, no nodes to form a solution.")

    current_solution = random.sample(nodes, len(nodes))
    best_solution = current_solution
    best_cost = calculate_cost(G, best_solution)

    # Tabu List i liczba iteracji tabu
    tabu_list = {}
    iteration = 0

    print("Tabu Search started...\n")
    print(f"Initial solution: {current_solution}, Initial cost: {best_cost}\n")

    # Tworzenie pustego obrazu i okna
    width, height = 1600, 800
    image = np.zeros((height, width, 3), dtype=np.uint8)

    while iteration < max_iterations:
        neighborhood = generate_neighborhood(current_solution)
        best_move = None
        best_move_cost = float('inf')

        for move in neighborhood:
            # Sprawdzenie, czy ruch jest tabu
            move_tuple = tuple(move)
            if move_tuple not in tabu_list or tabu_list[move_tuple] < iteration:
                move_cost = calculate_cost(G, move)
                if move_cost < best_move_cost:
                    best_move_cost = move_cost
                    best_move = move

        # Jeżeli najlepszy ruch nie został znaleziony, należy uniknąć przypisania None
        if best_move is not None:
            # Aktualizacja rozwiązania
            current_solution = best_move
            current_cost = best_move_cost

            # Aktualizacja najlepszego rozwiązania
            if current_cost < best_cost:
                best_solution = current_solution
                best_cost = current_cost

            # Dodanie ruchu do Tabu List
            tabu_list[tuple(current_solution)] = iteration + tabu_tenure

            # Wyświetlanie stanu w trakcie procesu graficznie
            print(f"Iteration {iteration + 1}:")
            print(f"Best Move: {best_move}, Cost: {best_move_cost}")
            print(f"Current Solution: {current_solution}, Current Cost: {current_cost}")
            print(f"Best Solution: {best_solution}, Best Cost: {best_cost}\n")

            # Rysowanie grafu z aktualnym rozwiązaniem
            draw(G, current_solution, image)

        else:
            print(f"Iteration {iteration + 1}: No move found. Current solution remains the same.\n")

        iteration += 1

    print("Tabu Search completed.")
    return best_solution, best_cost


# Funkcja ładująca graf z pliku XML
def load(path):

    tree = ET.parse(path)
    root = tree.getroot()

    G = nx.DiGraph()
    vertex_id = 0

    # Populate the graph with edges and vertices from the XML
    for vertex in root.findall('.//graph//vertex'):
        for edge in vertex.findall('edge'):
            edge_cost = edge.attrib.get('cost', 'Brak kosztu')  # Default if no cost attribute
            edge_cost = float(edge_cost)
            to_vertex = edge.text.strip()

            G.add_edge(vertex_id, int(to_vertex), weight=edge_cost)

        vertex_id += 1

    print("Graph loaded with nodes:", list(G.nodes))  # Debugging: Print nodes after loading the graph
    print("Graph loaded with edges:", list(G.edges))  # Debugging: Print edges after loading the graph

    return G


# Rysowanie grafu z aktualnym rozwiązaniem
def draw(Graph, current_solution, image):
    pos = nx.spring_layout(Graph, seed=42)
    width, height = 1600, 800
    node_positions = {k: (int(v[0] * width / 2 + width / 2), int(v[1] * height / 2 + height / 2)) for k, v in
                      pos.items()}

    # Rysowanie krawędzi
    image.fill(0)  # Czyszczenie obrazu na każdą iterację
    for edge in Graph.edges():
        start_node = edge[0]
        end_node = edge[1]
        start_pos = node_positions[start_node]
        end_pos = node_positions[end_node]
        cv2.line(image, start_pos, end_pos, (192, 192, 192), 2)

    # Rysowanie wierzchołków
    for node, (x, y) in node_positions.items():
        cv2.circle(image, (x, y), 20, (135, 206, 235), -1)
        cv2.putText(image, str(node), (x - 10, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    # Rysowanie trasy
    for i in range(len(current_solution) - 1):
       start_pos = node_positions[current_solution[i]]
       end_pos = node_positions[current_solution[i + 1]]
       cv2.line(image, start_pos, end_pos, (255, 0, 0), 4)

    # Dodać linię powrotną do pierwszego wierzchołka
    start_pos = node_positions[current_solution[-1]]
    end_pos = node_positions[current_solution[0]]
    cv2.line(image, start_pos, end_pos, (255, 0, 0), 4)

    # Wyświetlanie grafu w oknie
    cv2.imshow('Graph Visualization', image)
    cv2.waitKey(1)  # Zamiast 200 ms, teraz tylko 1 ms, aby odświeżyć okno
    # Bez cv2.destroyAllWindows() - żeby nie zamykać okna po każdej iteracji


if __name__ == "__main__":
    # Load graph
    path = "E:/TSP_TABU/pythonProject1/TSP-TabuSearch/l.xml"
    G = load(path)

    # Run Tabu Search for TSP
    try:
        best_solution, best_cost = tabu_search(G, max_iterations=40000, tabu_tenure=10)
        print("Best Solution (Tour):", best_solution)
        print("Best Cost:", best_cost)
    except ValueError as e:
        print(e)

    # Gdy zakończono, czekaj na zamknięcie okna przez użytkownika
    cv2.waitKey(0)
    cv2.destroyAllWindows()
