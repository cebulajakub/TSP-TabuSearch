import random
import time
import numpy as np
import networkx as nx
import cv2
import keyboard
import xml.etree.ElementTree as ET


def calculate_cost(G, tour):

    total_cost = 0
    #print(f'Number of nodes { len(tour)} ')
    for i in range(0,len(tour)-1):
        total_cost += G[tour[i]][tour[i+1]]['weight']

    # last to first
    total_cost += G[tour[-1]][tour[0]]['weight']


    #print(f'Total cost is {total_cost}')

    return total_cost
'''
musimy miec permutacje:
normlanie bedzie ich  n!
ale nie musimy przegladac wszystkich 
tylko (n-1)!/2 zeby nie powtarzac
wybieramy randomowa permutacje  i szukamy najblizszych sasiadów (roziwązań) tzw. downhill
 najprosztsza metod 2-opt (zamiana 2 krawedzi liczymyu zamiane z całego zbioru a nie tylko z 2 losowych wierzchołkó) :
 
 Aby zaminiec trzeba zrobic liste najlepszych ruchów czyli policzyc wszystkie mozliwe trasy
 my mamy graf pelny wiec duzo liczenia to zajmie(chyba) i uwzgledniac tabu liste ( nwm jak ja narazie reprezentowac)
 
 
 3-opt złożoność n(n−1)(n−2)/6 ----- O(n^3)
 
 następnie wykonac zamiane policzyc koszt z uwzglednieniem tabu listy
 
 Wyznaczamy zbiory liczb któr mozna przestrawiac i dla kazdej pary wyznaczamy permutacje
 
 Algorytm unika oscylacji wokół optimum lokalnego dzięki przechowywaniu informacji o sprawdzonych już rozwiązaniach w postaci listy tabu (TL).
 
 
'''
# def generate_tabu_matrix(G):


def generate_3opt_neighborhood(tour):
    neighborhood = []
    n = len(tour)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                new_tour_1 = tour[:i] + tour[i:j][::-1] + tour[j:k][::-1] + tour[k:]
                new_tour_2 = tour[:i] + tour[j:k] + tour[i:j] + tour[k:]
                new_tour_3 = tour[:i] + tour[j:k][::-1] + tour[i:j] + tour[k:]
                neighborhood.extend([new_tour_1, new_tour_2, new_tour_3])
    return neighborhood

def generate_2opt_neighborhood(tour):
    neighborhood = []
    n = len(tour)
    for i in range(n - 1):
        for j in range(i + 2, n):  # Upewniamy się, że nie zamieniamy sąsiednich wierzchołków
            new_tour = tour[:i] + tour[i:j][::-1] + tour[j:]
            neighborhood.append(new_tour)
    return neighborhood

def random_edge_swap(tour):
    """Randomly swap two edges in the tour"""
    n = len(tour)
    i, j = random.sample(range(n), 2)  # Randomly select two different indices
    # Swap positions of the nodes in the tour
    tour[i], tour[j] = tour[j], tour[i]
    return tour

def diversify_solution(tour, diversification_factor):
    """Dywersyfikacja: losowo zmienia część trasy."""
    n = len(tour)
    num_swaps = int(diversification_factor * n)
    print("Number of vertex swap :",num_swaps)
    for _ in range(num_swaps):
        i, j = random.sample(range(n), 2)
        tour[i], tour[j] = tour[j], tour[i]
    return tour

def dynamic_tabu_tenure(iteration, max_iterations, base_tenure=10):
    """Dynamiczne dostosowanie długości listy tabu."""
    return base_tenure + int((iteration / max_iterations) * base_tenure * base_tenure)

def dynamic_max_no_improve(iteration, max_iterations, base_no_improve=15):
    """Dynamiczne dostosowanie limitu iteracji bez poprawy."""
    print("base :",base_no_improve,"iteration :",iteration,"max_iteration", max_iterations)
    dynamic_no_imp = base_no_improve + int((iteration / max_iterations) * base_no_improve * base_no_improve)
    print("dynamic :",dynamic_no_imp, "+", (iteration / max_iterations) * base_no_improve * base_no_improve)
    return dynamic_no_imp

def greedy_solution(G):
    start_node = random.choice(list(G.nodes))
    unvisited = set(G.nodes)
    unvisited.remove(start_node)
    tour = [start_node]
    
    while unvisited:
        last_node = tour[-1]
        next_node = min(unvisited, key=lambda node: G[last_node][node]['weight'])
        tour.append(next_node)
        unvisited.remove(next_node)
    
    return tour
'''def get_nearest_neighbor(last_node, unvisited, G):
   min_weight = float('inf')
    nearest_node = None
    
    for node in unvisited:
        weight = G[last_node][node]['weight']
        if weight < min_weight:
            min_weight = weight
            nearest_node = node
    
    return nearest_node

def greedy_solution(G):
    start_node = random.choice(list(G.nodes))
    unvisited = set(G.nodes)
    unvisited.remove(start_node)
    tour = [start_node]

    while unvisited:
        last_node = tour[-1]
        next_node = get_nearest_neighbor(last_node, unvisited, G)
        tour.append(next_node)
        unvisited.remove(next_node)

    return tour

'''

def tabu_search(G, max_iterations=1000, base_tabu_tenure=10, diversification_factor=0.15):
    """Algorytm Tabu Search z adaptacyjną dywersyfikacją i dynamicznymi parametrami."""
    nodes = list(G.nodes)
    print("nodes", nodes)
    if not nodes:
        raise ValueError("The graph is empty, no nodes to form a solution.")

    # Inicjalizacja
    current_solution = greedy_solution(G)
    best_solution = current_solution
    best_cost = calculate_cost(G, best_solution)
    global_best_cost=best_cost
    global_best_solution=best_solution
    tabu_list = []
    iteration = 0
    no_improve_count = 0
    max_no_improve = 25
    #last_diversification_iteration = -float('inf')  # Ostatnia iteracja z dywersyfikacją

    print("Tabu Search started...\n")
    print(f"Initial solution: {current_solution}, Initial cost: {best_cost}\n")

    while iteration < max_iterations:
        # Dynamiczne dostosowanie parametrow
        tabu_tenure = max(15, min(dynamic_tabu_tenure(iteration, max_iterations), 50))
        
        #max_no_improve = dynamic_max_no_improve(iteration, max_iterations)

        # Generowanie sasiedztwa
        if iteration % 10 == 0 or iteration % 11 == 0:
            neighborhood = generate_3opt_neighborhood(current_solution)
        else:
            neighborhood = generate_2opt_neighborhood(current_solution)
        best_move = None
        best_move_cost = float('inf')
        best_move_edges = None

        # Przeszukiwanie sasiedztwa
        for move in neighborhood:
            move_edges = set(zip(move, move[1:] + [move[0]]))  # Zapisujemy krawedzie
            move_cost = calculate_cost(G, move)

            # aspiracja
            if move_edges not in tabu_list or move_cost < best_cost:
                if move_cost < best_move_cost:
                    best_move_cost = move_cost
                    best_move = move
                    best_move_edges = move_edges
            

        # Jeśli znaleziono ruch
        if best_move is not None:
            current_solution = best_move
            current_cost = best_move_cost

            # Aktualizacja najlepszego rozwiązania  gdy lepsze
            if current_cost < best_cost:
                best_solution = current_solution
                best_cost = current_cost
                no_improve_count = 0  # Reset licznika
            else:
                no_improve_count += 1

            #print(f"tabulistbefore = {tabu_list}")
            #if tabu_tenure >= 50:
                #tabu_tenure = max(15, tabu_tenure - 10)
                #print(f"tabulistafter = {tabu_list}")
            # Dodanie ruchu do listy tabu
            tabu_list.append(best_move_edges)
            if len(tabu_list) > tabu_tenure:
                tabu_list.pop(0)  # Ograniczenie dlugości listy tabu
            
            #print(f"tabulistafter22 = {tabu_list}")

            print(f"Iteration {iteration + 1}: Best Cost = {global_best_cost}, Current = {current_cost}, Tabu Tenure = {tabu_tenure}, Max No Improve = {max_no_improve}")

            if no_improve_count > max_no_improve:  
                    print(f"Iteration {iteration + 1}: No improvement for 5 steps, applying diversification.")
                    current_solution = diversify_solution(current_solution, diversification_factor)
                    current_cost = calculate_cost(G, current_solution)
                    no_improve_count = 0
                    best_cost=current_cost

            if current_cost < global_best_cost:
                global_best_cost = current_cost
                global_best_solution = current_solution
                
        iteration += 1
        if keyboard.is_pressed('q'):
            break
    print("Tabu Search completed.")
    return global_best_solution, global_best_cost

# Rysowanie grafu z aktualnym rozwiązaniem
def draw(Graph, current_solution, image):
    # Używamy spring_layout do rozplanowania wierzchołków
    pos = nx.spring_layout(Graph, seed=42, scale=0.8)
    width, height = 1600, 800  # Ustalamy wymiary obrazu
    node_positions = {k: (int(v[0] * width / 2 + width / 2), int(v[1] * height / 2 + height / 2)) for k, v in pos.items()}

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
        cv2.circle(image, (x, y), 20, (135, 206, 235), -1)  # Kolor wierzchołków
        cv2.putText(image, str(node), (x - 10, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    # Rysowanie trasy
    for i in range(len(current_solution) - 1):
        start_pos = node_positions[current_solution[i]]
        end_pos = node_positions[current_solution[i + 1]]
        cv2.line(image, start_pos, end_pos, (255, 0, 0), 4)  # Kolor trasy (czerwony)

    # Dodać linię powrotną do pierwszego wierzchołka
    start_pos = node_positions[current_solution[-1]]
    end_pos = node_positions[current_solution[0]]
    cv2.line(image, start_pos, end_pos, (255, 0, 0), 4)

    # Wyświetlanie grafu w oknie
    cv2.imshow('Graph Visualization', image)
    cv2.waitKey(0)  # Zamiast 200 ms, teraz tylko 1 ms, aby odświeżyć okno
    # Nie używamy cv2.destroyAllWindows() - żeby nie zamykać okna po każdej iteracji
    # Bez cv2.destroyAllWindows() - żeby nie zamykać okna po każdej iteracji



def load(path):
    tree = ET.parse(path)
    root = tree.getroot()
    G = nx.DiGraph()
    vertex_id = 0

    for vertex in root.findall('.//graph//vertex'):
        for edge in vertex.findall('edge'):
            edge_cost = float(edge.attrib.get('cost', 0))
            to_vertex = edge.text.strip()
            G.add_edge(vertex_id, int(to_vertex), weight=edge_cost)
        vertex_id += 1

    return G



if __name__ == "__main__":
    path = "D:\AlgorytmyOptumalizacji\\swiss42.xml"
    G = load(path)
    image = np.zeros((800, 1600, 3), dtype=np.uint8)
    try:
        start_time = time.time()
        best_solution, best_cost = tabu_search(G, max_iterations=1000, diversification_factor=0.1)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Time taken for Tabu Search: {execution_time:.2f} seconds")
        print("Best Solution (Tour):", best_solution)
        print("Best Cost:", best_cost)
        draw(G, best_solution, image)
    except ValueError as e:
        print(e)