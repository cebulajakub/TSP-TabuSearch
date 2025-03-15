import random
import numpy as np
import networkx as nx
import cv2
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
 
 
 następnie wykonac zamiane policzyc koszt z uwzglednieniem tabu listy
 
 Wyznaczamy zbiory liczb któr mozna przestrawiac i dla kazdej pary wyznaczamy permutacje
 
 Algorytm unika oscylacji wokół optimum lokalnego dzięki przechowywaniu informacji o sprawdzonych już rozwiązaniach w postaci listy tabu (TL).
 
 
'''
# def generate_tabu_matrix(G):


def generate_2opt_neighborhood(tour):
    neighborhood = []
    n = len(tour)
    for i in range(n - 1):
        for j in range(i + 2, n):  # Upewniamy się, że nie zamieniamy sąsiednich wierzchołków
            new_tour = tour[:i] + tour[i:j][::-1] + tour[j:]
            neighborhood.append(new_tour)
    return neighborhood


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


def diversify_solution(tour, diversification_factor=0.15):
    """Dywersyfikacja: losowo zmienia część trasy."""
    n = len(tour)
    num_swaps = int(diversification_factor * n)
    print("Number of vertex swap :",num_swaps)
    for _ in range(num_swaps):
        i, j = random.sample(range(n), 2)
        tour[i], tour[j] = tour[j], tour[i]
    return tour

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

def tabu_search(G, max_iterations=1000, tabu_tenure=10, diversification_factor=0.15, max_no_improve=100):
    nodes = list(G.nodes)
    if not nodes:
        raise ValueError("The graph is empty, no nodes to form a solution.")
    
    current_solution = greedy_solution(G)
    best_solution = current_solution
    best_cost = calculate_cost(G, best_solution)
    global_best_cost = best_cost
    tabu_list = []  # Lista krawędzi zamiast pełnych tras
    iteration = 0
    no_improve_count = 0
    
    
    print("Tabu Search started...\n")
    print(f"Initial solution: {current_solution}, Initial cost: {best_cost}\n")
    
    while iteration < max_iterations and no_improve_count < max_no_improve:
        # Tworzymy sąsiedztwo w zależności od liczby iteracji bez poprawy
        if iteration % 10 == 0 or iteration % 11 == 0:
            neighborhood = generate_3opt_neighborhood(current_solution)
        else:
            neighborhood = generate_2opt_neighborhood(current_solution)
        
        best_move = None
        best_move_cost = float('inf')
        best_move_edges = None
        
        # Szukamy najlepszego ruchu
        for move in neighborhood:
            move_edges = set(zip(move, move[1:] + [move[0]]))  # Zapisujemy tylko krawędzie
            move_cost = calculate_cost(G, move)
            
            if move_edges not in tabu_list or move_cost < best_cost:  # Aspiracja
                if move_cost < best_move_cost:
                    best_move_cost = move_cost
                    best_move = move
                    best_move_edges = move_edges
        
        if best_move is not None:
            current_solution = best_move
            current_cost = best_move_cost
            
            if current_cost < best_cost:
                best_solution = current_solution
                best_cost = current_cost
                no_improve_count = 0  # Reset licznika poprawy
            else:
                no_improve_count += 1

            tabu_list.append(best_move_edges)
            if len(tabu_list) > tabu_tenure:
                tabu_list.pop(0)  # Ograniczenie długości listy tabu
                
            if no_improve_count > 25:  
                print(f"Iteration {iteration + 1}: No improvement for 5 steps, applying diversification.")
                current_solution = diversify_solution(current_solution, diversification_factor)
                current_cost = calculate_cost(G, current_solution)
                no_improve_count = 0
                best_cost=current_cost
                
            if current_cost < global_best_cost:
                global_best_cost = current_cost
                
            print(f"Iteration {iteration + 1}: Best Cost = {global_best_cost}")
            print(f"current_cost: {current_cost}")

        iteration += 1

    print("Tabu Search completed.")
    return best_solution, best_cost

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
    # Load graph
    path = "D:\AlgorytmyOptumalizacji\\berlin52.xml"
    G = load(path)
    try:
        best_solution, best_cost = tabu_search(G, max_iterations=40000, tabu_tenure=10)
        print("Best Solution (Tour):", best_solution)
        print("Best Cost:", best_cost)
    except ValueError as e:
        print(e)