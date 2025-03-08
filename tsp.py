import xml.etree.ElementTree as ET
import networkx as nx
import numpy as np
import cv2



def load(path):
    # Parse the XML
    tree = ET.parse(path)
    root = tree.getroot()


    # Initialize the graph
    G = nx.DiGraph()  # Directed graph (edges have direction)

    vertex_id = 0

        # Populate the graph with edges and vertices from the XML
    for vertex in root.findall('.//graph//vertex'):
        for edge in vertex.findall('edge'):
            edge_cost = edge.attrib.get('cost', 'Brak kosztu')  # Default if no cost attribute
            edge_cost = float(edge_cost)
            to_vertex = edge.text.strip()
            
            # Add edges to the graph
            G.add_edge(vertex_id, int(to_vertex), weight=edge_cost)

        vertex_id += 1
    return G    


def draw(graph):
    # Visualization part (drawing the graph using OpenCV)
    width, height = 1800, 1000
    image = np.zeros((height, width, 3), dtype=np.uint8)

    # Define positions of nodes
    pos = nx.spring_layout(G, seed=42)
    node_positions = {k: (int(v[0] * width / 2 + width / 2), int(v[1] * height / 2 + height / 2)) for k, v in pos.items()}

    # Draw the edges
    for edge in graph.edges():
        start_node = edge[0]
        end_node = edge[1]
        start_pos = node_positions[start_node]
        end_pos = node_positions[end_node]
        cv2.line(image, start_pos, end_pos, (192, 192, 192), 2)

    # Draw the nodes
    for node, (x, y) in node_positions.items():
        cv2.circle(image, (x, y), 20, (135, 206, 235), -1)
        cv2.putText(image, str(node), (x - 10, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    # Draw the edge labels (costs)
    for edge in graph.edges(data=True):  # edge contains both start, end, and data
        start_pos = node_positions[edge[0]]
        end_pos = node_positions[edge[1]]
        data = edge[2]  # Extract the data (weight) for the edge
        
        # Midpoint for placing the label
        label_pos = ((start_pos[0] + end_pos[0]) // 2, (start_pos[1] + end_pos[1]) // 2)
        
        # Display the edge label (cost)
        edge_label = f"{int(data['weight'])}"  # Display cost as integer
        cv2.putText(image, edge_label, label_pos, cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 67, 33), 1)

    # Display the graph
    cv2.imshow('Graph Visualization', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__=='__main__':
    path = "D:\AlgorytmyOptumalizacji\l.xml"
    G = load(path)
    draw(G) 