import xml.etree.ElementTree as ET
import networkx as nx
import numpy as np
import cv2

# Parse the XML
tree = ET.parse('l.xml')
root = tree.getroot()

# Initialize the graph
G = nx.DiGraph()  # Directed graph (edges have direction)

vertex_id = 0

# Populate the graph with edges and vertices from the XML
for vertex in root.findall('.//graph//vertex'):
    for edge in vertex.findall('edge'):
        edge_cost = edge.attrib.get('cost', 'Brak kosztu')  # Default if no cost attribute
        #print(f"Edge cost before processing: {edge_cost}")
        
        edge_cost = float(edge_cost)

        to_vertex = edge.text.strip()
        
        # Add edges to the graph
        G.add_edge(vertex_id, int(to_vertex), weight=edge_cost)
    
    vertex_id += 1

# Create a blank image (black background)
width, height = 1800, 1000
image = np.zeros((height, width, 3), dtype=np.uint8)

# Define positions of nodes (you can use a layout like spring_layout)
pos = nx.spring_layout(G, seed=42)
node_positions = {k: (int(v[0] * width / 2 + width / 2), int(v[1] * height / 2 + height / 2)) for k, v in pos.items()}

# Draw the edges (lines between nodes)
for edge in G.edges():
    start_node = edge[0]
    end_node = edge[1]
    
    # Get positions for start and end node
    start_pos = node_positions[start_node]
    end_pos = node_positions[end_node]
    
    # Draw the line (edge)
    cv2.line(image, start_pos, end_pos, (192, 192, 192), 2)  # Light gray color for edges

# Draw the nodes (vertices)
for node, (x, y) in node_positions.items():
    cv2.circle(image, (x, y), 20, (135, 206, 235), -1)  # Sky blue color for nodes
    cv2.putText(image, str(node), (x - 10, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

# Draw the edge labels (costs)
for edge in G.edges(data=True):  # Corrected: edge contains both start, end, and data
    start_pos = node_positions[edge[0]]
    end_pos = node_positions[edge[1]]
    data = edge[2]  # Extract the data (weight) for the edge
    
    # Midpoint for placing the label
    label_pos = ((start_pos[0] + end_pos[0]) // 2, (start_pos[1] + end_pos[1]) // 2)
    
    # Display the edge label (cost)
    edge_label = f"{int(data['weight'])}"
    cv2.putText(image, edge_label, label_pos, cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 67, 33), 1)

# Display the graph
cv2.imshow('Graph Visualization', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
