import xml.etree.ElementTree as ET
import networkx as nx
import matplotlib.pyplot as plt

# Parse the XML
tree = ET.parse('a280.xml')
root = tree.getroot()

# Initialize the graph
G = nx.DiGraph()  # Directed graph (edges have direction)


vertex_id = 0


for vertex in root.findall('.//graph//vertex'):
    for edge in vertex.findall('edge'):
        edge_cost = edge.attrib.get('cost', 'Brak kosztu')  # Default if no cost attribute
        print(f"Edge cost before processing: {edge_cost}")
        
        edge_cost = float(edge_cost)

        to_vertex = edge.text.strip()
        
 
        G.add_edge(vertex_id, int(to_vertex), weight=edge_cost)
    

    vertex_id += 1

# Drawing the graph
pos = nx.spring_layout(G)  # Layout for visualization
plt.figure(figsize=(10, 10))

# Draw the nodes, edges, and labels
nx.draw(G, pos, with_labels=True, node_size=500, node_color='skyblue', font_size=10, font_weight='bold', edge_color='gray')

# Draw edge labels (the weights/costs)
edge_labels = nx.get_edge_attributes(G, 'weight')

# Filter out edges with zero cost for drawing the edge labels
edge_labels = {k: v for k, v in edge_labels.items() if v != 0}

nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

# Show the plot
plt.title("Wizualizacja grafu")
plt.show()
