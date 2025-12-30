import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import json

# =========================
# CONFIG
# =========================
CSV_DELIMITER = ';'
DECIMAL = ','

# File names (as uploaded)
NODE_FILE = 'BSM307_317_Guz2025_TermProject_NodeData.csv'
EDGE_FILE = 'BSM307_317_Guz2025_TermProject_EdgeData.csv'
DEMAND_FILE = 'BSM307_317_Guz2025_TermProject_DemandData.csv'

# =========================
# 1) Load Data from CSV
# =========================
def load_graph_from_csv(node_file, edge_file):
    # Read CSVs
    # Note: decimal=',' is used because the files use European format (e.g., 0,962)
    node_df = pd.read_csv(node_file, sep=CSV_DELIMITER, decimal=DECIMAL)
    edge_df = pd.read_csv(edge_file, sep=CSV_DELIMITER, decimal=DECIMAL)
    
    G = nx.Graph()

    # Add Nodes with attributes
    for _, row in node_df.iterrows():
        node_id = int(row['node_id'])
        G.add_node(node_id, 
                   s_ms=row['s_ms'], 
                   r_node=row['r_node'])

    # Add Edges with attributes
    for _, row in edge_df.iterrows():
        u, v = int(row['src']), int(row['dst'])
        G.add_edge(u, v, 
                   capacity_mbps=row['capacity_mbps'], 
                   delay_ms=row['delay_ms'], 
                   r_link=row['r_link'])
    
    return G

# =========================
# 2) SAVE JSON
# =========================
def save_graph_json(G, filename="network_final_project.json"):
    data = {"nodes": [], "edges": []}

    # Nodes
    for n in sorted(G.nodes()):
        data["nodes"].append({
            "node_id": int(n),
            "processing_delay_ms": G.nodes[n].get('s_ms', 0),
            "node_reliability": G.nodes[n].get('r_node', 0)
        })

    # Edges
    for u, v in sorted(G.edges()):
        e = G.edges[u, v]
        data["edges"].append({
            "source": int(u),
            "destination": int(v),
            "bandwidth_mbps": e.get('capacity_mbps', 0),
            "delay_ms": e.get('delay_ms', 0),
            "link_reliability": e.get('r_link', 0)
        })

    with open(filename, "w") as f:
        json.dump(data, f, indent=2)

    print(f"JSON saved with full attribute names → {filename}")

# =========================
# 3) VISUALIZE NETWORK
# =========================
def visualize_network(G):
    plt.figure(figsize=(12, 12))
    # Using a faster layout for large graphs (250 nodes)
    pos = nx.spring_layout(G, k=0.15, iterations=20)

    nx.draw_networkx_nodes(G, pos, node_size=30, node_color="#4DBFFF")
    nx.draw_networkx_edges(G, pos, width=0.2, alpha=0.3, edge_color="gray")

    plt.title(f"Network Topology (Nodes={G.number_of_nodes()}, Edges={G.number_of_edges()})")
    plt.axis("off")
    plt.savefig("network_visualization.png") # Saving instead of show()
    print("Visualization saved as 'network_visualization.png'")

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    print("Loading network from CSV files...")
    
    try:
        # Load the graph
        G = load_graph_from_csv(NODE_FILE, EDGE_FILE)
        
        print(f"Graph Loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        print(f"Is Connected: {nx.is_connected(G)}")

        # Optional: Load demand data if needed for later calculations
        demand_df = pd.read_csv(DEMAND_FILE, sep=CSV_DELIMITER)
        print(f"Demand data loaded: {len(demand_df)} requests")

        print("Saving JSON...")
        save_graph_json(G)

        print("Visualizing network...")
        visualize_network(G)

        print("DONE ✓ Veriler başarıyla yüklendi ve işlendi!")
        
    except FileNotFoundError as e:
        print(f"Error: Could not find file. {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
