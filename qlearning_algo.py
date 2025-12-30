import pandas as pd
import networkx as nx
import random
import math
from collections import defaultdict

# Q LEARNING TABANLI AĞ OPTİMİSATİON ALGORİTMASI

# eğitim parametreleri 
EPISODES = 500          # 20 000 den 500 e indirild (gui için
MAX_STEPS = 50          # 50 den 20 ye indirildi

# Q Learning parametreleri
ALPHA = 0.1             # Öğrenme hızı 
GAMMA = 0.9             # discount

#EPSILON-GREEDY STRATEJİSİ
# keşif  Sömürü dengesi
EPSILON_START = 1.0     # Başlangıçta %100 rastgele
EPSILON_END = 0.05      # Minimum keşif oranı
EPSILON_DECAY = 0.99    # Her episode sonrası azalma oranı


# Q Tablosu
# Q[(current_node, target_node)][next_node] = Q değeri
Q = defaultdict(lambda: defaultdict(float))

# cost fonksiyonu ağırlıkları (GUI üzerinden değiştirilebilir)
w_d = 1.0   
w_r = 1.0   
w_u = 1.0  

w_total = w_d + w_r + w_u         # Wdelay + Wreliability + Wresource = 1 

w_d = w_d/w_total
w_r = w_r/w_total
w_u = w_u/w_total


#NORMALİZASYON ÜST SINIRLARI
MAX_DELAY = 100.0
MAX_RESOURCE = 100.0
MAX_RELIABILITY = 10.0


def build_graph_from_csv(node_csv, edge_csv):

    print(f"Veriler yükleniyor: {node_csv} ve {edge_csv}...")
    G = nx.Graph()

    try:
        nodes = pd.read_csv(node_csv, sep=';', decimal=',')
        edges = pd.read_csv(edge_csv, sep=';', decimal=',')
        
        global MAX_DELAY, MAX_RESOURCE, MAX_RELIABILITY
        
        # delay için %95 persentil kullanılır
        MAX_DELAY = nodes["s_ms"].quantile(0.95) + edges["delay_ms"].quantile(0.95)
        
        # resource maliyeti için düşük kapasiteye göre üst sınır
        min_cap = edges["capacity_mbps"].quantile(0.05)
        if min_cap > 0:
            MAX_RESOURCE = 1000 / min_cap
            
        # reliability için en kötü olası senaryo
        min_r_link = edges["r_link"].quantile(0.05)
        min_r_node = nodes["r_node"].quantile(0.05)
        if min_r_link > 0 and min_r_node > 0:
            MAX_RELIABILITY = (-math.log(min_r_link) - math.log(min_r_node))

    except FileNotFoundError:
        print("KRİTİK HATA: CSV dosyaları bulunamadı.")
        return None
    except Exception as e:
        print(f"HATA: {e}")
        return None

    # node ekleme
    for _, n in nodes.iterrows():
        G.add_node(
            int(n["node_id"]),
            s_ms=float(n["s_ms"]),
            r_node=float(n["r_node"])
        )

    # edge ekleme
    for _, e in edges.iterrows():
        G.add_edge(
            int(e["src"]),
            int(e["dst"]),
            capacity_mbps=float(e["capacity_mbps"]),
            delay_ms=float(e["delay_ms"]),
            r_link=float(e["r_link"])
        )

    print(f"Ağ Yüklendi: {G.number_of_nodes()} node, {G.number_of_edges()} edge")
    return G


# DEMAND YÜKLEME
def load_demands(demand_csv):

    try:
        return pd.read_csv(demand_csv, sep=';', decimal=',')
    except Exception:
        return None


#REWARD FONKSİYONU

def compute_reward(G, u, v):

    edge = G[u][v]
    node = G.nodes[v]

    #HAM METRİKLER
    delay = edge["delay_ms"] + node["s_ms"]
    resource = 1000 / edge["capacity_mbps"] if edge["capacity_mbps"] > 0 else 9999         # 1gbps yerine 1000mbps
    
    r_link = edge["r_link"] if edge["r_link"] > 0 else 0.0001
    r_node = node["r_node"] if node["r_node"] > 0 else 0.0001
    reliability = -math.log(r_link) - math.log(r_node)             #reliabilityCost(P)=X(i,j)∈P[−log( RLinkReliabilityij)]+Xk∈P[−log(N odeReliabilityk)]

    #NORMALİZASYON
    delay_norm = delay / MAX_DELAY if MAX_DELAY > 0 else 0
    resource_norm = resource / MAX_RESOURCE if MAX_RESOURCE > 0 else 0
    reliability_norm = reliability / MAX_RELIABILITY if MAX_RELIABILITY > 0 else 0

    # hepsi minimize
    total_cost = (
        w_d * delay_norm +
        w_r * resource_norm +
        w_u * reliability_norm
    )
    
    # reward = -cost      reward = +benefit
    return -10.0 * total_cost


#HARD CONSTRAINT
def feasible_neighbors(G, current_node, demand_bw):
   
    #minimum bandwidth talebini karşılayan komşuları döndürür.
    valid_neighbors = []
    for n in G.neighbors(current_node):
        if G[current_node][n]["capacity_mbps"] >= demand_bw:
            valid_neighbors.append(n)
    return valid_neighbors


def train_q_learning(G, demands):
    
    if isinstance(demands, pd.Series):
        demands = pd.DataFrame([demands])

    #epsilon decay
    epsilon = EPSILON_START

    for episode in range(EPISODES):
        if demands.empty:
            break
        
        # rastgele bir demand seçilir
        d = demands.sample(1).iloc[0]
        current = int(d["src"])
        target = int(d["dst"])
        demand_bw = float(d["demand_mbps"])

        for _ in range(MAX_STEPS):
            neighbors = feasible_neighbors(G, current, demand_bw) # eğer bu node'dan uygun çıkış yoksa episode burada bitiyor
            if not neighbors:
                break

            #EPSILON GREEDY
            if random.random() < epsilon:
                next_node = random.choice(neighbors)   # keşfetme
            else:
                if (current, target) in Q and Q[(current, target)]:
                    next_node = max(
                        neighbors,
                        key=lambda n: Q[(current, target)].get(n, -9999)
                    )
                else:
                    next_node = random.choice(neighbors)

            # ödül hesaplama
            reward = compute_reward(G, current, next_node)
            if next_node == target:
                reward += 100  # hedefe ulaşma bonusu

            #GELECEK ÖDÜL
            best_future = 0
            if (next_node, target) in Q:
                vals = Q[(next_node, target)].values()
                if vals:
                    best_future = max(vals)

            # Q GÜNCELLEME
            Q[(current, target)][next_node] += ALPHA * (
                reward + GAMMA * best_future - Q[(current, target)][next_node]
            )

            current = next_node
            if current == target:
                break
        
        # episode sonunda epsilon azaltılır
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)


#POLİCY TABLOSU VE ROTA BULMA

def extract_policy(Q):

    policy = {}
    for (current, target), actions in Q.items():
        if actions:
            policy[(current, target)] = max(actions, key=actions.get)
    return policy


def get_best_path(policy, src, dst, max_hops=50):

    path = [src]
    current = src

    for _ in range(max_hops):
        if current == dst:
            break
        next_node = policy.get((current, dst))
        if next_node is None or next_node in path:
            break
        path.append(next_node)
        current = next_node

    return path






