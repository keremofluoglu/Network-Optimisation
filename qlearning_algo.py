import pandas as pd
import networkx as nx
import random
import math
from collections import defaultdict

# ============================================================
# Q-LEARNING TABANLI AĞ YÖNLENDİRME ALGORİTMASI
#
# Bu modül:
#  - Ağ yönlendirme problemini Reinforcement Learning ile çözer
#  - Çok kriterli cost modelini reward fonksiyonuna dönüştürür
#  - Demand bandwidth kısıtını hard constraint olarak uygular
#  - Tek bir Q-tablosu ile farklı trafik taleplerini destekler
# ============================================================


# ============================================================
# 1. GLOBAL AYARLAR ve HİPERPARAMETRELER
# ============================================================

# Eğitim parametreleri (GUI için düşük tutulmuştur)
EPISODES = 500          # Eğitim turu sayısı
MAX_STEPS = 50          # Bir rotadaki maksimum hop sayısı

# Q-Learning parametreleri
ALPHA = 0.1             # Öğrenme hızı (Learning Rate)
GAMMA = 0.9             # İndirgeme faktörü (Discount Factor)

# ---------- EPSILON-GREEDY STRATEJİSİ ----------
# Keşif (exploration) → Sömürü (exploitation) dengesi
EPSILON_START = 1.0     # Başlangıçta %100 rastgele
EPSILON_END = 0.05      # Minimum keşif oranı
EPSILON_DECAY = 0.99    # Her episode sonrası azalma oranı


# ============================================================
# 2. GLOBAL DEĞİŞKENLER
# ============================================================

# Q-Tablosu:
# Q[(current_node, target_node)][next_node] = Q-değeri
Q = defaultdict(lambda: defaultdict(float))

# Cost fonksiyonu ağırlıkları (GUI üzerinden değiştirilebilir)
w_d = 1.0   # Delay ağırlığı
w_r = 1.0   # Resource ağırlığı
w_u = 1.0   # Reliability ağırlığı


# ============================================================
# 3. NORMALİZASYON ÜST SINIRLARI
# ============================================================

# Farklı metriklerin aynı reward ölçeğinde etkili olması için
# maksimum değerler istatistiksel olarak belirlenir
MAX_DELAY = 100.0
MAX_RESOURCE = 100.0
MAX_RELIABILITY = 10.0


# ============================================================
# 4. GRAPH OLUŞTURMA
# ============================================================
def build_graph_from_csv(node_csv, edge_csv):
    """
    Node ve Edge CSV dosyalarından NetworkX graph oluşturur
    ve normalizasyon için üst sınırları hesaplar.
    """
    print(f"Veriler yükleniyor: {node_csv} ve {edge_csv}...")
    G = nx.Graph()

    try:
        nodes = pd.read_csv(node_csv, sep=';', decimal=',')
        edges = pd.read_csv(edge_csv, sep=';', decimal=',')
        
        # ---------- NORMALİZASYON ÜST SINIRLARI ----------
        global MAX_DELAY, MAX_RESOURCE, MAX_RELIABILITY
        
        # Delay için %95 persentil kullanılır (outlier etkisini azaltır)
        MAX_DELAY = nodes["s_ms"].quantile(0.95) + edges["delay_ms"].quantile(0.95)
        
        # Resource maliyeti için düşük kapasiteye göre üst sınır
        min_cap = edges["capacity_mbps"].quantile(0.05)
        if min_cap > 0:
            MAX_RESOURCE = 1000 / min_cap
            
        # Reliability için en kötü olası senaryo
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

    # Node ekleme
    for _, n in nodes.iterrows():
        G.add_node(
            int(n["node_id"]),
            s_ms=float(n["s_ms"]),
            r_node=float(n["r_node"])
        )

    # Edge ekleme
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


# ============================================================
# 5. DEMAND VERİLERİ
# ============================================================
def load_demands(demand_csv):
    """
    Trafik taleplerini (src, dst, demand_mbps) yükler.
    """
    try:
        return pd.read_csv(demand_csv, sep=';', decimal=',')
    except Exception:
        return None


# ============================================================
# 6. REWARD (ÖDÜL) FONKSİYONU
# ============================================================
def compute_reward(G, u, v):
    """
    Bir aksiyonun (u → v) ödülünü hesaplar.

    Reward = - (Normalize edilmiş toplam maliyet)
    """
    edge = G[u][v]
    node = G.nodes[v]

    # ---------- HAM METRİKLER ----------
    delay = edge["delay_ms"] + node["s_ms"]
    resource = 1000 / edge["capacity_mbps"] if edge["capacity_mbps"] > 0 else 9999
    
    r_link = edge["r_link"] if edge["r_link"] > 0 else 0.0001
    r_node = node["r_node"] if node["r_node"] > 0 else 0.0001
    reliability = -math.log(r_link) - math.log(r_node)

    # ---------- NORMALİZASYON ----------
    delay_norm = delay / MAX_DELAY if MAX_DELAY > 0 else 0
    resource_norm = resource / MAX_RESOURCE if MAX_RESOURCE > 0 else 0
    reliability_norm = reliability / MAX_RELIABILITY if MAX_RELIABILITY > 0 else 0

    # ---------- TOTAL COST ----------
    total_cost = (
        w_d * delay_norm +
        w_r * resource_norm +
        w_u * reliability_norm
    )
    
    # Maliyet minimizasyonu → Reward maksimizasyonu
    return -10.0 * total_cost


# ============================================================
# 7. HARD CONSTRAINT: FEASIBLE NEIGHBORS
# ============================================================
def feasible_neighbors(G, current_node, demand_bw):
    """
    Minimum bandwidth talebini karşılayan komşuları döndürür.
    Bu kısıt ödül fonksiyonuna değil,
    doğrudan aksiyon uzayına uygulanır (hard constraint).
    """
    valid_neighbors = []
    for n in G.neighbors(current_node):
        if G[current_node][n]["capacity_mbps"] >= demand_bw:
            valid_neighbors.append(n)
    return valid_neighbors


# ============================================================
# 8. Q-LEARNING EĞİTİM DÖNGÜSÜ
# ============================================================
def train_q_learning(G, demands):
    """
    Tek bir Q-tablosu ile birden fazla demand üzerinden eğitim yapar.
    """
    if isinstance(demands, pd.Series):
        demands = pd.DataFrame([demands])

    epsilon = EPSILON_START

    for episode in range(EPISODES):
        if demands.empty:
            break
        
        # Rastgele bir demand seçilir
        d = demands.sample(1).iloc[0]
        current = int(d["src"])
        target = int(d["dst"])
        demand_bw = float(d["demand_mbps"])

        for _ in range(MAX_STEPS):
            neighbors = feasible_neighbors(G, current, demand_bw)
            if not neighbors:
                break

            # ---------- EPSILON-GREEDY AKSİYON SEÇİMİ ----------
            if random.random() < epsilon:
                next_node = random.choice(neighbors)   # Keşfet
            else:
                if (current, target) in Q and Q[(current, target)]:
                    next_node = max(
                        neighbors,
                        key=lambda n: Q[(current, target)].get(n, -9999)
                    )
                else:
                    next_node = random.choice(neighbors)

            # ---------- ÖDÜL ----------
            reward = compute_reward(G, current, next_node)
            if next_node == target:
                reward += 100  # Hedefe ulaşma bonusu

            # ---------- GELECEK ÖDÜL ----------
            best_future = 0
            if (next_node, target) in Q:
                vals = Q[(next_node, target)].values()
                if vals:
                    best_future = max(vals)

            # ---------- Q-GÜNCELLEME ----------
            Q[(current, target)][next_node] += ALPHA * (
                reward + GAMMA * best_future - Q[(current, target)][next_node]
            )

            current = next_node
            if current == target:
                break
        
        # Episode sonunda epsilon azaltılır
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)


# ============================================================
# 9. POLICY ÇIKARIMI ve ROTA BULMA
# ============================================================
def extract_policy(Q):
    """
    Q-tablosundan greedy policy çıkarır.
    """
    policy = {}
    for (current, target), actions in Q.items():
        if actions:
            policy[(current, target)] = max(actions, key=actions.get)
    return policy


def get_best_path(policy, src, dst, max_hops=50):
    """
    Öğrenilmiş policy üzerinden greedy şekilde rota oluşturur.
    """
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
