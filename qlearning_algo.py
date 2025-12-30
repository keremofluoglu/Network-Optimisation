import pandas as pd
import networkx as nx
import random
import math
from collections import defaultdict

# --- AYARLAR ---
# GUI için varsayılan değerler
EPISODES = 500  # Arayüzde hızlı sonuç için
MAX_STEPS = 50  # Bir rota en fazla 50 adım olabilir (Döngüye girmesin diye)
ALPHA = 0.1     # Öğrenme hızı
GAMMA = 0.9     # Gelecek ödülün önemi

# Epsilon Ayarları (YENİ EKLENDİ - KRİTİK DÜZELTME)
EPSILON_START = 1.0   # Başlangıçta %100 rastgele
EPSILON_END = 0.05    # En son %5 rastgele (Hata payı)
EPSILON_DECAY = 0.99  # Her turda %1 azalacak

# Global Değişkenler
Q = defaultdict(lambda: defaultdict(float))
w_d = 1.0
w_r = 1.0
w_u = 1.0

# Normalizasyon Değişkenleri
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
        MAX_DELAY = nodes["s_ms"].quantile(0.95) + edges["delay_ms"].quantile(0.95)
        
        min_cap = edges["capacity_mbps"].quantile(0.05)
        if min_cap > 0: MAX_RESOURCE = 1000 / min_cap
            
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

    for _, n in nodes.iterrows():
        G.add_node(int(n["node_id"]), s_ms=float(n["s_ms"]), r_node=float(n["r_node"]))

    for _, e in edges.iterrows():
        G.add_edge(int(e["src"]), int(e["dst"]),
                   capacity_mbps=float(e["capacity_mbps"]),
                   delay_ms=float(e["delay_ms"]),
                   r_link=float(e["r_link"]))

    print(f"Ağ Yüklendi: {G.number_of_nodes()} node, {G.number_of_edges()} edge")
    return G

def load_demands(demand_csv):
    try:
        return pd.read_csv(demand_csv, sep=';', decimal=',')
    except Exception:
        return None

def compute_reward(G, u, v):
    edge = G[u][v]
    node = G.nodes[v]

    delay = edge["delay_ms"] + node["s_ms"]
    resource = 1000 / edge["capacity_mbps"] if edge["capacity_mbps"] > 0 else 9999
    
    r_link = edge["r_link"] if edge["r_link"] > 0 else 0.0001
    r_node = node["r_node"] if node["r_node"] > 0 else 0.0001
    reliability = -math.log(r_link) - math.log(r_node)

    delay_norm = delay / MAX_DELAY if MAX_DELAY > 0 else 0
    resource_norm = resource / MAX_RESOURCE if MAX_RESOURCE > 0 else 0
    reliability_norm = reliability / MAX_RELIABILITY if MAX_RELIABILITY > 0 else 0

    total_cost = (w_d * delay_norm + w_r * resource_norm + w_u * reliability_norm)
    
    # Maliyet ne kadar azsa ödül o kadar büyük olmalı (Negatif Maliyet)
    return -10.0 * total_cost

def feasible_neighbors(G, current_node, demand_bw):
    valid_neighbors = []
    for n in G.neighbors(current_node):
        edge = G[current_node][n]
        if edge["capacity_mbps"] >= demand_bw:
            valid_neighbors.append(n)
    return valid_neighbors

def train_q_learning(G, demands):
    # Eğer tek satır geldiyse DataFrame'e çevir
    if isinstance(demands, pd.Series):
        demands = pd.DataFrame([demands])

    # EPSILON (Keşfetme Oranı) her seferinde sıfırlanmalı ki yeniden öğrensin
    epsilon = EPSILON_START

    for episode in range(EPISODES):
        if demands.empty: break
        
        d = demands.sample(1).iloc[0]
        current = int(d["src"])
        target = int(d["dst"])
        demand_bw = float(d["demand_mbps"])

        for _ in range(MAX_STEPS):
            neighbors = feasible_neighbors(G, current, demand_bw)
            if not neighbors:
                break

            # DÜZELTME: Epsilon artık sabit değil, azalıyor
            if random.random() < epsilon:
                next_node = random.choice(neighbors) # Rastgele (Keşfet)
            else:
                # Bilinen en iyi yolu seç (Sömür)
                if (current, target) in Q and Q[(current, target)]:
                    next_node = max(neighbors, key=lambda n: Q[(current, target)].get(n, -9999))
                else:
                    next_node = random.choice(neighbors)

            reward = compute_reward(G, current, next_node)
            if next_node == target: reward += 100

            best_future = 0
            future_neighbors = list(G.neighbors(next_node))
            if future_neighbors and (next_node, target) in Q:
                 vals = [Q[(next_node, target)].get(n, 0) for n in future_neighbors]
                 if vals: best_future = max(vals)

            Q[(current, target)][next_node] += ALPHA * (reward + GAMMA * best_future - Q[(current, target)][next_node])

            current = next_node
            if current == target: break
        
        # Her bölüm sonunda Epsilon'u azalt (Akıllanma süreci)
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

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
        if current == dst: break
        next_node = policy.get((current, dst))
        if next_node is None: break
        if next_node in path: break # Döngü engelleme
        path.append(next_node)
        current = next_node
    return path

# --- STANDART ÇALIŞMA (GUI İÇİN) ---
if __name__ == "__main__":
    G = build_graph_from_csv("nodedata.csv", "edgedata.csv")
    demands = load_demands("demanddata.csv")
    if G and demands is not None:
        train_q_learning(G, demands)
        print("Modül Hazır.")

# ------------------------------------------------------------------
#  OTOMATİK TEST VE RAPORLAMA KODLARI
#  Bu bölüm sadece dosya doğrudan çalıştırıldığında çalışır.
# ------------------------------------------------------------------
"""
import time

if __name__ == "__main__":
    print("\\n[INFO] QL Otomatik test sureci baslatiliyor (Kapsam: 20 Senaryo x 5 Tekrar)...")
    
    # --- KRİTİK NOKTA: TEST İÇİN ZEKA SEVİYESİNİ ARTIRIYORUZ ---
    # Normalde arayüz donmasın diye 500 kullanıyoruz.
    # Ama Excel raporunda rotalar KISA ve DOĞRU çıksın diye
    # burada geçici olarak Kerem'in kullandığı sayıya (20.000) çıkarıyoruz.
    globals()['EPISODES'] = 20000 
    
    # Testler için global G nesnesini yükle
    G = build_graph_from_csv("nodedata.csv", "edgedata.csv")
    demands = load_demands("demanddata.csv")

    test_results = []

    # Deney Tekrar Döngüsü (5 Tekrar)
    for repetition in range(1, 6):
        print(f"\\n[INFO] Deney Seti: {repetition}/5 calistiriliyor. (20.000 Tur Egitim - Lutfen Bekleyiniz)...")
        
        # Q-Table Sifirlama ve Modelin Yeniden Egitilmesi
        Q.clear() 
        train_q_learning(G, demands)
        policy = extract_policy(Q)
        
        # Ilk 20 Senaryonun Test Edilmesi
        for i in range(20):
            try:
                row = demands.iloc[i]
                src = int(row["src"])
                dst = int(row["dst"])
                bw = row["demand_mbps"]
                
                # Performans Olcumu (Sure)
                start_time = time.time()
                # max_hops=50 sınırı koyuyoruz ki sonsuz döngü olmasın
                path = get_best_path(policy, src, dst, max_hops=50)
                end_time = time.time()
                
                raw_duration = end_time - start_time
                duration_tr = f"{raw_duration:.6f}".replace('.', ',')
                
                result_record = {
                    "Algoritma": "Q-Learning",
                    "Tekrar_No": repetition,
                    "Senaryo_ID": i,
                    "Kaynak_Node": src,
                    "Hedef_Node": dst,
                    "Talep_Mbps": bw,
                    "Bulunan_Rota": str(path),
                    "Rota_Uzunlugu": len(path) - 1,
                    "Islem_Suresi_sn": duration_tr
                }
                test_results.append(result_record)
                
                # Ekrana sade bilgi basıyoruz
                print(f"   -> Senaryo {i} bitti. Rota Uzunlugu: {len(path)-1} (Sure: {duration_tr}s)")
                
            except Exception as e:
                print(f"   [ERROR] Senaryo {i} sirasinda hata olustu: {e}")

    # --- SONUÇLARI KAYDET ---
    print("\\n[INFO] Test sonuclari CSV dosyasina yaziliyor...")
    try:
        df_results = pd.DataFrame(test_results)
        output_filename = "QLearning_Final_Test_Sonuclari_TR.csv"
        df_results.to_csv(output_filename, sep=";", index=False)
        print(f"[SUCCESS] Islem basariyla tamamlandi. Yeni dosya: {output_filename}")
    except Exception as e:
        print(f"[ERROR] Dosya kaydetme hatasi: {e}")
"""

