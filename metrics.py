import networkx as nx
import pandas as pd
import numpy as np
import math
import sys


#MALİYET VE METRİK HESAPLAMA MODÜLÜ



#GRAPH OLUŞTURMA
def build_graph_from_csv(node_csv, edge_csv):

    print(f"Veriler yükleniyor: {node_csv} ve {edge_csv}...")
    G = nx.Graph()

    try:
        # CSV okuma ayarları:
        # Noktalı virgül ayırıcı
        # Virgül ondalık gösterimi
        nodes = pd.read_csv(node_csv, sep=';', decimal=',')
        edges = pd.read_csv(edge_csv, sep=';', decimal=',')
    except FileNotFoundError:
        print(f"KRİTİK HATA: '{node_csv}' veya '{edge_csv}' bulunamadı.")
        print("Lütfen dosya isimlerinin tamamen küçük harf olduğundan emin olun.")
        return None
    except Exception as e:
        print(f"HATA: Dosyalar okunurken sorun oluştu: {e}")
        return None

    #NODE EKLEME
    #işlem gecikmesi (s_ms)
    #node güvenilirliği (r_node)
    for _, n in nodes.iterrows():
        G.add_node(
            int(n["node_id"]),
            s_ms=float(n["s_ms"]),
            r_node=float(n["r_node"])
        )

    #EDGE EKLEME
    #Her link için:
    #bant genişliği kapasitesi
    #link gecikmesi
    #link güvenilirliği
    for _, e in edges.iterrows():
        G.add_edge(
            int(e["src"]),
            int(e["dst"]),
            capacity_mbps=float(e["capacity_mbps"]),
            delay_ms=float(e["delay_ms"]),
            r_link=float(e["r_link"])
        )
    
    print(f"Ağ Yüklendi: {G.number_of_nodes()} Düğüm, {G.number_of_edges()} Kenar.")
    return G


# DEMAND VERİLERİNİ YÜKLEM

def load_demands(demand_csv):

    try:
        return pd.read_csv(demand_csv, sep=';', decimal=',')
    except FileNotFoundError:
        print(f"KRİTİK HATA: '{demand_csv}' dosyası bulunamadı.")
        return None
    except Exception as e:
        print(f"HATA: Talep dosyası okunurken hata: {e}")
        return None


# YOL GEÇERLİLİK KONTROLÜ
def is_valid_path(G, path):
    """
    Yolun fiziksel olarak ağda var olup olmadığını kontrol eder.
    """
    if not path or len(path) < 2:
        return False

    for i in range(len(path) - 1):
        if not G.has_edge(path[i], path[i+1]):
            return False
    return True


# TOPLAM GECİKME
def total_delay(G, path):

    delay = 0.0
    
    # Link gecikmeleri
    for i in range(len(path) - 1):
        delay += G.edges[path[i], path[i+1]]["delay_ms"]

    # Ara düğümler (S ve D hariç)
    if len(path) > 2:
        for n in path[1:-1]:
            delay += G.nodes[n]["s_ms"]

    return delay


# GÜVENİLİRLİK MALİYETİ
def reliability_cost(G, path):

    cost = 0.0

    # Link güvenilirliği
    for i in range(len(path) - 1):
        r = G.edges[path[i], path[i+1]]["r_link"]
        cost += -math.log(r) if r > 0 else 100

    # Node güvenilirliği
    for n in path:
        r = G.nodes[n]["r_node"]
        cost += -math.log(r) if r > 0 else 100

    return cost


# AĞ KAYNAK KULLANIMI
def resource_cost(G, path):

    cost = 0.0
    for i in range(len(path) - 1):
        cap = G.edges[path[i], path[i+1]]["capacity_mbps"]
        cost += (1000.0 / cap) if cap > 0 else 1000.0
    return cost


# TOTAL COST
def total_cost(G, path, Wd, Wr, Wres):
    """
    Çok kriterli maliyet fonksiyonu.

    total_cost =
      Wd   * delay_cost
    + Wr   * reliability_cost
    + Wres * resource_cost
    """
    # Geçersiz yol için ağır ceza
    if not is_valid_path(G, path):
        return 999999.0

    c_delay = total_delay(G, path)
    c_rel   = reliability_cost(G, path)
    c_res   = resource_cost(G, path)
    
    return (Wd * c_delay) + (Wr * c_rel) + (Wres * c_res)


# OPTİMİZASYON DÖNGÜSÜ
def run_optimization(G, demands, weight_sets):

    results = []
    print(f"{len(demands)} adet talep işleniyor...")

    for idx, d in demands.iterrows():
        try:
            src = int(d["src"])
            dst = int(d["dst"])
            bw  = float(d["demand_mbps"])
        except KeyError:
            print("Hata: demanddata.csv sütun isimleri hatalı")
            return pd.DataFrame()

        #HARD CONSTRAINT
        # Talep edilen bandwidth’i karşılamayan linkler devre dışı bırakılır
        valid_edges = [
            (u, v) for u, v, data in G.edges(data=True)
            if data['capacity_mbps'] >= bw
        ]
        Gf = G.edge_subgraph(valid_edges).copy()
        
        if not nx.has_path(Gf, src, dst):
            print(f"Uyarı: {src} -> {dst} için {bw} Mbps kapasiteli yol bulunamadı.")
            continue

        # Referans yol (baseline)
        path = nx.shortest_path(Gf, src, dst, weight="delay_ms")

        # Farklı ağırlık senaryoları için maliyet hesapla
        for (Wd, Wr, Wres) in weight_sets:
            cost = total_cost(G, path, Wd, Wr, Wres)

            results.append({
                "Request_ID": idx + 1,
                "Source": src,
                "Destination": dst,
                "Demand_Mbps": bw,
                "W_Delay": Wd,
                "W_Reliability": Wr,
                "W_Resource": Wres,
                "Total_Cost": round(cost, 4),
                "Path_Length": len(path)
            })

    return pd.DataFrame(results)


