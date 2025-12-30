import pandas as pd
import networkx as nx
import random
import math
import time  # <-- İŞTE BU EKSİKTİ, EKLENDİ.

# --- YARDIMCI SINIF: NETWORK ---
class Network:
    def __init__(self):
        self.G = nx.Graph()
        
    def load_nodes(self, path):
        try:
            df = pd.read_csv(path, delimiter=";", decimal=",")
            for _, row in df.iterrows():
                self.G.add_node(int(row['node_id']), 
                                s_ms=float(row['s_ms']), 
                                r_node=float(row['r_node']))
        except Exception as e:
            print(f"Node okuma hatası: {e}")

    def load_edges(self, path):
        try:
            df = pd.read_csv(path, delimiter=";", decimal=",")
            for _, row in df.iterrows():
                self.G.add_edge(int(row['src']), int(row['dst']),
                                capacity_mbps=float(row['capacity_mbps']),
                                delay_ms=float(row['delay_ms']),
                                r_link=float(row['r_link']))
        except Exception as e:
            print(f"Edge okuma hatası: {e}")

# --- ANA SINIF: GENETİK ALGORİTMA ---
class GeneticAlgorithm:
    def __init__(self, network, source, dest, demand, pop_size=20, generations=10, mutation_rate=0.1, w_delay=0.33, w_rel=0.33, w_res=0.34):
        self.network = network
        self.G = network.G
        self.source = source
        self.dest = dest
        self.demand = demand
        
        # Performans için popülasyonu küçük tutuyoruz
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        
        self.w_delay = w_delay
        self.w_rel = w_rel
        self.w_res = w_res
        
        self.max_delay = 50 
        self.max_res = 10   

    def create_individual(self):
        """Rastgele ama geçerli bir yol oluşturur."""
        try:
            path = nx.shortest_path(self.G, self.source, self.dest, weight='delay_ms')
            if len(path) > 3:
                idx = random.randint(1, len(path)-2)
                neighbors = list(self.G.neighbors(path[idx-1]))
                if len(neighbors) > 1:
                    new_node = random.choice(neighbors)
                    return path
            return path
        except:
            return []

    def fitness(self, individual):
        if not individual or individual[0] != self.source or individual[-1] != self.dest:
            return float('inf')
        
        total_delay = 0
        total_rel_log = 0
        total_res = 0
        
        for i in range(len(individual) - 1):
            u, v = individual[i], individual[i+1]
            data = self.G.get_edge_data(u, v)
            if not data: return float('inf')
            if data.get('capacity_mbps', 0) < self.demand:
                return float('inf')
            
            total_delay += data.get('delay_ms', 0)
            r_val = data.get('r_link', 0.99)
            if r_val <= 0: r_val = 0.0001
            total_rel_log += -math.log(r_val)
            cap = data.get('capacity_mbps', 1)
            total_res += (1000 / cap)
            
        for node in individual:
            ndata = self.G.nodes[node]
            total_delay += ndata.get('s_ms', 0)
            nr_val = ndata.get('r_node', 0.99)
            if nr_val <= 0: nr_val = 0.0001
            total_rel_log += -math.log(nr_val)

        cost = (self.w_delay * total_delay) + \
               (self.w_rel * total_rel_log * 100) + \
               (self.w_res * total_res)
        return cost

    def crossover(self, parent1, parent2):
        common_nodes = [node for node in parent1 if node in parent2 and node != self.source and node != self.dest]
        if not common_nodes: return parent1 
        split_node = random.choice(common_nodes)
        idx1 = parent1.index(split_node)
        idx2 = parent2.index(split_node)
        child = parent1[:idx1] + parent2[idx2:]
        if len(child) != len(set(child)): return parent1
        return child

    def mutate(self, individual):
        return self.create_individual()

    def run(self):
        # 1. BAŞLANGIÇ POPÜLASYONU OLUŞTURMA (GÜNCELLENDİ)
        population = []
        try:
            # Sadece en kısa yolu değil, en iyi 'pop_size' kadar alternatif yolu buluyoruz.
            # Bu sayede elimizde hem kısa hem de alternatif (belki daha güvenli) yollar oluyor.
            # weight='delay_ms' diyerek gecikmeye göre sıralı alternatifleri alıyoruz.
            k_paths_generator = nx.shortest_simple_paths(self.G, self.source, self.dest, weight='delay_ms')
            
            for _ in range(self.pop_size):
                try:
                    path = next(k_paths_generator)
                    population.append(path)
                except StopIteration:
                    break # Başka alternatif yol kalmadıysa dur
        except Exception as e:
            # Eğer yol hiç yoksa veya hata olursa boş dön
            print(f"Yol bulunamadı: {e}")
            return []
            
        if not population: return []

        # Eğer popülasyon dolmadıysa (yeterince alternatif yoksa), kalanları rastgele doldurmaya çalışma.
        # Var olanlar üzerinden evrimleşsin.

        # 2. JENERASYON DÖNGÜSÜ
        for _ in range(self.generations):
            # Sırala (Fitness değeri DÜŞÜK olan daha iyidir)
            population.sort(key=self.fitness)
            
            # En iyileri seç (Elitizm - En iyi %50)
            new_population = population[:int(len(population)/2)]
            
            # Eğer popülasyon çok azaldıysa döngüyü kır (Hata önleyici)
            if len(new_population) < 2:
                population = new_population
                break

            # Çocuk üret (Crossover)
            # Popülasyon sayısını korumak için eksilen kadar yeni birey üret
            while len(new_population) < self.pop_size:
                p1 = random.choice(new_population)
                p2 = random.choice(new_population)
                
                child = self.crossover(p1, p2)
                
                # Mutasyon: Bazen yolu değiştirmeyi dene
                if random.random() < self.mutation_rate:
                    mutated = self.mutate(child)
                    if mutated: child = mutated # Eğer geçerli bir yol döndüyse al
                
                new_population.append(child)
            
            population = new_population

        # En son eldeki en iyi yolu döndür
        best_path = min(population, key=self.fitness)
        return best_path
# ------------------------------------------------------------------
#  OTOMATİK TEST VE RAPORLAMA KODLARI
#  Çalıştırmak isterseniz aşağıdaki üç tırnakları (""") kaldırın.
# ------------------------------------------------------------------

if __name__ == "__main__":
    print("\\n[INFO] GA Otomatik test sureci baslatiliyor (Kapsam: 20 Senaryo x 5 Tekrar)...")
    
    # Network Yükle
    net_obj = Network()
    # Dosya yollarını kontrol et
    try:
        net_obj.load_nodes("NodeData.csv")
        net_obj.load_edges("EdgeData.csv")
    except:
        print("HATA: Veri dosyaları bulunamadı.")

    # Talepleri Yükle
    try:
        demands = pd.read_csv("demanddata.csv", sep=';')
    except:
        print("HATA: demanddata.csv bulunamadı.")
        demands = pd.DataFrame()

    test_results = []

    # Deney Tekrar Döngüsü (5 Tekrar)
    for repetition in range(1, 6):
        print(f"\\n[INFO] Deney Seti: {repetition}/5 calistiriliyor...")
        
        # Ilk 20 Senaryonun Test Edilmesi
        for i in range(20):
            if i >= len(demands): break
            try:
                row = demands.iloc[i]
                src = int(row["src"])
                dst = int(row["dst"])
                bw = row["demand_mbps"]
                
                # GA Nesnesi Oluştur ve Çalıştır
                ga = GeneticAlgorithm(net_obj, src, dst, bw)
                
                start_time = time.time()
                path = ga.run()
                end_time = time.time()
                
                raw_duration = end_time - start_time
                duration_tr = f"{raw_duration:.6f}".replace('.', ',')
                
                result_record = {
                    "Algoritma": "Genetic Algorithm",
                    "Tekrar_No": repetition,
                    "Senaryo_ID": i,
                    "Kaynak_Node": src,
                    "Hedef_Node": dst,
                    "Talep_Mbps": bw,
                    "Bulunan_Rota": str(path),
                    "Islem_Suresi_sn": duration_tr
                }
                test_results.append(result_record)
                print(f"   -> Senaryo {i} tamamlandi. (Sure: {duration_tr}s)")
                
            except Exception as e:
                print(f"   [ERROR] Senaryo {i} sirasinda hata olustu: {e}")

    # --- SONUÇLARI KAYDET ---
    print("\\n[INFO] Test sonuclari CSV dosyasina yaziliyor...")
    try:
        df_results = pd.DataFrame(test_results)
        output_filename = "GA_Final_Test_Sonuclari_TR.csv"
        df_results.to_csv(output_filename, sep=";", index=False)
        print(f"[SUCCESS] Islem basariyla tamamlandi. Yeni dosya: {output_filename}")
    except Exception as e:
        print(f"[ERROR] Dosya kaydetme hatasi: {e}")

