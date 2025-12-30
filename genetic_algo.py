import pandas as pd
import networkx as nx
import random
import math
import time  # Algoritmanın çalışma süresini ölçmek için kullanılır

#  Ağ topolojisini (node ve edge’ler) yüklemekten sorumlu
class Network:
    def __init__(self):
        # NetworkX graph yapısı
        self.G = nx.Graph()
        
    def load_nodes(self, path):

        try:
            df = pd.read_csv(path, delimiter=";", decimal=",")
            for _, row in df.iterrows():
                self.G.add_node(
                    int(row['node_id']),
                    s_ms=float(row['s_ms']),
                    r_node=float(row['r_node'])
                )
        except Exception as e:
            print(f"Node okuma hatası: {e}")

    def load_edges(self, path):

        try:
            df = pd.read_csv(path, delimiter=";", decimal=",")
            for _, row in df.iterrows():
                self.G.add_edge(
                    int(row['src']),
                    int(row['dst']),
                    capacity_mbps=float(row['capacity_mbps']),
                    delay_ms=float(row['delay_ms']),
                    r_link=float(row['r_link'])
                )
        except Exception as e:
            print(f"Edge okuma hatası: {e}")

#GENETİK ALGORİTMA

class GeneticAlgorithm:
    def __init__(
        self, network, source, dest, demand,
        pop_size=20, generations=10, mutation_rate=0.1,
        w_delay=0.33, w_rel=0.33, w_res=0.34
    ):
        # Ağ bilgileri
        self.network = network
        self.G = network.G
        
        # Routing bilgileri
        self.source = source
        self.dest = dest
        self.demand = demand  # Minimum gerekli bant genişliği
        
        # GA parametreleri
        self.pop_size = pop_size          # Popülasyon büyüklüğü
        self.generations = generations    # Jenerasyon sayısı
        self.mutation_rate = mutation_rate
        
        # Cost ağırlıkları (arayüzden değiştirilebilir)
        self.w_delay = w_delay
        self.w_rel = w_rel
        self.w_res = w_res
        
        # Normalizasyon için üst sınırlar
        self.max_delay = 50
        self.max_res = 10

    def create_individual(self):
        """
        Rastgele fakat geçerli bir yol (birey) üretir.
        Başlangıç olarak en kısa yolu alır, küçük bir varyasyon dener.
        """
        try:
            path = nx.shortest_path(
                self.G, self.source, self.dest, weight='delay_ms'
            )
            
            # Yol üzerinde küçük bir rastgele değişiklik denemesi
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

        # Geçersiz yol kontrolü
        if not individual or individual[0] != self.source or individual[-1] != self.dest:
            return float('inf')
        
        total_delay = 0
        total_rel_log = 0
        total_res = 0
        
        # Edge bazlı maliyet hesapları
        for i in range(len(individual) - 1):
            u, v = individual[i], individual[i+1]
            data = self.G.get_edge_data(u, v)
            if not data:
                return float('inf')
            
            # HARD CONSTRAINT: Bant genişliği yetersizse yol geçersiz
            if data.get('capacity_mbps', 0) < self.demand:
                return float('inf')
            
            # Gecikme
            total_delay += data.get('delay_ms', 0)
            
            # Güvenilirlik (log dönüşümü ile çarpan etkisi azaltılır)
            r_val = data.get('r_link', 0.99)
            if r_val <= 0:
                r_val = 0.0001
            total_rel_log += -math.log(r_val)
            
            # Kaynak kullanımı (kapasite azaldıkça ceza artar)
            cap = data.get('capacity_mbps', 1)
            total_res += (1000 / cap)
            
        # Node bazlı maliyetler
        for node in individual:
            ndata = self.G.nodes[node]
            total_delay += ndata.get('s_ms', 0)
            
            nr_val = ndata.get('r_node', 0.99)
            if nr_val <= 0:
                nr_val = 0.0001
            total_rel_log += -math.log(nr_val)

        # Ağırlıklı toplam maliyet
        cost = (
            self.w_delay * total_delay +
            self.w_rel * total_rel_log * 100 +
            self.w_res * total_res
        )
        return cost

    def crossover(self, parent1, parent2):

        common_nodes = [
            node for node in parent1
            if node in parent2 and node != self.source and node != self.dest
        ]
        
        if not common_nodes:
            return parent1
        
        split_node = random.choice(common_nodes)
        idx1 = parent1.index(split_node)
        idx2 = parent2.index(split_node)
        
        child = parent1[:idx1] + parent2[idx2:]
        
        # Döngü oluşmasını engelle
        if len(child) != len(set(child)):
            return parent1
        
        return child

    def mutate(self, individual):

        return self.create_individual()

    def run(self):

        # 1) BAŞLANGIÇ POPÜLASYONU OLUŞTURMA
        population = []
        try:
            # Gecikmeye göre en kısa alternatif yollar üretilir
            k_paths_generator = nx.shortest_simple_paths(
                self.G, self.source, self.dest, weight='delay_ms'
            )
            
            for _ in range(self.pop_size):
                try:
                    path = next(k_paths_generator)
                    population.append(path)
                except StopIteration:
                    break
        except Exception as e:
            print(f"Yol bulunamadı: {e}")
            return []
            
        if not population:
            return []


        # 2) JENERASYON DÖNGÜSÜ
        for _ in range(self.generations):
            # Fitness’a göre sırala (küçük daha iyi)
            population.sort(key=self.fitness)
            
            # Elitizm: En iyi %50 korunur
            new_population = population[:int(len(population)/2)]
            
            if len(new_population) < 2:
                population = new_population
                break

            # Crossover + Mutasyon ile yeni bireyler üret
            while len(new_population) < self.pop_size:
                p1 = random.choice(new_population)
                p2 = random.choice(new_population)
                
                child = self.crossover(p1, p2)
                
                if random.random() < self.mutation_rate:
                    mutated = self.mutate(child)
                    if mutated:
                        child = mutated
                
                new_population.append(child)
            
            population = new_population

        # En iyi yolu döndür
        best_path = min(population, key=self.fitness)
        return best_path

