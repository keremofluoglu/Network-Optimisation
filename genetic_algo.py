import pandas as pd
import networkx as nx
import random
import math
import time  # algoritmanın çalışma süresini ölçmek için kullanılır

#  ağ topolojisini (node ve edge’ler) yüklemekten sorumlu
class Network:
    def __init__(self):
        # networkX graph yapısı
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
        # ağ bilgileri
        self.network = network
        self.G = network.G
        
        # routing bilgileri
        self.source = source
        self.dest = dest
        self.demand = demand  # minimum gerekli bant genişliği
        
        # GA parametreleri
        self.pop_size = pop_size          # popülasyon büyüklüğü
        self.generations = generations    # jenerasyon sayısı
        self.mutation_rate = mutation_rate
        
        # cost ağırlıkları (arayüzden değiştirilebilir)
        self.w_delay = w_delay
        self.w_rel = w_rel
        self.w_res = w_res
        
        # normalizasyon için üst sınırlar
        self.max_delay = 50
        self.max_res = 10

    def create_individual(self):

        try:
            path = nx.shortest_path(
                self.G, self.source, self.dest, weight='delay_ms'
            )
            
            # yol üzerinde küçük bir rastgele değişiklik denemesi
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

        # geçersiz yol kontrolü
        if not individual or individual[0] != self.source or individual[-1] != self.dest:
            return float('inf')
        
        total_delay = 0
        total_rel_log = 0
        total_res = 0
        
        # edge bazlı maliyet hesapları
        for i in range(len(individual) - 1):
            u, v = individual[i], individual[i+1]
            data = self.G.get_edge_data(u, v)
            if not data:
                return float('inf')
            
            # bant genişliği yetersizse yol geçersiz
            if data.get('capacity_mbps', 0) < self.demand:
                return float('inf')
            
            # gecikme
            total_delay += data.get('delay_ms', 0)
            
            # güvenilirlik (log dönüşümü ile çarpan etkisi azaltılır)
            r_val = data.get('r_link', 0.99)
            if r_val <= 0:
                r_val = 0.0001
            total_rel_log += -math.log(r_val)
            
            # kaynak kullanımı (kapasite azaldıkça ceza artar)
            cap = data.get('capacity_mbps', 1)
            total_res += (1000 / cap)
            
        # node bazlı maliyetler
        for node in individual:
            ndata = self.G.nodes[node]
            total_delay += ndata.get('s_ms', 0)
            
            nr_val = ndata.get('r_node', 0.99)
            if nr_val <= 0:
                nr_val = 0.0001
            total_rel_log += -math.log(nr_val)

        # ağırlıklı toplam maliyet
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
        
        # döngü oluşmasını engelle
        if len(child) != len(set(child)):
            return parent1
        
        return child

    def mutate(self, individual):

        return self.create_individual()

    def run(self):

        # BAŞLANGIÇ POPÜLASYONU OLUŞTURMA
        population = []
        try:
            # gecikmeye göre en kısa alternatif yollar üretilir
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


        #JENERASYON DÖNGÜSÜ
        for _ in range(self.generations):
            # fitnessa göre sırala 
            population.sort(key=self.fitness)
            
            # elitizimde en iyi %50 korunur
            new_population = population[:int(len(population)/2)]
            
            if len(new_population) < 2:
                population = new_population
                break

            # crossover + mutasyon ile yeni bireyler üret
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

        # en iyi yolu döndür
        best_path = min(population, key=self.fitness)
        return best_path


