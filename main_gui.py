import site
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from tkinter import messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import networkx as nx
import pandas as pd

# Modülleri yükle
try:
    import genetic_algo  
    import qlearning_algo 
    import metrics       
    print("Modüller başarıyla yüklendi.")
except ImportError as e:
    print(f"UYARI: Modüller tam yüklenemedi ({e}).")

NODE_FILE = "NodeData.csv"
EDGE_FILE = "EdgeData.csv"

class NetworkProjectGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("BSM307 - Akıllı Ağ Optimizasyonu v2.0")
        self.root.geometry("1350x900")
        self.G = None
        self.pos = None
        self.setup_ui()
        self.load_and_draw_initial_graph()
        
    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=BOTH, expand=YES)

        control_panel = ttk.Frame(main_frame)
        control_panel.pack(side=LEFT, fill=Y, padx=(0, 15))
        
        header_frame = ttk.Frame(control_panel)
        header_frame.pack(fill=X, pady=(0, 10))
        ttk.Label(header_frame, text="QoS ROTALAMA", font=("Helvetica", 22, "bold"), bootstyle="primary").pack(anchor=W)
        
        pnl_route = ttk.Labelframe(control_panel, text="Güzergah Ayarları", padding=15, bootstyle="info")
        pnl_route.pack(fill=X, pady=10)
        
        ttk.Label(pnl_route, text="Kaynak (Source):", bootstyle="inverse-info").pack(anchor=W)
        self.cb_source = ttk.Combobox(pnl_route, state="readonly", bootstyle="info")
        self.cb_source.pack(fill=X, pady=(2, 10))
        
        ttk.Label(pnl_route, text="Hedef (Destination):", bootstyle="inverse-info").pack(anchor=W)
        self.cb_target = ttk.Combobox(pnl_route, state="readonly", bootstyle="info")
        self.cb_target.pack(fill=X, pady=(2, 10))
        
        pnl_algo = ttk.Labelframe(control_panel, text="Algoritma Motoru", padding=15, bootstyle="warning")
        pnl_algo.pack(fill=X, pady=10)
        
        self.algo_var = ttk.StringVar(value="GA")
        ttk.Radiobutton(pnl_algo, text="Genetik Algoritma (GA)", variable=self.algo_var, value="GA", bootstyle="warning-toolbutton").pack(fill=X, pady=2)
        ttk.Radiobutton(pnl_algo, text="Pekiştirmeli Öğrenme (RL)", variable=self.algo_var, value="QL", bootstyle="warning-toolbutton").pack(fill=X, pady=2)
        
        pnl_weights = ttk.Labelframe(control_panel, text="QoS Öncelikleri", padding=15, bootstyle="success")
        pnl_weights.pack(fill=X, pady=10)
        
        self.scale_delay = self.create_meter(pnl_weights, "Gecikme (Delay)", 0.4)
        self.scale_rel = self.create_meter(pnl_weights, "Güvenilirlik (Rel.)", 0.3)
        self.scale_res = self.create_meter(pnl_weights, "Kaynak (Resource)", 0.3)
        
        pnl_result = ttk.Labelframe(control_panel, text="Analiz Sonuçları", padding=15, bootstyle="danger")
        pnl_result.pack(fill=X, pady=5)
        
        self.lbl_path = ttk.Label(pnl_result, text="Rota: -", font=("Consolas", 10, "bold"), wraplength=250)
        self.lbl_path.pack(anchor=W)

        self.lbl_gecikme = ttk.Label(pnl_result, text="Toplam Gecikme: -", font=("Helvetica", 10), bootstyle="warning")
        self.lbl_gecikme.pack(anchor=W, pady=2)

        self.lbl_guvenilirlik = ttk.Label(pnl_result, text="Güvenilirlik Oranı: -", font=("Helvetica", 10), bootstyle="success")
        self.lbl_guvenilirlik.pack(anchor=W, pady=2)

        self.lbl_kaynak = ttk.Label(pnl_result, text="Ağ Kaynak Kullanımı: -", font=("Helvetica", 10), bootstyle="primary")
        self.lbl_kaynak.pack(anchor=W, pady=2)

        self.lbl_cost = ttk.Label(pnl_result, text="Durum: Bekleniyor...", font=("Helvetica", 9, "bold"), bootstyle="danger")
        self.lbl_cost.pack(anchor=W, pady=(5,0))

        self.btn_run = ttk.Button(control_panel, text="⚡ OPTİMİZASYONU BAŞLAT", command=self.run_optimization, bootstyle="danger", width=25)
        self.btn_run.pack(fill=X, pady=20)

        graph_panel = ttk.Frame(main_frame)
        graph_panel.pack(side=RIGHT, fill=BOTH, expand=YES)
        
        plt.style.use('dark_background')
        self.figure = plt.Figure(figsize=(8, 6), dpi=100, facecolor='#060606') 
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor('#060606')
        self.ax.axis('off')
        
        self.canvas = FigureCanvasTkAgg(self.figure, graph_panel)
        self.canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=YES)
        
        toolbar = NavigationToolbar2Tk(self.canvas, graph_panel)
        toolbar.config(background='#060606')
        toolbar._message_label.config(background='#060606', foreground='#00f2ff')
        toolbar.update()
        self.canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=YES)

    def create_meter(self, parent, label, default):
        frame = ttk.Frame(parent)
        frame.pack(fill=X, pady=5)
        header_frame = ttk.Frame(frame)
        header_frame.pack(fill=X)
        ttk.Label(header_frame, text=label, font=("Helvetica", 8)).pack(side=LEFT)
        value_label = ttk.Label(header_frame, text=f"{default:.2f}", font=("Helvetica", 8, "bold"), bootstyle="success")
        value_label.pack(side=RIGHT)
        scale = ttk.Scale(
            frame, from_=0, to=1, value=default, bootstyle="success",
            command=lambda v: value_label.config(text=f"{float(v):.2f}")
        )
        scale.pack(fill=X)
        return scale

    def load_and_draw_initial_graph(self):
        try:
            nodes_df = pd.read_csv(NODE_FILE, delimiter=";", decimal=",")
            edges_df = pd.read_csv(EDGE_FILE, delimiter=";", decimal=",")
            
            self.G = nx.Graph()
            for _, row in nodes_df.iterrows():
                self.G.add_node(int(row['node_id']), s_ms=float(row['s_ms']), r_node=float(row['r_node']))
            for _, row in edges_df.iterrows():
                self.G.add_edge(int(row['src']), int(row['dst']),
                                capacity_mbps=float(row['capacity_mbps']),
                                delay_ms=float(row['delay_ms']),
                                r_link=float(row['r_link']))
            
            self.pos = nx.spring_layout(self.G, seed=42, k=0.15)
            node_list = sorted([str(n) for n in self.G.nodes()])
            self.cb_source['values'] = node_list
            self.cb_target['values'] = node_list
            self.cb_source.set("8")
            self.cb_target.set("44")
            self.draw_graph(path=None)
        except Exception as e:
            messagebox.showerror("Veri Hatası", f"Dosyalar okunurken hata oluştu:\n{e}")

    def draw_graph(self, path=None):
        self.ax.clear()
        self.ax.axis('off')
        nx.draw_networkx_nodes(self.G, self.pos, ax=self.ax, node_size=30, node_color='#2a3e52', alpha=0.4)
        nx.draw_networkx_edges(self.G, self.pos, ax=self.ax, edge_color='#1a252f', width=0.5, alpha=0.3)
        
        if path:
            path_edges = list(zip(path, path[1:]))
            nx.draw_networkx_edges(self.G, self.pos, ax=self.ax, edgelist=path_edges, edge_color='#00f2ff', width=6, alpha=0.4)
            nx.draw_networkx_edges(self.G, self.pos, ax=self.ax, edgelist=path_edges, edge_color='#00f2ff', width=2)
            nx.draw_networkx_nodes(self.G, self.pos, ax=self.ax, nodelist=path, node_size=80, node_color='#feca57')
            nx.draw_networkx_nodes(self.G, self.pos, ax=self.ax, nodelist=[path[0]], node_size=150, node_color='#1dd1a1', label="Start")
            nx.draw_networkx_nodes(self.G, self.pos, ax=self.ax, nodelist=[path[-1]], node_size=150, node_color='#ff6b6b', label="End")
            self.ax.set_title(f"✔ Rota Bulundu: {len(path)} Adım", color="white", fontsize=14, fontweight='bold')
        else:
            self.ax.set_title(f"Ağ Topolojisi: {len(self.G.nodes)} Düğüm", color="white", fontsize=10)
        self.canvas.draw()

    def run_optimization(self):
        try:
            src = int(self.cb_source.get())
            dst = int(self.cb_target.get())
            
            demands_df = pd.read_csv("demanddata.csv", sep=';')
            demands_df.columns = demands_df.columns.str.strip()
            match = demands_df[(demands_df['src'] == src) & (demands_df['dst'] == dst)]
            
            if not match.empty:
                demand = float(match.iloc[0]['demand_mbps'])
                self.lbl_cost.config(text=f"✔ Talep Dosyadan: {demand} Mbps", bootstyle="success")
            else:
                demand = 100.0 
                self.lbl_cost.config(text=f"ℹ Varsayılan Talep (100 Mbps)", bootstyle="info")

            w_d = self.scale_delay.get()
            w_r = self.scale_rel.get()
            w_res = self.scale_res.get()
            total = (w_d + w_r + w_res) or 1
            weights = {'delay': w_d/total, 'reliability': w_r/total, 'resource': w_res/total}
            
            selected_algo = self.algo_var.get()
            
            if selected_algo == "GA":
                found_path = self.run_genetic_algorithm(src, dst, demand, weights)
            else:
                found_path = self.run_qlearning(src, dst, demand, weights)

            if found_path:
                self.display_path_results(found_path) 
                self.draw_graph(found_path)
                self.lbl_cost.config(text="✔ Optimizasyon Tamamlandı", bootstyle="success")
            else:
                self.lbl_cost.config(text="❌ Yol Bulunamadı!", bootstyle="danger")

        except Exception as e:
            messagebox.showerror("Kritik Hata", f"Hata: {str(e)}")

    def display_path_results(self, found_path):
        import math
        import metrics
        
        c_delay = metrics.total_delay(self.G, found_path)
        c_rel_cost = metrics.reliability_cost(self.G, found_path)
        c_res = metrics.resource_cost(self.G, found_path)
        yuzde_guven = math.exp(-c_rel_cost) * 100
        
        self.lbl_path.config(text=f"Rota: {found_path}")
        self.lbl_gecikme.config(text=f"Toplam Gecikme: {c_delay:.2f} ms")
        self.lbl_guvenilirlik.config(text=f"Güvenilirlik Oranı: %{yuzde_guven:.2f}")
        self.lbl_kaynak.config(text=f"Ağ Kaynak Kullanımı: {c_res:.2f}")

    def run_genetic_algorithm(self, src, dst, demand, weights):
        try:
            net_obj = genetic_algo.Network()
            net_obj.load_nodes(NODE_FILE)
            net_obj.load_edges(EDGE_FILE)
            
            ga = genetic_algo.GeneticAlgorithm(
                network=net_obj, source=src, dest=dst, demand=demand,
                pop_size=30, generations=15, mutation_rate=0.1,
                w_delay=weights['delay'], w_rel=weights['reliability'], w_res=weights['resource']
            )
            best_ind = ga.run()
            if best_ind and hasattr(best_ind, 'path'):
                return best_ind.path
            return best_ind 
        except Exception:
            return nx.shortest_path(self.G, src, dst, weight='delay')

    def run_qlearning(self, src, dst, demand, weights):
        import qlearning_algo
        import pandas as pd
        
        # Kullanıcıyı bilgilendir ve ekranı güncelle
        self.lbl_cost.config(text="🔄 RL Hesaplanıyor...", bootstyle="warning")
        self.root.update()

        try:
            # 1. Hafızayı Temizle
            qlearning_algo.Q.clear()

            # 2. Ağırlıkları Ayarla ve Normalize Et
            qlearning_algo.w_d = weights['delay']
            qlearning_algo.w_r = weights['reliability']
            qlearning_algo.w_u = weights['resource']
            
            total = qlearning_algo.w_d + qlearning_algo.w_r + qlearning_algo.w_u
            if total > 0:
                qlearning_algo.w_d /= total
                qlearning_algo.w_r /= total
                qlearning_algo.w_u /= total

            # 3. Tekil Talep
            custom_demand = pd.DataFrame([{'src': src, 'dst': dst, 'demand_mbps': demand}])

            # 4. Hızlı Eğitim
            qlearning_algo.train_q_learning(self.G, custom_demand)

            # 5. Sonucu Çıkar
            policy = qlearning_algo.extract_policy(qlearning_algo.Q)
            found_path = qlearning_algo.get_best_path(policy, src, dst)
            
            return found_path
            
        except Exception as e:
            print(f"Hata: {e}")
            return []

if __name__ == "__main__":
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except:
        pass
    app = ttk.Window(themename="cyborg")
    NetworkProjectGUI(app)
    app.mainloop()
