import site
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from tkinter import messagebox

# grafik ve görselleştirme kütüphaneleri
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# ağ ve veri işleme kütüphaneleri
import networkx as nx
import pandas as pd

#  PROJE MODÜLLERİNİ YÜKLE
#  GA, Q Learning ve metrik hesaplama modülleri
try:
    import genetic_algo     # genetik Algoritma modülü
    import qlearning_algo   # Q learning modülü
    import metrics          # delay, reliability, resource hesapları
    print("Modüller başarıyla yüklendi.")
except ImportError as e:
    print(f"UYARI: Modüller tam yüklenemedi ({e}).")

# veri dosyaları
NODE_FILE = "NodeData.csv"
EDGE_FILE = "EdgeData.csv"

# =========================================================
#  ANA GUI SINIFI
# =========================================================
class NetworkProjectGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("BSM307 - Akıllı Ağ Optimizasyonu v2.0")
        self.root.geometry("1350x900")

        # Ağ grafiği ve node pozisyonları
        self.G = None
        self.pos = None

        # Arayüzü kur ve grafiği çiz
        self.setup_ui()
        self.load_and_draw_initial_graph()
        
    #  ARAYÜZ KURULUMU
    def setup_ui(self):
        # Ana çerçeve
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=BOTH, expand=YES)

        # SOL KONTROL PANELİ
        control_panel = ttk.Frame(main_frame)
        control_panel.pack(side=LEFT, fill=Y, padx=(0, 15))
        
        # başlık
        header_frame = ttk.Frame(control_panel)
        header_frame.pack(fill=X, pady=(0, 10))
        ttk.Label(
            header_frame,
            text="QoS ROTALAMA",
            font=("Helvetica", 22, "bold"),
            bootstyle="primary"
        ).pack(anchor=W)
        
        # GÜZERGAH SEÇİMİ
        pnl_route = ttk.Labelframe(
            control_panel,
            text="Güzergah Ayarları",
            padding=15,
            bootstyle="info"
        )
        pnl_route.pack(fill=X, pady=10)
        
        ttk.Label(pnl_route, text="Kaynak (Source):", bootstyle="inverse-info").pack(anchor=W)
        self.cb_source = ttk.Combobox(pnl_route, state="readonly", bootstyle="info")
        self.cb_source.pack(fill=X, pady=(2, 10))
        
        ttk.Label(pnl_route, text="Hedef (Destination):", bootstyle="inverse-info").pack(anchor=W)
        self.cb_target = ttk.Combobox(pnl_route, state="readonly", bootstyle="info")
        self.cb_target.pack(fill=X, pady=(2, 10))
        
        # ALGORİTMA SEÇİMİ
        pnl_algo = ttk.Labelframe(
            control_panel,
            text="Algoritma Motoru",
            padding=15,
            bootstyle="warning"
        )
        pnl_algo.pack(fill=X, pady=10)
        
        self.algo_var = ttk.StringVar(value="GA")
        ttk.Radiobutton(
            pnl_algo,
            text="Genetik Algoritma (GA)",
            variable=self.algo_var,
            value="GA",
            bootstyle="warning-toolbutton"
        ).pack(fill=X, pady=2)

        ttk.Radiobutton(
            pnl_algo,
            text="Pekiştirmeli Öğrenme (RL)",
            variable=self.algo_var,
            value="QL",
            bootstyle="warning-toolbutton"
        ).pack(fill=X, pady=2)
        
        # QoS AĞIRLIK AYARLARI
        pnl_weights = ttk.Labelframe(
            control_panel,
            text="QoS Öncelikleri",
            padding=15,
            bootstyle="success"
        )
        pnl_weights.pack(fill=X, pady=10)
        
        #kullanıcı tarafından ayarlanabilen ağırlıklar
        self.scale_delay = self.create_meter(pnl_weights, "Gecikme (Delay)", 0.4)
        self.scale_rel   = self.create_meter(pnl_weights, "Güvenilirlik (Rel.)", 0.3)
        self.scale_res   = self.create_meter(pnl_weights, "Kaynak (Resource)", 0.3)
        
        # SONUÇ PANELİ
        pnl_result = ttk.Labelframe(
            control_panel,
            text="Analiz Sonuçları",
            padding=15,
            bootstyle="danger"
        )
        pnl_result.pack(fill=X, pady=5)
        
        self.lbl_path = ttk.Label(
            pnl_result,
            text="Rota: -",
            font=("Consolas", 10, "bold"),
            wraplength=250
        )
        self.lbl_path.pack(anchor=W)

        self.lbl_gecikme = ttk.Label(pnl_result, text="Toplam Gecikme: -", bootstyle="warning")
        self.lbl_gecikme.pack(anchor=W, pady=2)

        self.lbl_guvenilirlik = ttk.Label(pnl_result, text="Güvenilirlik Oranı: -", bootstyle="success")
        self.lbl_guvenilirlik.pack(anchor=W, pady=2)

        self.lbl_kaynak = ttk.Label(pnl_result, text="Ağ Kaynak Kullanımı: -", bootstyle="primary")
        self.lbl_kaynak.pack(anchor=W, pady=2)

        self.lbl_cost = ttk.Label(
            pnl_result,
            text="Durum: Bekleniyor...",
            font=("Helvetica", 9, "bold"),
            bootstyle="danger"
        )
        self.lbl_cost.pack(anchor=W, pady=(5,0))

        # optimizasyon başlatma butonu
        self.btn_run = ttk.Button(
            control_panel,
            text="⚡ OPTİMİZASYONU BAŞLAT",
            command=self.run_optimization,
            bootstyle="danger",
            width=25
        )
        self.btn_run.pack(fill=X, pady=20)

        # SAĞ PANEL: GRAF GÖRSELİ
        graph_panel = ttk.Frame(main_frame)
        graph_panel.pack(side=RIGHT, fill=BOTH, expand=YES)
        
        # matplotlib ayarları
        plt.style.use('dark_background')
        self.figure = plt.Figure(figsize=(8, 6), dpi=100, facecolor='#060606')
        self.ax = self.figure.add_subplot(111)
        self.ax.axis('off')
        
        # canvas
        self.canvas = FigureCanvasTkAgg(self.figure, graph_panel)
        self.canvas.get_tk_widget().pack(fill=BOTH, expand=YES)
        
        # toolbar
        toolbar = NavigationToolbar2Tk(self.canvas, graph_panel)
        toolbar.update()

    #  SLIDER (AĞIRLIK) OLUŞTURMA
    
    def create_meter(self, parent, label, default):

        frame = ttk.Frame(parent)
        frame.pack(fill=X, pady=5)

        header_frame = ttk.Frame(frame)
        header_frame.pack(fill=X)

        ttk.Label(header_frame, text=label, font=("Helvetica", 8)).pack(side=LEFT)
        value_label = ttk.Label(
            header_frame,
            text=f"{default:.2f}",
            font=("Helvetica", 8, "bold"),
            bootstyle="success"
        )
        value_label.pack(side=RIGHT)

        scale = ttk.Scale(
            frame,
            from_=0,
            to=1,
            value=default,
            bootstyle="success",
            command=lambda v: value_label.config(text=f"{float(v):.2f}")
        )
        scale.pack(fill=X)
        return scale

