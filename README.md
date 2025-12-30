Q-Learning ve Genetik Algoritma ile Ağ Yönlendirme Optimizasyonu

Bu proje, çok kriterli ağ yönlendirme (routing) problemini çözmek için
Q-Learning (Pekiştirmeli Öğrenme) ve Genetik Algoritma yaklaşımlarını kullanır.

Amaç; gecikme, bant genişliği kullanımı ve güvenilirlik gibi metrikleri dikkate alarak,
minimum bant genişliği gereksinimlerini kesin olarak sağlayan en uygun yolları bulmaktır.

Problem Tanımı
  Verilenler:
    Bir ağ grafiği (node ve edge’ler)
    
    Edge özellikleri:
      gecikme (delay)
      bant genişliği kapasitesi
      link güvenilirliği
      
    Node özellikleri:
      işlem gecikmesi
      node güvenilirliği

    Trafik talepleri (demand):
      kaynak (src)
      hedef (dst)
      minimum gerekli bant genişliği (demand_mbps)

Amaç:
  Minimum bant genişliği kısıtını (hard constraint) ihlal etmeden
  Toplam maliyeti minimize eden,
  Farklı trafik taleplerine uyum sağlayabilen,
  Etkin yönlendirme yolları bulmak.
