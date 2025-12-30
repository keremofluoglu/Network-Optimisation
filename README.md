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

Q-Learning (Pekiştirmeli Öğrenme)

Bu projede Q-Learning, ağ yönlendirme probleminde en uygun geçiş politikasını öğrenmek amacıyla kullanılmıştır.

Ağdaki her düğüm bir durum (state) olarak modellenmiştir.

Bir düğümden komşu bir düğüme geçiş, bir aksiyon (action) olarak tanımlanmıştır.

Q tablosu, yalnızca komşu düğümler arasındaki geçişlerin uzun vadeli değerlerini tutar.

Q-Learning, deneme–yanılma yoluyla her düğümde hangi komşuya gitmenin uzun vadede daha avantajlı olduğunu öğrenir.
Bu süreçte yalnızca anlık maliyetler değil, gelecekte elde edilecek ödüller de dikkate alınır.

Genetik Algoritma, çok kriterli maliyet fonksiyonunda kullanılan ağırlıkların optimize edilmesi amacıyla kullanılmıştır.

Ağ yönlendirme problemi; gecikme, bant genişliği kullanımı ve güvenilirlik gibi
birden fazla ve birbiriyle çelişebilen metriği içermektedir.
Bu nedenle sabit ağırlıklar yerine, en uygun ağırlık kombinasyonlarının
evrimsel olarak bulunması hedeflenmiştir.

Her birey, maliyet fonksiyonundaki ağırlıkları temsil eder.

Uygunluk (fitness) değeri, elde edilen yolların toplam performansına göre hesaplanır.

Seçilim, çaprazlama ve mutasyon işlemleriyle daha iyi çözümler üretilir.

Genetik Algoritma sonucunda elde edilen en iyi ağırlıklar,
Q-Learning ödül fonksiyonunda kullanılarak öğrenme süreci yönlendirilir.
