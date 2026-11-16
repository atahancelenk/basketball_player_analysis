# Basketbol Görüntülerinde Oyuncu Tespiti

Bu projede basketbol maç görüntülerindeki saha oyuncuları, önceden eğitilmiş YOLOv8n modeli uyarlanarak tespit edilir. Kaynak: [Roboflow Basketball Player Detection v18](https://universe.roboflow.com/roboflow-jvuqo/basketball-player-detection-3-ycjdo/dataset/18), CC BY 4.0. Kaynaktaki `player`, `player-in-possession`, `player-jump-shot`, `player-layup-dunk` ve `player-shot-block` etiketleri tek `player` sınıfında birleştirilir. Hakem, top, pota ve forma numarası hedef dışındadır.

## Yerel klasör düzeni

Bu bilgisayarda kod bu klasördedir; indirilen ham veri, hazırlanmış veri, model ağırlıkları, eğitim çıktıları ve proje yönergesi kardeş `../staj4` klasöründedir. `project_paths.py` bu düzeni otomatik tanır. Kod GitHub'a tek başına yüklenebilir. Başka bir bilgisayarda veri setini indirip bu README'nin bulunduğu klasöre açmak yeterlidir; kod bu düzeni de otomatik tanır. Büyük veri ve eğitim çıktıları `.gitignore` ile GitHub dışında tutulur.

## Kurulum ve çalıştırma

Python 3.10 veya üstü gerekir. VS Code Python eklentisi önerilir. CUDA ve ayrı bir YOLO programı zorunlu değildir. Yeni bir bilgisayarda proje klasöründe:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe prepare_dataset.py
.\.venv\Scripts\python.exe explore_data.py
.\.venv\Scripts\python.exe train.py
.\.venv\Scripts\python.exe evaluate.py
```

Bu bilgisayarda paketler `../staj4/.venv` ortamında zaten kuruludur. VS Code'da `Python: Select Interpreter` ile o ortamdaki `Scripts/python.exe` seçilebilir. Yeni kuruluma gerek kalmadan terminalde şu biçim kullanılabilir:

```powershell
..\staj4\.venv\Scripts\python.exe train.py
..\staj4\.venv\Scripts\python.exe evaluate.py
```

`prepare_dataset.py` ham veriyi değiştirmez. `player_dataset` klasörünü oluşturur; dolu bir klasörün üzerine yazmaz. `explore_data.py` rapor için dört veri keşif görseli üretir. `train.py` varsayılan olarak 20 epoch, 320 piksel ve batch size 8 kullanır; eğitim çıktısı `outputs/train` altındadır. `evaluate.py` test bölümünü değerlendirir ve hata analizini çalıştırır. Bu bilgisayardaki `outputs` yolu `../staj4/outputs` olur.

## Bu çalışmanın sonuçları

Kaynakta 654 görüntü vardır: 464 eğitim, 96 doğrulama, 94 test. Tek sınıfa dönüştürüldükten sonra eğitimde 4556, doğrulamada 951, testte 944 oyuncu kutusu bulunur. 20 epoch sonunda ayrı test verisinde şu değerler elde edildi:

| Metrik | Değer |
| --- | ---: |
| Precision | 0.9388 |
| Recall | 0.8776 |
| F1 | 0.9071 |
| mAP@50 | 0.9494 |
| mAP@50:95 | 0.5943 |

`analyze_errors.py`, güven eşiği 0.25 ve eşleşme IoU eşiği 0.5 ile testte 851 doğru tespit, 83 yanlış tespit ve 93 kaçırılan oyuncu saydı. Bu sayılar sabit eşiklere aittir; yukarıdaki Ultralytics precision ve recall hesaplarıyla aynı işletim noktasını temsil etmez. Oyuncuların örtüşmesi, hakemlerin veya saha kenarındaki kişilerin oyuncu sanılması hata örneklerindendir. Eğitim, doğrulama ve test görüntüleri farklı kısa video kesitlerinden gelse de birkaç aynı maçtan geldiği için farklı maçlara genelleme ayrıca sınanmalıdır.

## Rapor çıktıları

- `outputs/eda/`: veri keşfi görselleri.
- `outputs/train/results.png`: eğitim ve doğrulama kaybı ile metrik eğrileri.
- `outputs/test_metrics.json` ve `outputs/test_evaluation/BoxPR_curve.png`: test ölçümleri ve precision-recall eğrisi.
- `outputs/test_errors/per_image.csv` ve `error_*.jpg`: görüntü bazında hata sayıları ve görsel örnekler. Yeşil doğru, turuncu yanlış tespit, kırmızı kaçırılan oyuncudur.

Nesne tespitinde accuracy yerine precision, recall, F1 ve mAP raporlanır. Kod dışındaki büyük veri, model ağırlıkları ve rapor dosyası GitHub deposuna eklenmez; raporda veri kaynağı ile bu deponun bağlantısı verilir.
