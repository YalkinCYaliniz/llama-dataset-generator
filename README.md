# Llama Fine-Tuning Dataset Generator

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

OpenAI **gpt-4o-mini** ile **instruction-tuning** için JSONL veri üretir: önce alan/özelliğe göre **User Story + Acceptance Criteria**, ardından buna bağlı **5 test case** (pozitif / negatif / edge).

Varsayılan hedef ~10.000 örnek; konu çeşitliliği `topics.json` içindeki **50 domain × 20 özellik = 1000** kombinasyon üzerinden doldurulur (tekrarlarla örnek sayısı artar).

## Gereksinimler

- Python 3.10+
- Geçerli [OpenAI API](https://platform.openai.com/) anahtarı

## Kurulum

```bash
git clone https://github.com/<kullanici>/llama-fine-tuning-dataset-generator.git
cd llama-fine-tuning-dataset-generator
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# .env içinde OPENAI_API_KEY değerini doldurun
```

## Kullanım

| Komut                            | Açıklama                                             |
| -------------------------------- | ---------------------------------------------------- |
| `python main.py`                 | Varsayılan **10.000** örnek üretir → `dataset.jsonl` |
| `python main.py --count 50`      | Hızlı deneme için 50 örnek                           |
| `python main.py --resume`        | Kesinti sonrası kaldığı yerden devam (checkpoint)    |
| `python main.py --workers 15`    | Eşzamanlı istek sayısı (rate limit’e göre ayarlayın) |
| `python main.py --lang en`       | Sadece İngilizce (`tr` / `mixed` de seçilebilir)     |
| `python main.py --validate-only` | Mevcut `dataset.jsonl` doğrulaması                   |

**Not:** `--resume` kullanmadığınızda yeni koşu `dataset.jsonl` dosyasını siler ve `checkpoint.json` içindeki ilerlemeyi sıfırlar.

## Çıktı formatı

Her satır bir JSON nesnesi (JSONL), tipik instruction tuning şeması:

```json
{
  "instruction": "Based on the following User Story and Acceptance Criteria, write 5 test cases.",
  "input": "**User Story:** ... **Acceptance Criteria:** ...",
  "output": "**Test Case 1:** ..."
}
```

Llama / diğer modeller için eğitim pipeline’ınıza göre alan adlarını map etmeniz yeterli.

## Konu havuzu (`topics.json`)

Domain → özellik string listesi. Kendi ürün alanınıza göre düzenleyebilir veya yeni domain ekleyebilirsiniz. `main.py` çalışma zamanında bu dosyayı yükler.

## Maliyet

Script kabaca örnek başına maliyet tahmini gösterir; gerçek ücret kullanım ve güncel OpenAI fiyatlarına bağlıdır. Üretim öncesi `--count` ile küçük deneme yapmanız önerilir.

## Lisans

Bu proje [MIT Lisansı](LICENSE) altında yayınlanır. Özetle: ticari veya kişisel kullanımda yazılımı özgürce kullanabilir, değiştirebilir ve dağıtabilirsiniz; tek koşul, telif ve izin metninin kopyalarını korumaktır. Garanti verilmez.
