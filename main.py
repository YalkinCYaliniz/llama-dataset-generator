"""
Llama Fine-Tuning Dataset Generator
====================================
User Story + Acceptance Criteria → 5 Test Case üretimi için
~10.000 örneklik çeşitli bir training dataset oluşturur.

Kullanım:
    python main.py                    # Tüm 10.000 veriyi üret
    python main.py --count 50         # Sadece 50 örnek üret (test için)
    python main.py --resume           # Kaldığı yerden devam et
    python main.py --workers 15       # Paralel worker (varsayılan 15)
    python main.py --lang tr          # Sadece Türkçe veri üret (tr/en/mixed)
"""

import os
import json
import time
import random
import argparse
import asyncio
from pathlib import Path
from datetime import datetime

from openai import AsyncOpenAI
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# KONU HAVUZU (topics.json — domain → özellik listesi)
# ─────────────────────────────────────────────

_TOPICS_PATH = Path(__file__).resolve().parent / "topics.json"
with open(_TOPICS_PATH, encoding="utf-8") as _f:
    TOPICS: dict[str, list[str]] = json.load(_f)
# ─────────────────────────────────────────────
# SYSTEM PROMPT ve GENERATION FUNCTIONS
# ─────────────────────────────────────────────

SYSTEM_PROMPT_EN = """You are an expert QA Engineer and test case writer. You will be given a User Story and Acceptance Criteria.
Your task is to generate exactly 5 comprehensive test cases.

IMPORTANT RULES:
- Generate exactly 5 test cases (mix of positive, negative, and edge cases)
- Each test case MUST follow the exact format below
- Be specific and detailed in steps
- Include realistic test data
- Cover different scenarios (happy path, error handling, boundary values, etc.)

FORMAT for each test case:
**Test Case [number]: [Title]**
- **Priority:** [High/Medium/Low]
- **Preconditions:** [What must be true before the test]
- **Steps:**
  1. [Step 1]
  2. [Step 2]
  ...
- **Expected Result:** [What should happen]"""

SYSTEM_PROMPT_TR = """Sen uzman bir QA Mühendisi ve test case yazarısın. Sana bir User Story ve Acceptance Criteria verilecek.
Görevin tam olarak 5 adet kapsamlı test case oluşturmaktır.

ÖNEMLİ KURALLAR:
- Tam olarak 5 test case üret (pozitif, negatif ve edge case karışımı)
- Her test case MUTLAKA aşağıdaki formatı takip etmeli
- Adımlarda spesifik ve detaylı ol
- Gerçekçi test verileri kullan
- Farklı senaryoları kapsa (happy path, hata yönetimi, sınır değerler, vb.)

Her test case için FORMAT:
**Test Case [numara]: [Başlık]**
- **Öncelik:** [Yüksek/Orta/Düşük]
- **Ön Koşullar:** [Test öncesi sağlanması gereken koşullar]
- **Adımlar:**
  1. [Adım 1]
  2. [Adım 2]
  ...
- **Beklenen Sonuç:** [Ne olması gerektiği]"""

USER_STORY_PROMPT_EN = """Generate a realistic and detailed User Story with Acceptance Criteria for the following context:
- Domain/Industry: {domain}
- Feature/Topic: {feature}

Requirements:
1. Write a clear User Story in "As a [role], I want [feature], so that [benefit]" format
2. Include 3-5 specific and testable Acceptance Criteria
3. Make it realistic and detailed enough to write test cases from
4. Include edge cases in acceptance criteria

Respond ONLY with the User Story and Acceptance Criteria, nothing else.

Format:
**User Story:** As a [role], I want [feature], so that [benefit].

**Acceptance Criteria:**
1. [Criterion 1]
2. [Criterion 2]
3. [Criterion 3]
..."""

USER_STORY_PROMPT_TR = """Aşağıdaki bağlam için gerçekçi ve detaylı bir User Story ile Acceptance Criteria üret:
- Alan/Sektör: {domain}
- Özellik/Konu: {feature}

Gereksinimler:
1. "Bir [rol] olarak, [özellik] istiyorum, böylece [fayda]" formatında açık bir User Story yaz
2. 3-5 adet spesifik ve test edilebilir Acceptance Criteria ekle
3. Test case yazılabilecek kadar gerçekçi ve detaylı olsun
4. Acceptance criteria'da edge case'leri de dahil et

SADECE User Story ve Acceptance Criteria ile yanıt ver, başka bir şey yazma.

Format:
**User Story:** Bir [rol] olarak, [özellik] istiyorum, böylece [fayda].

**Acceptance Criteria:**
1. [Kriter 1]
2. [Kriter 2]
3. [Kriter 3]
..."""

INSTRUCTION_EN = "Based on the following User Story and Acceptance Criteria, write 5 test cases."
INSTRUCTION_TR = "Aşağıdaki User Story ve Acceptance Criteria'ya göre 5 adet test case yaz."


def get_prompts(lang: str):
    """Dil tercihine göre prompt'ları döndür."""
    if lang == "tr":
        return SYSTEM_PROMPT_TR, USER_STORY_PROMPT_TR, INSTRUCTION_TR
    elif lang == "en":
        return SYSTEM_PROMPT_EN, USER_STORY_PROMPT_EN, INSTRUCTION_EN
    else:  # mixed
        return None, None, None


# ─────────────────────────────────────────────
# CHECKPOINT SİSTEMİ
# ─────────────────────────────────────────────

CHECKPOINT_FILE = "checkpoint.json"
OUTPUT_FILE = "dataset.jsonl"


def load_checkpoint() -> dict:
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"completed": 0, "failed": 0, "last_update": None}


def save_checkpoint(data: dict):
    data["last_update"] = datetime.now().isoformat()
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def append_to_dataset(entry: dict):
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def count_existing_entries() -> int:
    if not os.path.exists(OUTPUT_FILE):
        return 0
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


# ─────────────────────────────────────────────
# WORK ITEM GENERATION
# ─────────────────────────────────────────────

def generate_work_items(total_count: int) -> list[dict]:
    items = []
    all_combinations = []

    for domain, features in TOPICS.items():
        for feature in features:
            all_combinations.append({"domain": domain, "feature": feature})

    random.shuffle(all_combinations)
    total_combinations = len(all_combinations)

    repeats = total_count // total_combinations
    remainder = total_count % total_combinations

    for combo in all_combinations:
        for _ in range(repeats):
            items.append(combo.copy())

    random.shuffle(all_combinations)
    for i in range(remainder):
        items.append(all_combinations[i].copy())

    random.shuffle(items)
    return items[:total_count]


# ─────────────────────────────────────────────
# ASYNC VERİ ÜRETİMİ (OPENAI)
# ─────────────────────────────────────────────

async def generate_single_example(
    client: AsyncOpenAI,
    domain: str,
    feature: str,
    lang: str,
    semaphore: asyncio.Semaphore,
    retry_count: int = 4
) -> dict | None:
    """Tek bir örnek üret: User Story → Test Cases."""

    # Dil seçimi
    if lang == "mixed":
        chosen_lang = random.choice(["en", "tr"])
    else:
        chosen_lang = lang

    sys_prompt, us_prompt_template, instruction = get_prompts(chosen_lang)
    if sys_prompt is None:
        chosen_lang = random.choice(["en", "tr"])
        sys_prompt, us_prompt_template, instruction = get_prompts(chosen_lang)

    for attempt in range(retry_count):
        try:
            async with semaphore:
                # ADIM 1: User Story + Acceptance Criteria üret
                us_response = await client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a senior product manager who writes detailed user stories and acceptance criteria."},
                        {"role": "user", "content": us_prompt_template.format(domain=domain, feature=feature)}
                    ],
                    temperature=random.uniform(0.7, 1.0),
                    max_tokens=600
                )
                user_story_text = us_response.choices[0].message.content.strip()

                # Küçük bir bekleme - rate limit'e takılmamak için
                await asyncio.sleep(random.uniform(0.1, 0.4))

                # ADIM 2: User Story'den Test Case'ler üret
                tc_response = await client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_story_text}
                    ],
                    temperature=random.uniform(0.6, 0.9),
                    max_tokens=1500
                )
                test_cases_text = tc_response.choices[0].message.content.strip()

                # JSONL formatında döndür
                return {
                    "instruction": instruction,
                    "input": user_story_text,
                    "output": test_cases_text
                }

        except Exception as e:
            if attempt < retry_count - 1:
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                await asyncio.sleep(wait_time)
            else:
                print(f"\n❌ Hata ({domain}/{feature}): {e}")
                return None

    return None


async def generate_dataset(
    total_count: int,
    max_workers: int,
    lang: str,
    resume: bool
):
    """Ana dataset üretim fonksiyonu."""

    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # OpenAI rate limits generally allow much higher concurrency than free Gemini
    semaphore = asyncio.Semaphore(max_workers)

    checkpoint = load_checkpoint()
    start_from = 0

    if resume and checkpoint["completed"] > 0:
        start_from = checkpoint["completed"]
        print(f" Checkpoint bulundu: {start_from} örnek tamamlanmış, kaldığı yerden devam ediliyor...")
    else:
        # Yeni başlangıç
        if os.path.exists(OUTPUT_FILE):
            os.remove(OUTPUT_FILE)
        save_checkpoint({"completed": 0, "failed": 0, "last_update": None})

    remaining = total_count - start_from
    if remaining <= 0:
        print(f" Zaten {total_count} örnek tamamlanmış!")
        return

    work_items = generate_work_items(remaining)

    # Maliyet hesabi
    cost_per_example = 0.00039  # $0.15/1M in, $0.60/1M out
    estimated_cost = remaining * cost_per_example

    print(f"""
╔══════════════════════════════════════════════════╗
║   Llama Fine-Tuning Dataset Generator        ║
╠══════════════════════════════════════════════════╣
║  Toplam hedef  : {total_count:>7,} örnek               ║
║  Kalan         : {remaining:>7,} örnek               ║
║  Paralel worker: {max_workers:>7}                    ║
║  Dil           : {lang:>7}                    ║
║  API           : OpenAI (gpt-4o-mini)            ║
║  Çıktı dosyası : {OUTPUT_FILE:<20}           ║
║  Tahmini Maliyet: ~${estimated_cost:<6.2f}                   ║
╚══════════════════════════════════════════════════╝
    """)

    completed = start_from
    failed = 0
    pbar = tqdm(total=remaining, desc="Dataset üretiliyor", unit="örnek",
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

    try:
        # Batch halinde işle
        batch_size = max_workers * 2
        for batch_start in range(0, remaining, batch_size):
            batch_end = min(batch_start + batch_size, remaining)
            batch = work_items[batch_start:batch_end]

            tasks = [
                generate_single_example(client, item["domain"], item["feature"], lang, semaphore)
                for item in batch
            ]

            results = await asyncio.gather(*tasks)

            for result in results:
                if result is not None:
                    append_to_dataset(result)
                    completed += 1
                else:
                    failed += 1

                pbar.update(1)

            # Her batch sonunda checkpoint kaydet
            save_checkpoint({
                "completed": completed,
                "failed": failed,
                "total_target": total_count,
            })
            
            # API'yi bogmamak icin ufak bekleme
            await asyncio.sleep(0.5)

    except asyncio.CancelledError:
        pass
    except Exception as e:
        print(f"\nBeklenmeyen hata: {e}")
    finally:
        pbar.close()

    # Sonuç raporu
    actual_count = count_existing_entries()
    print(f"""
╔══════════════════════════════════════════════════╗
║    Dataset Üretimi Durduruldu/Tamamlandı!      ║
╠══════════════════════════════════════════════════╣
║  Başarılı      : {completed:>7,} örnek               ║
║  Başarısız     : {failed:>7,} örnek               ║
║  Dosyadaki     : {actual_count:>7,} örnek               ║
║  Dosya         : {OUTPUT_FILE:<20}           ║
╚══════════════════════════════════════════════════╝
    """)

    validate_dataset()


# ─────────────────────────────────────────────
# DOĞRULAMA
# ─────────────────────────────────────────────

def validate_dataset():
    print("\n Dataset doğrulanıyor...")

    if not os.path.exists(OUTPUT_FILE):
        print(" Dataset dosyası bulunamadı!")
        return

    valid = 0
    invalid = 0
    errors = []

    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            try:
                entry = json.loads(line)
                if not all(k in entry for k in ["instruction", "input", "output"]):
                    invalid += 1
                    errors.append(f"Satır {i}: Eksik alan")
                elif not entry["input"].strip() or not entry["output"].strip():
                    invalid += 1
                    errors.append(f"Satır {i}: Boş input veya output")
                else:
                    valid += 1
            except json.JSONDecodeError:
                invalid += 1
                errors.append(f"Satır {i}: Geçersiz JSON")

    print(f"   Geçerli: {valid:,}")
    print(f"   Geçersiz: {invalid}")

    if errors:
        print("\n  İlk 5 hata:")
        for err in errors[:5]:
            print(f"    - {err}")

    size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
    print(f"\n  📁 Dosya boyutu: {size_mb:.2f} MB")

    if valid > 0:
        print("\n   Örnek entry:")
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            first_line = f.readline()
            sample = json.loads(first_line)
            print(f"    instruction: {sample['instruction'][:80]}...")
            print(f"    input: {sample['input'][:120]}...")
            print(f"    output: {sample['output'][:120]}...")


# ─────────────────────────────────────────────
# ANA PROGRAM
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Llama Fine-Tuning Dataset Generator - User Story → Test Cases"
    )
    parser.add_argument(
        "--count", type=int, default=10000,
        help="Üretilecek toplam örnek sayısı (varsayılan: 10000)"
    )
    parser.add_argument(
        "--workers", type=int, default=15,
        help="Paralel worker sayısı (varsayılan: 15, OpenAI için genelde daha yüksek olabilir)"
    )
    parser.add_argument(
        "--lang", type=str, default="en", choices=["tr", "en", "mixed"],
        help="Dil tercihi: tr (Türkçe), en (İngilizce), mixed (karışık)"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Kaldığı yerden devam et"
    )
    parser.add_argument(
        "--validate-only", action="store_true",
        help="Sadece mevcut dataset'i doğrula"
    )

    args = parser.parse_args()

    if args.validate_only:
        validate_dataset()
        return

    if not os.getenv("OPENAI_API_KEY"):
        print(" OPENAI_API_KEY bulunamadı!")
        print("   Lütfen .env dosyasına ekleyin veya environment variable olarak ayarlayın.")
        return

    try:
        asyncio.run(generate_dataset(
            total_count=args.count,
            max_workers=args.workers,
            lang=args.lang,
            resume=args.resume
        ))
    except KeyboardInterrupt:
        print("\n\n İşlem sonlandırıldı. Tekrar başlatmak için --resume kullanın.")


if __name__ == "__main__":
    main()