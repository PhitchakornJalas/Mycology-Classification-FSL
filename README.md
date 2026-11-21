# Mycology-Classification-FSL

## 🎯 ปัญหาและความสำคัญ
...

## 📦 ชุดข้อมูลที่ใช้

ชุดข้อมูลภาพเชื้อราบนจานเพาะเชื้อจาก **OpenFungi** ([Zenodo Record: 15692070](https://zenodo.org/records/15692070)) ประกอบด้วยเชื้อราทั้งหมด 6 คลาส รวมจำนวน **507 รูป** โดยมีรายละเอียดสัดส่วนข้อมูลในแต่ละคลาสดังนี้:

| Index | ชื่อคลาส (Class Name) | จำนวนรูป (Images) |
| :---: | :--- | :---: |
| `0` | *Alternaria spp.* (`Alternaria_spp`) | 109 |
| `1` | *Aspergillus* section *Flavi* (`Aspergillus_section_Flavi`) | 222 |
| `2` | *Aspergillus* section *Nigri* (`Aspergillus_section_Nigri`) | 131 | 
| `3` | *Fusarium spp.* (`Fusarium_spp`) | 6 | 
| `4` | *Penicillium spp.* (`Penicillium_spp`) | 22 | 
| `5` | *Rhizopus spp.* (`Rhizopus_spp`) | 17 |
| **รวม** | **ทั้งหมด 6 คลาส** | **507** |

## 📁 โครงสร้างโฟลเดอร์
```text
fungal-plate-fewshot-classifier/
├── src/                        # โค้ดหลักของระบบ
│   ├── build_registry.py       # รวบรวมไฟล์ภาพเชื้อราจาก data/openfungi สร้าง registry.csv และไฟล์แมป Label
│   ├── split_data.py           # แบ่งชุดข้อมูลเป็น train / val / test แบบ Stratified
│   ├── build_pool.py           # สร้างไฟล์ pools_registry.json สำหรับการสุ่มตัวอย่างแบบ Episodic
│   ├── transforms.py           # กำหนดค่า Preprocessing (Resize, Normalize) และ Data Augmentation
│   ├── dataset.py              # คลาส EpisodicDataset สำหรับสุ่ม N-way K-shot Episode
│   ├── plot_episode.py         # แสดงภาพตัวอย่างของ Episode เพื่อตรวจสอบความถูกต้องของข้อมูลและ Augmentation
│   ├── _embedder.py            # ส่วนสกัด Feature Vector จาก Pre-trained Vision Backbone
│   ├── _projection.py          # โครงสร้าง Projection Head สำหรับแปลง Feature สู่ Embedding Space
│   ├── _similarity.py          # ฟังก์ชันคำนวณ Cosine Similarity และ Class Prototypes
│   ├── _losses.py              # ฟังก์ชันคำนวณ Prototypical Contrastive Loss
│   ├── models.py               # สถาปัตยกรรมหลัก FewShotContrastivePipeline
│   ├── train.py                # สคริปต์ฝึกสอนโมเดลแบบ Episodic Training พร้อมบันทึก Checkpoint
│   ├── plot_history.py         # สร้างกราฟประวัติการเรียนรู้ (Loss และ Validation Accuracy)
│   ├── evaluate.py             # ประเมินผลแบบ N-way K-shot Leave-One-Out คำนวณ Metrics รายคลาสและพล็อต CM
│   ├── export_prototypes.py    # ส่งออกน้ำหนักโมเดลและเวกเตอร์ Prototype ประจำคลาสเป็นไฟล์ deployed_model_*.pt
│   └── app.py                  # Web Application (Gradio) รองรับการอัปโหลดรูปและถ่ายภาพสด พร้อมแสดง Top-3
│
├── data/                       # ชุดข้อมูลภาพจานเพาะเชื้อ
│   ├── openfungi/              # โฟลเดอร์เก็บภาพต้นฉบับแยกตามสปีชีส์เชื้อรา
│   ├── registry.csv            # ดัชนีรวมตำแหน่งไฟล์ภาพและป้ายกำกับทั้งหมด
│   ├── train_registry.csv      # ชุดข้อมูลฝึกสอน
│   ├── val_registry.csv        # ชุดข้อมูลตรวจสอบระหว่างเทรน
│   ├── test_registry.csv       # ชุดข้อมูลทดสอบ
│   ├── pools_registry.json     # โครงสร้าง Pool แยกตาม Split สำหรับ Episodic Sampling
│   ├── index2label_registry.json # ไฟล์แมปรหัสคลาส 
│   └── label2index_registry.json # ไฟล์แมปชื่อสปีชีส์เชื้อราเป็นรหัสคลาส
│
├── checkpoints/                # เก็บไฟล์น้ำหนักโมเดลที่ดีที่สุด และไฟล์ที่ Export 
├── logs/                       # เก็บประวัติการเทรนและผลลัพธ์การประเมิน
├── plots/                      # เก็บกราฟประวัติการเรียนรู้และกราฟ Confusion Matrix 
└── requirements.txt            # รายการ Library ทั้งหมดของโปรเจกต์
```

## 🚀 ลำดับขั้นตอนการรัน

เปิด Terminal แล้ว `cd` เข้ามาที่โฟลเดอร์หลักของโปรเจกต์ จากนั้นดำเนินการตามลำดับดังนี้:

### ขั้นที่ 1 · ติดตั้งสภาพแวดล้อม
สร้าง Virtual Environment และติดตั้ง Library ทั้งหมดตามไฟล์ `requirements.txt`

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

แล้ว `cd` เข้ามาที่โฟลเดอร์ `src` ของโปรเจกต์ จากนั้นดำเนินการตามลำดับดังนี้:

### ขั้นที่ 2 · เตรียมข้อมูลและสร้าง Episodic Pool
รันสคริปต์เรียงตามลำดับเพื่อสร้างดัชนีไฟล์ภาพ แบ่งชุดข้อมูล สร้าง Pool สำหรับสุ่ม Episode และตรวจสอบภาพตัวอย่าง

```bash
# 1. สร้างไฟล์ดัชนี registry.csv และไฟล์แมป index2label / label2index
python build_registry.py --export

# 2. แบ่งข้อมูลเป็นชุด train / val / test
python split_data.py --export

# 3. สร้างไฟล์ pools_registry.json สำหรับ Episodic Dataset
python build_pool.py --export

# 4. ตรวจสอบภาพตัวอย่างในแต่ละ Episode
python plot_episode.py --export
```

### ขั้นที่ 3 · ตรวจสอบ Pipeline และฝึกสอนโมเดล
ทดสอบการทำงานของ DataLoader และโครงสร้างโมเดล จากนั้นเริ่มฝึกสอนแบบ Episodic Contrastive Learning

```bash
# 1. ทดสอบการโหลดข้อมูลและโครงสร้าง Tensor ของโมเดล
python models.py

# 2. เทรนโมเดลแบบ 3-way 2-shot
python train.py --n_way 3 --k_shot 2 --epochs 15 --train_episodes 60 --val_episodes 30 --patience 4
```

### ขั้นที่ 4 · พล็อตกราฟและประเมินผลบน Test Set
สร้างกราฟประวัติการเรียนรู้และประเมินประสิทธิภาพเชิงลึกด้วยวิธี Leave-One-Out Episodic Evaluation

```bash
# 1. สร้างกราฟ Loss และ Validation Accuracy
python plot_history.py --export

# 2. ประเมินผลลัพธ์บน Test Set คำนวณ Diagnostic Metrics รายคลาส และส่งออกกราฟ Confusion Matrix
python evaluate.py --n_way 3 --k_shot 2 --test_episodes 100 --export
```

### ขั้นที่ 5 · ส่งออกโมเดลและเปิดใช้งาน Web Demo
สร้างเวกเตอร์ Prototype ประจำแต่ละสปีชีส์เชื้อราและเปิดหน้าเว็บสำหรับทดสอบอัปโหลดหรือถ่ายภาพจานเพาะเชื้อ

```bash
# 1. ส่งออกไฟล์น้ำหนักโมเดลและ Prototypes
python export_prototypes.py 

# 2. เปิดใช้งาน Web Application
python app.py
```