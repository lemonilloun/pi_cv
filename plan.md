# План реализации: Вариант B — семантическая карта комнаты
## Pi 5 (клиент, захват + сегментация) → M2 MacBook (сервер, глубина + позы + TSDF + граф)

> Документ рассчитан на передачу Claude Code агенту. Разбит на майлстоуны M0–M6, каждый с критерием приёмки. Pi-часть и Mac-часть развязаны через файловый «контракт данных» (раздел 2) — можно разрабатывать и тестировать независимо.

---

## 0. Архитектура и принцип работы

```
┌──────────────── Raspberry Pi 5 + Hailo-8 (26 TOPS) ────────────────┐
│ Camera Module 3 (фикс. фокус/экспозиция)                           │
│   └─> picamera2 → RGB 1536x864 (запись) + 640x640 (инференс)       │
│         └─> yolov5m_seg.hef (инстанс-сегментация, NMS в HEF)       │
│               └─> HailoTracker (track_id между кадрами)            │
│                     └─> Keyframe selector (~2 кадра/с)             │
│                           └─> Сессия на диск: rgb / masks / meta   │
└───────────────────────────── rsync по Wi-Fi ───────────────────────┘
┌──────────────────────── M2 MacBook (offline) ──────────────────────┐
│ 1. Позы камеры: COLMAP (sequential) — up-to-scale                  │
│ 2. Метрическая глубина: Depth Anything V2 metric (ViT-S, indoor)   │
│ 3. Выравнивание масштаба COLMAP-поз по метрической глубине         │
│ 4. TSDF-фьюжн: Open3D ScalableTSDFVolume → mesh комнаты            │
│ 5. Объекты: маски + глубина + позы → 3D-сегменты;                  │
│    ассоциация между кадрами: DINOv2-эмбеддинги + 3D-близость       │
│ 6. Семантика: CLIP (open_clip на Mac) → open-vocab метки           │
│ 7. Граф сцены: узлы-объекты + рёбра (near/on) → JSON + визуализация│
└────────────────────────────────────────────────────────────────────┘
```

Ключевые решения (и почему):
- **CLIP переносим на Mac (v1).** hailo-CLIP на Pi — рабочая, но лишняя сложность на старте: те же эмбеддинги считаются на Mac по кропам keyframe'ов через `open_clip` за миллисекунды. Pi в v1 делает только сегментацию + трекинг. hailo-CLIP на Pi — опциональный M7.
- **Позы — COLMAP, не ORB-SLAM3.** ORB-SLAM3 под macOS собирается мучительно (Pangolin, старый OpenCV). COLMAP ставится `brew install colmap`, offline-режим нас устраивает. MASt3R — запасной вариант (метрические позы сразу, но медленно на MPS).
- **Масштаб.** COLMAP даёт позы с точностью до масштаба; Depth Anything V2 metric даёт метры. Совмещаем: масштабный коэффициент = медиана отношений (COLMAP sparse depth / DAv2 depth) по всем кадрам. Это снимает главный риск монокулярного пайплайна.
- **Offline через rsync, не стриминг.** Реального времени не требуется — файловый обмен проще и надёжнее отлаживается. RTSP/ZeroMQ — потом.

---

## 1. Контракт данных (интерфейс Pi ↔ Mac)

Сессия записи = директория:

```
session_YYYYMMDD_HHMMSS/
├── intrinsics.json          # K-матрица, дисторсия, разрешение (из M0)
├── session_meta.json        # fps записи, модель, версии, длительность
└── keyframes/
    ├── 000001/
    │   ├── rgb.jpg          # полное разрешение 1536x864, quality=95
    │   ├── masks.png        # uint16 PNG, 0=фон, значение=instance_id кадра
    │   └── meta.json        # см. схему ниже
    ├── 000002/
    └── ...
```

`meta.json` на keyframe:
```json
{
  "frame_idx": 1,
  "timestamp_ns": 1234567890,
  "detections": [
    {
      "instance_id": 1,
      "track_id": 17,
      "class_coco": "chair",
      "confidence": 0.87,
      "bbox_xyxy": [120, 200, 340, 520]
    }
  ]
}
```

Правила:
- `masks.png` в разрешении rgb.jpg (маски ресайзятся с 640×640 nearest-neighbor).
- `track_id` — сквозной по сессии (от HailoTracker); `instance_id` — локальный для кадра, связывает пиксели masks.png с записью в detections.
- Всё, что нужно Mac-серверу — только эта директория. Никакой сетевой связности во время обработки.

---

## 2. M0 — Калибровка камеры (Pi, один раз)

**Зачем:** без точной K-матрицы TSDF и обратная проекция масок дадут «мыло». Camera Module 3 имеет автофокус — его ОБЯЗАТЕЛЬНО зафиксировать, иначе intrinsics плывут между кадрами.

Шаги:
1. Распечатать шахматную доску 9×6 (квадрат 25 мм), наклеить на жёсткую поверхность.
2. Скрипт `pi/calibrate.py`:
   - picamera2, разрешение записи (1536×864), **фиксация фокуса**: `picam2.set_controls({"AfMode": controls.AfModeEnum.Manual, "LensPosition": 2.0})` (LensPosition подобрать под рабочую дистанцию ~0.5–3 м, 1/м).
   - Захват 30–40 кадров доски под разными углами.
   - `cv2.findChessboardCorners` + `cv2.calibrateCamera` → K, dist.
   - Сохранить `intrinsics.json`: `{"width":1536,"height":864,"fx":...,"fy":...,"cx":...,"cy":...,"dist":[k1,k2,p1,p2,k3],"lens_position":2.0}`.
3. **Тот же LensPosition использовать во всех записях сессий.**

**Приёмка:** RMS reprojection error < 0.5 px; повторная калибровка даёт fx/fy в пределах ±1%.

---

## 3. M1 — Клиент записи на Pi 5

### 3.1 Установка
```bash
# Raspberry Pi OS Bookworm 64-bit, всё обновлено
sudo apt update && sudo apt full-upgrade -y
sudo apt install -y hailo-all          # драйвер, HailoRT, TAPPAS-core
sudo reboot
hailortcli fw-control identify          # должен увидеть HAILO8

git clone https://github.com/hailo-ai/hailo-apps.git
cd hailo-apps && ./install.sh           # скачает HEF-модели, поставит venv
source setup_env.sh
```

### 3.2 Модели
- Основная: **yolov5m_seg** HEF для hailo8 (кладётся hailo-apps'ом в resources; иначе — из Hailo Model Zoo, ветка hailo8). Выбор обоснован: NMS внутри HEF → минимум нагрузки на CPU Pi.
- Fallback/эксперимент: yolov8s_seg (выше mask mAP ~36.6, но NMS на CPU — следить за загрузкой ядер).

### 3.3 Скрипт `pi/recorder.py`
На базе instance segmentation pipeline из hailo-apps (GStreamer: source → hailonet → hailofilter → **hailotracker** → user_callback):
1. В user_callback читать из буфера `HAILO_DETECTION` + маски инстансов + `HAILO_UNIQUE_ID` (track_id от hailotracker).
2. Keyframe selector: принимать кадр, если (а) прошло ≥ 0.5 с от предыдущего keyframe И (б) кадр не смазан (variance of Laplacian > порога, подобрать ~100).
3. Для keyframe: сохранить rgb.jpg (полный кадр из соседнего branch'а tee до ресайза), masks.png (uint16), meta.json по контракту.
4. Логировать: fps инференса, число треков, число keyframes.
5. Управление: старт/стоп по Ctrl+C, имя сессии из timestamp.

Параметры камеры в записи: тот же LensPosition из M0, зафиксировать AWB и экспозицию (`AeEnable: False` после автоподбора в первые 2 с) — стабильность цвета помогает и COLMAP, и DINOv2.

### 3.4 Передача
```bash
rsync -avP session_*/ user@macbook.local:~/mapping/sessions/
```

**Приёмка M1:** 3-минутный обход комнаты по кругу → сессия с 300–400 keyframes; masks.png корректно накладываются на rgb.jpg (проверочный скрипт-визуализатор `tools/overlay_check.py`); track_id стабильны на статичных объектах ≥ 80% времени видимости; CPU Pi < 60%, температура < 75°C.

---

## 4. M2 — Сервер на Mac: окружение + метрическая глубина

### 4.1 Окружение
```bash
brew install colmap ffmpeg
conda create -n mapping python=3.11 -y && conda activate mapping
pip install torch torchvision            # MPS из коробки
pip install open3d opencv-python open_clip_torch timm einops \
            networkx scipy matplotlib pycolmap
export PYTORCH_ENABLE_MPS_FALLBACK=1     # добавить в ~/.zshrc
```
Проверка MPS: `python -c "import torch; print(torch.backends.mps.is_available())"` → True.

### 4.2 Depth Anything V2 metric (основной вариант)
```bash
git clone https://github.com/DepthAnything/Depth-Anything-V2
# чекпойнт: depth_anything_v2_metric_hypersim_vits.pth (~99 MB, indoor)
# с HF: huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Small
```
Скрипт `mac/step1_depth.py`:
- Модель `DepthAnythingV2(encoder='vits', max_depth=20)`, device='mps'.
- Для каждого keyframe: rgb.jpg → depth в метрах → сохранить `depth.npy` (float32, метры) + `depth_vis.png` для глаз.
- Замерить время/кадр (ожидание: десятки–сотни мс на MPS — для offline ок).

### 4.3 UniDepth V2 (альтернатива, флаг `--depth-model unidepth`)
`lpiccinelli-eth/UniDepth`, модель `unidepth-v2-vits14`. Плюс: предсказывает метрику увереннее на «не-hypersim» сценах и умеет свои интринсики (мы всё равно подаём K из M0). Держать как сменный бэкенд за одним интерфейсом `DepthBackend.predict(rgb, K) -> depth_m`.

**Приёмка M2:** на 5 тестовых кадрах с рулеткой: измерить реальную дистанцию до 3 объектов (1–4 м) → медианная ошибка глубины < 15%. Если > 25% — переключиться на UniDepth и повторить.

---

## 5. M3 — Позы камеры + единый масштаб

Скрипт `mac/step2_poses.py`:
1. COLMAP через pycolmap (или CLI): feature_extractor (SIMPLE_RADIAL заменить на PINHOLE с нашей K, `--ImageReader.camera_params "fx,fy,cx,cy"`, `--ImageReader.single_camera 1`) → sequential_matcher (кадры упорядочены!) → mapper.
2. Результат: позы T_wc для подмножества зарегистрированных кадров + sparse-точки.
3. **Выравнивание масштаба:** для каждого зарегистрированного кадра спроецировать sparse-точки COLMAP в кадр, взять их COLMAP-глубины d_colmap и глубины из depth.npy d_metric в тех же пикселях → per-frame scale = median(d_metric / d_colmap) → глобальный scale = median по кадрам (отбраковка кадров с MAD-выбросами). Умножить трансляции всех поз на scale. Сохранить `poses.json` ({frame_id: 4x4 T_wc в метрах}) + `scale_report.json`.
4. Fallback (если COLMAP регистрирует < 70% кадров — голые стены, мало текстуры): MASt3R парный режим на подвыборке кадров (каждый 5-й), позы из него; пометить в отчёте.

**Приёмка M3:** ≥ 80% keyframes зарегистрированы; траектория замыкается (начало/конец обхода по кругу в пределах < 0.3 м); разброс per-frame scale (IQR/median) < 15% — иначе глубина нестабильна, вернуться к M2.

---

## 6. M4 — TSDF-фьюжн

Скрипт `mac/step3_tsdf.py`:
```python
volume = o3d.pipelines.integration.ScalableTSDFVolume(
    voxel_length=0.015, sdf_trunc=0.06,
    color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
# для каждого кадра с позой:
#   rgbd = create_from_color_and_depth(rgb, depth_m, depth_scale=1.0,
#                                      depth_trunc=5.0, convert_rgb_to_intensity=False)
#   volume.integrate(rgbd, intrinsic, np.linalg.inv(T_wc))
mesh = volume.extract_triangle_mesh()
```
- Глубину предварительно: медианный фильтр 5×5, обрезка > 5 м, эрозия краёв масок глубины (артефакты на границах объектов у монокулярных моделей).
- Сохранить `room_mesh.ply`, скрипт просмотра `tools/view_mesh.py`.

**Приёмка M4:** стены плоские (без «двойных стен»), пол горизонтален; ширина комнаты по mesh vs рулетка — ошибка < 10%.

---

## 7. M5 — 3D-объекты и ассоциация (DINOv2)

Скрипт `mac/step4_objects.py`:
1. **Кандидаты:** для каждого keyframe и каждой маски: пиксели маски + depth.npy + K + T_wc → облако точек объекта в мировых координатах; фильтр DBSCAN (eps=0.05, min_points=20) — оставить крупнейший кластер.
2. **Эмбеддинги:** DINOv2 `dinov2_vits14` через `torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')`, device='mps'. Вход: кроп по bbox, фон вне маски залит средним цветом, ресайз 224×224 → CLS-токен (384-d), L2-нормировать. Дополнительно CLIP-эмбеддинг того же кропа (open_clip `ViT-B-16`, laion2b) — для семантики в M6.
3. **Object bank (инкрементально по кадрам):** ассоциация кандидата с существующим объектом, если: cos(DINOv2) > 0.6 И расстояние центроидов < 0.5 м (ИЛИ 3D-IoU bbox > 0.1). Приоритет — совпадение track_id с Pi (если track жив, ассоциировать сразу, эмбеддинг — для сшивки разорванных треков и повторных заходов в кадр). При мердже: объединить облака (voxel downsample 1 см), эмбеддинг — скользящее среднее.
4. **Финализация:** объекты, наблюдавшиеся < 3 keyframes — отбросить. Сохранить `objects.json` (id, class-гипотезы, centroid, obb, n_observations, dinov2_emb, clip_emb) + `objects_pcd/` (ply на объект).

**Приёмка M5:** на сцене с ~10 известными объектами: precision ≥ 80% (нет дублей одного стула), recall ≥ 70%; стул, вышедший из кадра и вернувшийся через 30 с — один объект, не два.

---

## 8. M6 — Open-vocab метки и граф сцены

Скрипт `mac/step5_graph.py`:
1. **Метки:** словарь indoor-классов (~50–80 слов: chair, table, sofa, monitor, plant, door, window...) → текстовые CLIP-эмбеддинги (посчитать один раз) → метка объекта = argmax cos(clip_emb, text_emb); хранить top-3 с вероятностями. COCO-метка с Pi — как prior (если согласуется с CLIP top-3 — буст уверенности).
2. **Рёбра (геометрические эвристики):**
   - `near(A,B)`: dist(obb A, obb B) < 0.5 м;
   - `on(A,B)`: горизонтальное пересечение проекций + низ A в пределах 5 см от верха B;
   - `in_room`: все объекты → узел «room» (пока одна комната).
3. Экспорт: `scene_graph.json` (networkx node-link) + визуализация: Open3D-сцена (mesh полупрозрачный + OBB объектов + подписи) и 2D-план сверху (matplotlib) с позициями и метками.

**Приёмка M6:** метки top-1 верны ≥ 70% объектов; отношения «монитор on стол», «стул near стол» присутствуют в графе.

---

## 9. M7 (опционально, потом)
- hailo-CLIP на Pi → open-vocab теги на борту (если захочется убрать CLIP с Mac).
- Замена COLMAP на MASt3R-SLAM-бэкенд (вариант C из ресёрча).
- Стриминг вместо rsync (GStreamer RTSP / shared-memory helper из hailo-apps).
- Интеграция IMU (см. ниже) — когда появится движущаяся платформа.

## 10. Структура репозитория для агента
```
mapping/
├── pi/            calibrate.py, recorder.py, requirements.txt
├── mac/           step1_depth.py ... step5_graph.py, backends/ (depth, poses)
├── tools/         overlay_check.py, view_mesh.py, view_graph.py
├── configs/       default.yaml (пороги, пути, выбор бэкендов)
└── docs/          этот план, контракт данных
```
Порядок работы агента: M0 → M1 (на Pi) параллельно с M2 (на Mac, на любом тестовом видео/кадрах), затем M3 → M4 → M5 → M6 строго последовательно — каждый шаг питается артефактами предыдущего, критерии приёмки в конце каждого раздела.
