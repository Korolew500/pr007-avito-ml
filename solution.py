'''Определение ориентации текста на кропе (0° или 180°).

ПОДХОД
======
1. Обучение синтетическое: реальные метки не даны, поэтому кропы с текстом
   генерируются на лету — рендер случайных слов системными шрифтами на
   случайном фоне, половина переворачивается на 180°. Это даёт бесконечный
   поток размеченных данных и учит модель на инвариантах текста.
2. Модель — компактная CNN (~0.4M параметров), быстрая на CPU.
3. Препроцессинг: высота 48 px с сохранением пропорций, паддинг до max
   ширины в батче, нормализация в [-1, 1].
4. Инференс: sigmoid(model(x)) -> p_180 — вероятность поворота на 180°.
5. SEED=42 фиксирует random / numpy / torch — воспроизводимость.

ГДЕ ИЩУТСЯ ТЕСТОВЫЕ ДАННЫЕ
==========================
Все пути — относительно папки, где лежит этот скрипт (корень решения).
Скрипт автоматически перебирает стандартные расположения:

  data/test/*.jpg          — основной вариант (датасет организатора)
  test/*.jpg
  data/images/*.jpg
  images/*.jpg
  data/*.jpg

Также поддерживается CSV-режим (data/test.csv или test.csv): распознаются
колонки image_id и image_path (или их синонимы). Если в CSV только image_id,
путь достраивается по имени файла в соответствующей папке.

Если данные не найдены, скрипт печатает список всех проверенных путей и
завершается с ошибкой — тогда нужно положить картинки в одну из папок выше.

ПОЧЕМУ ТАК
==========
- Синтетика даёт бесплатные метки и учит модель на инвариантах, а не на шуме.
- Компактная модель выбрана осознанно: в проде решение будет применяться
  к миллионам кропов в день, поэтому скорость и размер критичны.
- BCEWithLogitsLoss — логарифмическое правдоподобие для Бернулли; его
  минимизация напрямую ведёт к минимизации метрики Brier.
- AdaptiveAvgPool2d((4, 1)) сохраняет вертикальную структуру — ключевой
  сигнал для различения 0° и 180° (верх vs низ букв).
- В словарь генератора включена кириллица: реальные данные русскоязычные.
'''

import sys, random, time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ---------- ВОСПРОИЗВОДИМОСТЬ ----------
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ---------- КОНСТАНТЫ ----------
WORK = Path(__file__).resolve().parent
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = WORK / 'model.pt'
OUT_CSV = WORK / 'submission.csv'

CROP_H = 48
MAX_W = 320
SYN_N = 12000
EPOCHS = 8
BATCH = 128

print(f'Device: {DEVICE}', flush=True)


# ============================================================
# СИНТЕТИЧЕСКИЙ ГЕНЕРАТОР
# ============================================================

def collect_fonts():
    # Собираем ttf/otf/ttc из системных папок Windows.
    dirs = [Path('C:/Windows/Fonts'),
            Path.home() / 'AppData/Local/Microsoft/Windows/Fonts']
    out = []
    for d in dirs:
        if d.exists():
            for ext in ('*.ttf', '*.otf', '*.ttc'):
                out.extend(str(p) for p in d.rglob(ext))
    return out

FONTS = collect_fonts()
print(f'Fonts found: {len(FONTS)}', flush=True)

# Словарь: латиница + кириллица + типичные для Avito вывески и бренды.
WORDS = ['coffee','chocolate','sale','milk','bread','water','fresh','organic',
         'natural','cafe','tea','green','black','vanilla','almond','sugar','salt',
         'rice','olive','oil','apple','orange','tomato','cheese','yogurt','butter',
         'cream','juice','cola','sprite','pepsi','nike','adidas','sony','philips',
         'bosch','siemens','coca','mars','snickers','bounty','twix','nestle',
         'danone','heinz','kelloggs','nescafe','jacobs','lavazza','illy','magnit',
         'pyaterochka','auchan','lenta','billa',
         'молоко','хлеб','масло','сыр','колбаса','вода','сок','пиво',
         'шоколад','конфеты','печенье','чай','кофе','сахар','соль','мука',
         'магнит','лента','ашан','дикси','акция','скидка','новинка','распродажа',
         'свежее','натуральное','вкусно','полезно','эко','био','фермерское']


def rand_color(lo=0, hi=255):
    # Случайный RGB-цвет в заданном диапазоне яркости.
    return tuple(random.randint(lo, hi) for _ in range(3))


def random_text():
    # Фраза из 1-4 случайных слов.
    return ' '.join(random.choice(WORDS) for _ in range(random.randint(1, 4)))


def render_crop():
    # Отрисовка одного синтетического кропа + аугментации JPEG/blur.
    W = random.randint(80, 320)
    H = random.randint(32, 96)

    # --- фон: однотонная заливка / градиент / шум ---
    r = random.random()
    if r < 0.45:
        bg = Image.new('RGB', (W, H), rand_color())
    elif r < 0.75:
        c1 = np.array(rand_color()); c2 = np.array(rand_color())
        arr = np.zeros((H, W, 3), dtype=np.uint8)
        vert = random.random() < 0.5
        n = H if vert else W
        for i in range(n):
            t = i / max(1, n - 1)
            c = (c1 * (1 - t) + c2 * t).astype(np.uint8)
            if vert: arr[i, :] = c
            else:    arr[:, i] = c
        bg = Image.fromarray(arr)
    else:
        arr = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
        bg = Image.fromarray(arr)

    draw = ImageDraw.Draw(bg)
    text = random_text()

    # --- шрифт и размер ---
    fs = random.randint(max(10, H // 2), max(12, int(H * 0.9)))
    font = None
    if FONTS:
        try: font = ImageFont.truetype(random.choice(FONTS), fs)
        except Exception: font = None
    if font is None:
        try: font = ImageFont.load_default()
        except Exception: return bg

    try:
        bb = draw.textbbox((0, 0), text, font=font)
        tw, th = bb[2] - bb[0], bb[3] - bb[1]
    except Exception:
        tw, th, bb = W, H, (0, 0, 0, 0)

    # Если текст не влезает по ширине — уменьшаем шрифт.
    if tw > W - 4 and tw > 0 and FONTS:
        fs = max(8, int(fs * (W - 4) / tw))
        try:
            font = ImageFont.truetype(random.choice(FONTS), fs)
            bb = draw.textbbox((0, 0), text, font=font)
            tw, th = bb[2] - bb[0], bb[3] - bb[1]
        except Exception:
            pass

    # Контрастный цвет текста к фону.
    mean_bg = sum(bg.resize((1, 1)).getpixel((0, 0)))
    fg = rand_color(0, 80) if mean_bg > 380 else rand_color(170, 255)

    x = max(0, (W - tw) // 2 - bb[0])
    y = max(0, (H - th) // 2 - bb[1])
    draw.text((x, y), text, fill=fg, font=font)

    # --- аугментации: JPEG-артефакты и расфокус ---
    import io as _io
    if random.random() < 0.5:
        quality = random.randint(30, 85)
        buf = _io.BytesIO()
        bg.save(buf, format='JPEG', quality=quality)
        buf.seek(0)
        bg = Image.open(buf).convert('RGB')
    if random.random() < 0.3:
        bg = bg.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.2)))
    return bg


# ============================================================
# ПРЕПРОЦЕССИНГ
# ============================================================

def preprocess(img):
    # Ресайз по высоте с сохранением пропорций + нормализация в [-1, 1].
    w, h = img.size
    nw = max(8, min(MAX_W, int(w * CROP_H / max(1, h))))
    img = img.resize((nw, CROP_H), Image.BILINEAR)
    a = np.asarray(img, dtype=np.float32) / 255.0
    a = (a - 0.5) / 0.5
    return a.transpose(2, 0, 1)


class SynthDataset(Dataset):
    # Онлайн-генератор: каждый __getitem__ создаёт новый кроп.
    # Метка 0 — исходная ориентация, 1 — поворот на 180.
    def __init__(self, n): self.n = n
    def __len__(self): return self.n
    def __getitem__(self, _):
        img = render_crop()
        y = random.randint(0, 1)
        if y == 1:
            img = img.rotate(180, expand=True)
        return torch.from_numpy(preprocess(img)), float(y)


def collate(batch):
    # Паддинг до max ширины в батче.
    xs, ys = zip(*batch)
    mw = max(x.shape[2] for x in xs)
    xs = [F.pad(x, (0, mw - x.shape[2])) for x in xs]
    return torch.stack(xs), torch.tensor(ys)


# ============================================================
# МОДЕЛЬ
# ============================================================

class TinyNet(nn.Module):
    # Компактная CNN. AdaptiveAvgPool2d((4, 1)) сохраняет вертикальные
    # полосы — это ключевой сигнал для различения 0 и 180 градусов.
    def __init__(self):
        super().__init__()
        def blk(ci, co, s=2):
            return nn.Sequential(
                nn.Conv2d(ci, co, 3, s, 1, bias=False),
                nn.BatchNorm2d(co),
                nn.ReLU(inplace=True))
        self.f = nn.Sequential(
            blk(3, 32, 1), blk(32, 32),
            blk(32, 64), blk(64, 64),
            blk(64, 128), blk(128, 128),
            nn.AdaptiveAvgPool2d((4, 1)))
        self.fc = nn.Linear(128 * 4, 1)

    def forward(self, x):
        return self.fc(self.f(x).flatten(1)).squeeze(1)


# ============================================================
# ПОИСК ТЕСТОВЫХ ДАННЫХ
# ============================================================

def discover_test():
    # Порядок проверки:
    #   1. CSV + папка с картинками (если организатор выдал CSV)
    #   2. Просто папка с картинками в стандартных местах
    # Все пути — относительно папки, где лежит solution.py.
    cands = [
        (WORK / 'data' / 'test.csv', WORK / 'data' / 'test'),
        (WORK / 'test.csv',          WORK / 'test'),
        (WORK / 'data' / 'test.csv', WORK / 'data' / 'images'),
        (WORK / 'test.csv',          WORK / 'images'),
    ]
    for cp, idir in cands:
        if cp.exists(): return cp, idir

    img_dirs = [WORK / 'data' / 'test', WORK / 'test',
                WORK / 'data' / 'images', WORK / 'images', WORK / 'data']
    for d in img_dirs:
        if d.exists():
            fs = []
            for e in ('*.jpg','*.jpeg','*.png','*.webp','*.bmp','*.JPG','*.PNG'):
                fs += list(d.glob(e))
            if fs: return None, d
    return None, None


def load_test_items():
    csv_p, img_dir = discover_test()
    if csv_p is None and img_dir is None:
        # Печатаем все места, где искали — чтобы было понятно, куда положить файлы.
        print('ERROR: test data not found.', flush=True)
        print('Looked for images in (any of these works):', flush=True)
        for p in ('data/test/', 'test/', 'data/images/', 'images/', 'data/'):
            print(f'  - {WORK / p}', flush=True)
        print('Looked for CSV + folder in:', flush=True)
        for p in ('data/test.csv + data/test/', 'test.csv + test/',
                  'data/test.csv + data/images/', 'test.csv + images/'):
            print(f'  - {WORK / p}', flush=True)
        print('Put the test images into one of the folders above and re-run.', flush=True)
        sys.exit(1)

    if csv_p is not None:
        df = pd.read_csv(csv_p)
        print(f'CSV: {csv_p} rows={len(df)} cols={list(df.columns)}', flush=True)
        id_c = path_c = None
        for c in df.columns:
            cl = c.lower().strip()
            if cl in ('image_id','id','filename','file','name','img_id'): id_c = c
            if cl in ('image_path','path','filepath','file_path','img_path'): path_c = c
        if id_c is None: id_c = df.columns[0]
        ids = df[id_c].astype(str).tolist()
        if path_c:
            paths = []
            for p in df[path_c].astype(str):
                pp = Path(p)
                if not pp.is_absolute(): pp = WORK / pp
                paths.append(pp)
        else:
            def res(i):
                for e in ('.jpg','.jpeg','.png','.webp','.bmp','.JPG','.PNG'):
                    p = img_dir / (i + e)
                    if p.exists(): return p
                return None
            paths = [res(i) for i in ids]
        return ids, paths

    fs = []
    for e in ('*.jpg','*.jpeg','*.png','*.webp','*.bmp'):
        fs += list(img_dir.glob(e))
    fs = sorted(fs)
    print(f'Found {len(fs)} images in {img_dir}', flush=True)
    return [f.stem for f in fs], fs


# ============================================================
# ОБУЧЕНИЕ
# ============================================================

def train():
    print(f'Training: {SYN_N} samples, {EPOCHS} epochs, batch={BATCH}', flush=True)
    try:
        dl = DataLoader(SynthDataset(SYN_N), batch_size=BATCH, num_workers=2,
                        collate_fn=collate, shuffle=True, persistent_workers=False)
    except Exception as e:
        print(f'num_workers=2 failed ({e}), fallback to 0', flush=True)
        dl = DataLoader(SynthDataset(SYN_N), batch_size=BATCH, num_workers=0,
                        collate_fn=collate, shuffle=True)

    model = TinyNet().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    lossf = nn.BCEWithLogitsLoss()

    model.train()
    for ep in range(EPOCHS):
        t0 = time.time(); tot = n = 0
        for xb, yb in dl:
            xb = xb.to(DEVICE); yb = yb.to(DEVICE)
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward(); opt.step()
            tot += loss.item() * xb.size(0); n += xb.size(0)
        sched.step()
        print(f'  epoch {ep+1}/{EPOCHS}  loss={tot/n:.4f}  {time.time()-t0:.1f}s', flush=True)

    torch.save(model.state_dict(), MODEL_PATH)
    print(f'Saved model -> {MODEL_PATH}', flush=True)
    return model


def load_model():
    m = TinyNet().to(DEVICE)
    m.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    m.eval()
    return m


# ============================================================
# ИНФЕРЕНС
# ============================================================

@torch.no_grad()
def _run_batch(model, buf):
    idx = [i for i, x in enumerate(buf) if x is not None]
    out = [0.5] * len(buf)
    if not idx: return out
    xs = [torch.from_numpy(buf[i]) for i in idx]
    mw = max(x.shape[2] for x in xs)
    xs = [F.pad(x, (0, mw - x.shape[2])) for x in xs]
    xb = torch.stack(xs).to(DEVICE)
    p = torch.sigmoid(model(xb)).cpu().numpy().tolist()
    for k, i in enumerate(idx): out[i] = float(p[k])
    return out


def predict(model, paths):
    probs = []
    B = 64
    buf = []
    total = len(paths)
    n_done = 0
    for p in paths:
        if p is None or not Path(p).exists():
            buf.append(None)
        else:
            try:
                buf.append(preprocess(Image.open(p).convert('RGB')))
            except Exception:
                buf.append(None)
        if len(buf) == B:
            probs += _run_batch(model, buf)
            buf = []
            n_done += B
            if n_done % 2000 == 0 or n_done >= total:
                print(f'  predicting {n_done}/{total}', flush=True)
    if buf:
        probs += _run_batch(model, buf)
        n_done += len(buf)
        print(f'  predicting {n_done}/{total}', flush=True)
    return probs


# ============================================================
# MAIN
# ============================================================

def main():
    if MODEL_PATH.exists():
        print(f'Model exists: {MODEL_PATH}, skip training', flush=True)
        model = load_model()
    else:
        model = train()
    ids, paths = load_test_items()
    probs = predict(model, paths)
    sub = pd.DataFrame({'image_id': ids, 'p_180': probs})
    sub.to_csv(OUT_CSV, index=False)
    print(f'Wrote {OUT_CSV} ({len(sub)} rows)', flush=True)
    print(sub.head(), flush=True)
    print(sub['p_180'].describe(), flush=True)


if __name__ == '__main__':
    main()
