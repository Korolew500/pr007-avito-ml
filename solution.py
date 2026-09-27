'''Определение ориентации текста на кропе (0° или 180°).

ПОДХОД
======
1. Синтетическое обучение: кропы генерируются на лету в формате реальных
   вывесок и логотипов — широкие полоски, крупный текст, обрезка краёв.
2. Backbone: MobileNetV3-Small (ImageNet). Разморожены последние 3 блока
   features и голова; остальные features заморожены.
3. Голова: AdaptiveAvgPool2d((4, 1)) вместо global average — сохраняет
   вертикальную структуру (верх vs низ букв), что критично для задачи 0/180.
4. Аугментации: JPEG, Gaussian/motion blur, яркость, контраст, hue,
   насыщенность, зерно, перспективный варп.
5. TTA при инференсе: прогон в исходном виде + поворот 180°, усреднение
   p и (1 - p).
6. Препроцессинг: высота 64 px, нормализация по статистикам ImageNet.
7. SEED=42 фиксирует random / numpy / torch.

ЛОГИ
====
Прогресс пишется в train.log (num_workers=0, без перезаписи).

ГДЕ ИЩУТСЯ ТЕСТОВЫЕ ДАННЫЕ
==========================
Относительно папки скрипта:
  data/test/*.jpg    test/*.jpg    data/images/*.jpg    images/*.jpg
'''

import sys, random, time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

# ---------- ВОСПРОИЗВОДИМОСТЬ ----------
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ---------- КОНСТАНТЫ ----------
WORK = Path(__file__).resolve().parent
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = WORK / 'model.pt'
OUT_CSV = WORK / 'submission.csv'
LOG_PATH = WORK / 'train.log'

CROP_H = 64
MAX_W = 1024
ASPECT_MIN = 2.0
ASPECT_MAX = 16.0
SYN_N = 20000
EPOCHS = 12
BATCH = 64
UNFREEZE_TAIL = 3

IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMG_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_log_f = open(LOG_PATH, 'w', encoding='utf-8')

def log(*args):
    msg = ' '.join(str(a) for a in args)
    print(msg, flush=True)
    _log_f.write(msg + '\n')
    _log_f.flush()

log(f'Device: {DEVICE}')
log(f'Log file: {LOG_PATH}')


# ============================================================
# СИНТЕТИЧЕСКИЙ ГЕНЕРАТОР
# ============================================================

def collect_fonts():
    dirs = [Path('C:/Windows/Fonts'),
            Path.home() / 'AppData/Local/Microsoft/Windows/Fonts']
    out = []
    for d in dirs:
        if d.exists():
            for ext in ('*.ttf', '*.otf', '*.ttc'):
                out.extend(str(p) for p in d.rglob(ext))
    return out

ALL_FONTS = collect_fonts()
# Отдельно выделяем жирные (bold/black/heavy) — они дают вывески.
BOLD_HINTS = ('bold','black','heavy','impact','bebas','oswald','anton',
              'condensed','narrow','grotesk','druk','geometria')
BOLD_FONTS = [f for f in ALL_FONTS if any(h in f.lower() for h in BOLD_HINTS)]
OTHER_FONTS = [f for f in ALL_FONTS if f not in BOLD_FONTS]
FONTS = ALL_FONTS
log(f'Fonts: total={len(ALL_FONTS)}, bold_hint={len(BOLD_FONTS)}')

# Словарь: продукты, бренды, вывески, магазины — латиница + кириллица.
WORDS = ['COFFEE','CHOCOLATE','SALE','MILK','BREAD','WATER','FRESH','ORGANIC',
         'NATURAL','CAFE','TEA','GREEN','BLACK','VANILLA','ALMOND','SUGAR','SALT',
         'RICE','OLIVE','OIL','APPLE','ORANGE','TOMATO','CHEESE','YOGURT','BUTTER',
         'CREAM','JUICE','COLA','SPRITE','PEPSI','NIKE','ADIDAS','SONY','PHILIPS',
         'BOSCH','SIEMENS','COCA','MARS','SNICKERS','BOUNTY','TWIX','NESTLE',
         'DANONE','HEINZ','KELLOGGS','NESCAFE','JACOBS','LAVAZZA','ILLY','MAGNIT',
         'PYATEROCHKA','AUCHAN','LENTA','BILLA',
         'STORE','SHOP','BAR','HOTEL','MARKET','MALL','PERFUME','PHARMACY',
         'SANYO','MODEL','CODE','PARFUMS','STYLO','FASHION','BEAUTY','SALON',
         'CENTER','CENTRE','ECO','BIO','FARM','FRESH','TASTE','PIZZA','BURGER',
         'ICE CREAM','YOGURT','BAKERY',
         'МОЛОКО','ХЛЕБ','МАСЛО','СЫР','КОЛБАСА','ВОДА','СОК','ПИВО',
         'ШОКОЛАД','КОНФЕТЫ','ПЕЧЕНЬЕ','ЧАЙ','КОФЕ','САХАР','СОЛЬ','МУКА',
         'МАГНИТ','ЛЕНТА','АШАН','ДИКСИ','АКЦИЯ','СКИДКА','НОВИНКА','РАСПРОДАЖА',
         'СВЕЖЕЕ','НАТУРАЛЬНОЕ','ВКУСНО','ПОЛЕЗНО','ЭКО','БИО','ФЕРМЕРСКОЕ',
         'ПЯТЁРОЧКА','ПЕРЕКРЁСТОК','ОТКРЫТИЕ','МАГАЗИН','АПТЕКА','САЛОН',
         'КРАСОТА','ЦЕНТР','ЭКОНОМ','ПРОДУКТЫ','ДОМ','БЫТА','ДОКА',
         'НИКОЛАЕВСКАЯ','ПАРИЖ','ЛОНДОН','МОСКВА','РОССИЯ','ДРУЖБА',
         'СТИЛЬ','МОДА','ОБУВЬ','ОДЕЖДА','ПОДАРКИ','СУВЕНИРЫ','КНИГИ',
         'РЕСТОРАН','КАФЕ','БАР','ПИЦЦА','СУШИ','БЛИНЫ','ПЕКАРНЯ']


def rand_color(lo=0, hi=255):
    return tuple(random.randint(lo, hi) for _ in range(3))


def random_text():
    n = random.choices([1, 2, 3], weights=[0.6, 0.3, 0.1])[0]
    return ' '.join(random.choice(WORDS) for _ in range(n))


def make_bg(W, H):
    r = random.random()
    if r < 0.30:
        bg = Image.new('RGB', (W, H), rand_color())
    elif r < 0.55:
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
    elif r < 0.75:
        c1 = np.array(rand_color()); c2 = np.array(rand_color())
        yy, xx = np.mgrid[0:H, 0:W]
        t = (xx / max(1, W - 1) + yy / max(1, H - 1)) / 2.0
        arr = np.zeros((H, W, 3), dtype=np.uint8)
        for k in range(3):
            arr[..., k] = (c1[k] * (1 - t) + c2[k] * t).astype(np.uint8)
        bg = Image.fromarray(arr)
    elif r < 0.90:
        arr = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
        bg = Image.fromarray(arr)
    else:
        arr = np.zeros((H, W, 3), dtype=np.uint8)
        c1 = np.array(rand_color()); c2 = np.array(rand_color())
        period = random.randint(4, max(5, H // 3))
        for y in range(H):
            c = c1 if (y // period) % 2 == 0 else c2
            arr[y, :] = c
        bg = Image.fromarray(arr)
    return bg


def pick_font_path():
    # 60% жирных (вывески), 40% из всех.
    if BOLD_FONTS and random.random() < 0.6:
        return random.choice(BOLD_FONTS)
    return random.choice(FONTS)


def fit_font(text, W, H):
    # Масштабируем так, чтобы текст занял 60–110% ширины (иногда вылезает).
    if not FONTS:
        return ImageFont.load_default()
    for _ in range(6):
        try:
            fpath = pick_font_path()
            probe = ImageFont.truetype(fpath, 100)
            tmp = Image.new('RGB', (10, 10))
            bb = ImageDraw.Draw(tmp).textbbox((0, 0), text, font=probe)
            w100 = bb[2] - bb[0]
            h100 = bb[3] - bb[1]
            if w100 <= 0 or h100 <= 0:
                continue
            target_w = W * random.uniform(0.60, 1.10)
            target_h = H * random.uniform(0.60, 1.05)
            scale = min(target_w / w100, target_h / h100)
            fs = max(10, int(100 * scale))
            return ImageFont.truetype(fpath, fs)
        except Exception:
            continue
    try:
        return ImageFont.load_default()
    except Exception:
        return None


def perspective_warp(img, k=0.06):
    # Небольшой perspective — как будто фото под углом.
    W, H = img.size
    mx = W * random.uniform(0, k)
    my = H * random.uniform(0, k)
    def rnd(): return random.uniform(-1, 1)
    coeffs = [
        1 + rnd()*k, rnd()*k, mx,
        rnd()*k, 1 + rnd()*k, my,
        rnd()*k*0.001, rnd()*k*0.001,
    ]
    try:
        return img.transform((W, H), Image.PERSPECTIVE, coeffs, Image.BICUBIC)
    except Exception:
        return img


def augment(img):
    # JPEG
    if random.random() < 0.5:
        quality = random.randint(25, 85)
        import io as _io
        buf = _io.BytesIO()
        img.save(buf, format='JPEG', quality=quality)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')
    # Gaussian blur
    if random.random() < 0.35:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.4)))
    # Motion blur
    if random.random() < 0.25:
        k = random.choice([3, 5, 7])
        blurred = img.copy()
        for off in range(1, k):
            shifted = Image.new('RGB', img.size, (0, 0, 0))
            shifted.paste(img, (-off, 0))
            blurred = Image.blend(blurred, shifted, 1.0 / k)
        img = blurred
    # Перспектива
    if random.random() < 0.3:
        img = perspective_warp(img, k=random.uniform(0.03, 0.10))
    # Яркость/контраст
    if random.random() < 0.5:
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.7, 1.3))
    if random.random() < 0.4:
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.7, 1.3))
    # Hue / saturation
    if random.random() < 0.3:
        img = ImageEnhance.Color(img).enhance(random.uniform(0.6, 1.4))
    # Зерно
    if random.random() < 0.4:
        arr = np.asarray(img).astype(np.int16)
        noise = np.random.randint(-15, 15, arr.shape, dtype=np.int16)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
    return img


def render_crop():
    aspect = random.uniform(ASPECT_MIN, ASPECT_MAX)
    H = random.randint(32, 128)
    W = max(32, min(int(H * aspect), MAX_W))
    bg = make_bg(W, H)
    text = random_text()
    font = fit_font(text, W, H)
    if font is None:
        return bg
    draw = ImageDraw.Draw(bg)
    try:
        mean_bg = sum(bg.resize((1, 1)).getpixel((0, 0)))
    except Exception:
        mean_bg = 380
    fg = rand_color(0, 80) if mean_bg > 380 else rand_color(170, 255)
    try:
        bb = draw.textbbox((0, 0), text, font=font)
        tw, th = bb[2] - bb[0], bb[3] - bb[1]
        # небольшой сдвиг — иногда текст смещается к краю
        shift_x = random.randint(-int(W*0.08), int(W*0.08))
        shift_y = random.randint(-int(H*0.08), int(H*0.08))
        x = (W - tw) // 2 - bb[0] + shift_x
        y = (H - th) // 2 - bb[1] + shift_y
        draw.text((x, y), text, fill=fg, font=font)
    except Exception:
        return bg
    return augment(bg)


# ============================================================
# ПРЕПРОЦЕССИНГ
# ============================================================

def preprocess(img):
    w, h = img.size
    nw = max(8, min(MAX_W, int(w * CROP_H / max(1, h))))
    img = img.resize((nw, CROP_H), Image.BILINEAR)
    a = np.asarray(img, dtype=np.float32) / 255.0
    a = (a - IMG_MEAN) / IMG_STD
    return a.transpose(2, 0, 1)


class SynthDataset(Dataset):
    def __init__(self, n): self.n = n
    def __len__(self): return self.n
    def __getitem__(self, _):
        img = render_crop()
        y = random.randint(0, 1)
        if y == 1:
            img = img.transpose(Image.ROTATE_180)
        return torch.from_numpy(preprocess(img)), float(y)


def collate(batch):
    xs, ys = zip(*batch)
    mw = max(x.shape[2] for x in xs)
    xs = [F.pad(x, (0, mw - x.shape[2])) for x in xs]
    return torch.stack(xs), torch.tensor(ys)


# ============================================================
# МОДЕЛЬ
# ============================================================

class OriNet(nn.Module):
    # MobileNetV3-Small features + AdaptiveAvgPool2d((4, 1)) + Linear.
    # Вертикальные полосы сохраняют верх/низ букв — сигнал для 0/180.
    def __init__(self, pretrained=True):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        m = mobilenet_v3_small(weights=weights)
        n_blocks = len(m.features)
        for i, blk in enumerate(m.features):
            freeze = i < (n_blocks - UNFREEZE_TAIL)
            for p in blk.parameters():
                p.requires_grad = not freeze
        self.features = m.features
        # На выходе features — (B, 576, H', W'). Возьмём (4, 1) пул.
        self.pool = nn.AdaptiveAvgPool2d((4, 1))
        self.head = nn.Sequential(
            nn.Linear(576 * 4, 512),
            nn.Hardswish(),
            nn.Dropout(0.2),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        f = self.features(x)         # (B, 576, h, w)
        f = self.pool(f)             # (B, 576, 4, 1)
        f = f.flatten(1)             # (B, 2304)
        return self.head(f).squeeze(1)


# ============================================================
# ПОИСК ТЕСТОВЫХ ДАННЫХ
# ============================================================

def discover_test():
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
        log('ERROR: test data not found.')
        for p in ('data/test/', 'test/', 'data/images/', 'images/', 'data/'):
            log(f'  - {WORK / p}')
        sys.exit(1)
    if csv_p is not None:
        df = pd.read_csv(csv_p)
        log(f'CSV: {csv_p} rows={len(df)} cols={list(df.columns)}')
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
    log(f'Found {len(fs)} images in {img_dir}')
    return [f.stem for f in fs], fs


# ============================================================
# ОБУЧЕНИЕ
# ============================================================

def train():
    log(f'Training: {SYN_N} samples, {EPOCHS} epochs, batch={BATCH}')
    log('Loading MobileNetV3-Small pretrained weights...')
    dl = DataLoader(SynthDataset(SYN_N), batch_size=BATCH, num_workers=0,
                    collate_fn=collate, shuffle=True)
    model = OriNet(pretrained=True).to(DEVICE)

    feat_params = [p for p in model.features.parameters() if p.requires_grad]
    head_params = list(model.head.parameters())
    log(f'Trainable features params: {sum(p.numel() for p in feat_params)}')
    log(f'Trainable head params:     {sum(p.numel() for p in head_params)}')

    opt = torch.optim.AdamW([
        {'params': feat_params, 'lr': 3e-4},
        {'params': head_params, 'lr': 1e-3},
    ], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    lossf = nn.BCEWithLogitsLoss()

    model.train()
    n_batches = len(dl)
    for ep in range(EPOCHS):
        t0 = time.time(); tot = n = correct = 0
        for bi, (xb, yb) in enumerate(dl):
            xb = xb.to(DEVICE); yb = yb.to(DEVICE)
            opt.zero_grad()
            logits = model(xb)
            loss = lossf(logits, yb)
            loss.backward(); opt.step()
            bs = xb.size(0)
            tot += loss.item() * bs; n += bs
            with torch.no_grad():
                pred = (torch.sigmoid(logits) > 0.5).float()
                correct += (pred == yb).sum().item()
            if (bi + 1) % 60 == 0 or (bi + 1) == n_batches:
                log(f'  ep {ep+1}/{EPOCHS}  batch {bi+1}/{n_batches}  '
                    f'loss={tot/n:.4f}  acc={correct/n:.4f}')
        sched.step()
        log(f'  epoch {ep+1}/{EPOCHS}  loss={tot/n:.4f}  acc={correct/n:.4f}  '
            f'{time.time()-t0:.1f}s')
    torch.save(model.state_dict(), MODEL_PATH)
    log(f'Saved model -> {MODEL_PATH}')
    return model


def load_model():
    m = OriNet(pretrained=False).to(DEVICE)
    m.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    m.eval()
    return m


# ============================================================
# ИНФЕРЕНС с TTA
# ============================================================

@torch.no_grad()
def _run_batch_tta(model, buf):
    # TTA: прогон в исходном виде + вертикально перевёрнутом (это 180°).
    # p = (p_orig + (1 - p_flipped)) / 2
    idx = [i for i, x in enumerate(buf) if x is not None]
    out = [0.5] * len(buf)
    if not idx: return out
    xs = [torch.from_numpy(buf[i]) for i in idx]
    mw = max(x.shape[2] for x in xs)
    xs = [F.pad(x, (0, mw - x.shape[2])) for x in xs]
    xb = torch.stack(xs).to(DEVICE)                     # (N, 3, H, W)
    xb_flip = torch.flip(xb, dims=[2, 3])               # 180° поворот

    p1 = torch.sigmoid(model(xb)).cpu().numpy()
    p2 = torch.sigmoid(model(xb_flip)).cpu().numpy()
    p = (p1 + (1.0 - p2)) / 2.0

    for k, i in enumerate(idx):
        out[i] = float(p[k])
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
            probs += _run_batch_tta(model, buf)
            buf = []
            n_done += B
            if n_done % 2000 == 0 or n_done >= total:
                log(f'  predicting {n_done}/{total}')
    if buf:
        probs += _run_batch_tta(model, buf)
        n_done += len(buf)
        log(f'  predicting {n_done}/{total}')
    return probs


# ============================================================
# MAIN
# ============================================================

def main():
    if MODEL_PATH.exists():
        log(f'Model exists: {MODEL_PATH}, skip training')
        model = load_model()
    else:
        model = train()
    ids, paths = load_test_items()
    probs = predict(model, paths)
    sub = pd.DataFrame({'image_id': ids, 'p_180': probs})
    sub.to_csv(OUT_CSV, index=False)
    log(f'Wrote {OUT_CSV} ({len(sub)} rows)')
    log('Statistics of p_180:')
    log(sub['p_180'].describe().to_string())


if __name__ == '__main__':
    main()
