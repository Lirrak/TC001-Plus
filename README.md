# TC001 Plus Thermal Viewer

Viewer Python cho TOPDON TC001 / TC001 Plus trên Windows. Mục tiêu chính của
project là đọc luồng radiometric thật từ TC001 Plus thông qua TOPDON SDK /
`libiruvc.dll`, sau đó hiển thị heatmap kèm nhiệt độ thật theo don vi Celsius.

> Luu y: cac che do OpenCV visual hoac `--estimate-temps` chi dung de xem anh
> mau/kiem tra camera. Nhiet do trong cac che do do khong phai du lieu do that
> tu TC001 Plus.

## Tinh nang hien co

- Mo cua so live viewer cho TC001 / TC001 Plus.
- Doc nhiet do that bang TOPDON SDK voi `--sdk-raw`.
- Hien thi average, center, min/max, hot/cold point va threshold.
- Ho tro color map, contrast, blur, zoom, rotate, flip, fullscreen.
- Co che do scan/list camera OpenCV de debug.
- Co snapshot, recording va cac probe/debug scripts trong `tools/`.

## Yeu cau

- Windows.
- Python 3.10+.
- TOPDON TopView da cai dat, mac dinh can thu muc:

```text
C:\Program Files\TOPDON\TopView\dll\dll_c001p
```

- TC001 Plus da cam vao may va khong bi ung dung khac giu camera.

## Cai dat

Tao virtual environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Cai dependency Python:

```powershell
py -m pip install -r requirements.txt
```

Neu khong dung `requirements.txt`, co the cai truc tiep:

```powershell
py -m pip install opencv-python numpy
```

## Lenh chay chinh

Mo viewer voi nhiet do that tu TC001 Plus:

```powershell
py .\tc001_thermal_viewer_v5.py --sdk-raw
```

Lenh nay la che do nen dung khi can do nhiet do that.

Project nay co chu y khong mo camera index `0`. Tat ca OpenCV visual/RGB va AI
mac dinh dung camera index `1`. Neu truyen `--device 0` hoac `--rgb-device 0`,
chuong trinh se dung ngay de tranh mo webcam tich hop.

## AI do tran va canh bao vat nong

AI chay theo kien truc 2 luong:

- `--sdk-raw` doc ma tran nhiet that tu TC001 Plus.
- `--rgb-device 1` doc anh RGB/visual de AI detect mat nguoi va vat the.

### 1. Calibration 4 diem

Chay mot lan de can khop anh RGB voi anh nhiet:

```powershell
py .\tc001_thermal_viewer_v5.py --sdk-raw --calibrate-ai --rgb-device 1
```

Trong cua so calibration:

1. Bam 4 diem de nhan ra tren anh RGB ben trai.
2. Bam dung 4 diem tuong ung tren anh nhiet ben phai.
3. Chuong trinh tu luu `tc001_alignment.json` sau khi du 8 click.

Nen dung cac diem co hinh dang ro, vi du 4 goc cua mot hop, to giay, man hinh
hoac vat the co canh ro. Neu thay doi vi tri/goc lap camera, hay calibration lai.

### 2. Do nhiet do tran

```powershell
py .\tc001_thermal_viewer_v5.py --sdk-raw --ai-forehead --rgb-device 1
```

Viewer se detect mat tren RGB, lay vung tran, map sang ma tran nhiet va hien:

```text
Forehead: 36.x degC
```

Nguong canh bao mac dinh la `37.5C`, co the doi:

```powershell
py .\tc001_thermal_viewer_v5.py --sdk-raw --ai-forehead --rgb-device 1 --forehead-threshold-c 37.8
```

Ket qua nay chi la ho tro giam sat, khong phai thiet bi chan doan y te.

### 3. Canh bao vat nong

Bat rule thermal phat hien vat nong:

```powershell
py .\tc001_thermal_viewer_v5.py --sdk-raw --hot-object-watch --rgb-device 1
```

Mac dinh vat nong la vung vuot `60C`, ton tai it nhat `2s`, va co dien tich tu
`8` pixel nhiet tro len. Co the chinh:

```powershell
py .\tc001_thermal_viewer_v5.py --sdk-raw --hot-object-watch --hot-threshold-c 55 --hot-persist-s 3
```

### 4. Chay ca do tran va vat nong

```powershell
py .\tc001_thermal_viewer_v5.py --sdk-raw --ai-forehead --hot-object-watch --rgb-device 1
```

### 5. YOLO label cho vat nguy hiem

Neu cai `ultralytics`, co the bat YOLO de gan nhan vat the gan hotspot:

```powershell
py .\tc001_thermal_viewer_v5.py --sdk-raw --hot-object-watch --rgb-device 1 --ai-use-yolo --ai-object-model yolo11n.pt
```

Luu y: model pretrained co the nhan duoc mot so lop pho bien nhu `laptop`,
`cell phone`, `tv`, nhung cac lop nhu `socket`, `outlet`, `adapter`, `power
strip` thuong can train/fine-tune model rieng de chinh xac.

## Cac lenh debug huu ich

Liet ke camera OpenCV:

```powershell
py .\tc001_thermal_viewer_v5.py --list --scan-min 1
```

Mo camera OpenCV index 1 de xem visual stream:

```powershell
py .\tc001_thermal_viewer_v5.py --device 1
```

Thu scan raw/radiometric qua OpenCV:

```powershell
py .\tc001_thermal_viewer_v5.py --find-raw --scan-min 1 --scan-max 5
```

Che do estimate tu visual frame, chi de tham khao:

```powershell
py .\tc001_thermal_viewer_v5.py --device 1 --estimate-temps --fallback-min-c 22 --fallback-max-c 45
```

## Dieu khien trong cua so viewer

- `q`: thoat.
- `c`: doi color map.
- `+` / `-`: tang/giam contrast.
- `z` / `x`: zoom in/out.
- `a`: fit heatmap vao cua so.
- `f`: fullscreen.
- `h`: bat/tat HUD.
- `l`: bat/tat label.
- `r`: record video.
- `s`: snapshot.
- `t` / `g`: tang/giam threshold.
- `u`: doi Celsius/Fahrenheit.
- `i`: invert mau.
- `?`: hien help overlay.

## Xu ly loi thuong gap

### `uvc_camera_stream_start failed: -11`

Thu cac buoc sau:

1. Dong TopView, Windows Camera, OBS, Teams, browser va cac ung dung co the dang
   giu camera.
2. Rut TC001 Plus ra va cam lai.
3. Chay lai:

```powershell
py .\tc001_thermal_viewer_v5.py --sdk-raw
```

Neu loi van con, khoi dong lai may hoac dung probe trong `tools/probes/` de
kiem tra thiet bi USB.

### Cua so hien `visual-estimate`

Day khong phai nhiet do that. Hay chay:

```powershell
py .\tc001_thermal_viewer_v5.py --sdk-raw
```

## Cau truc thu muc

```text
tc001_thermal_viewer_v5.py       Main viewer hien tai
tools/probes/                    Script probe SDK/libusb/TopView
tools/experiments/               Script thu nghiem raw frame
archive/legacy_viewers/          Cac ban viewer cu
captures/                        Anh/raw frame mau de debug
downloads/                       File tai ve phuc vu nghien cuu
```

## Ghi chu ve kien truc AI

Phan AI hien tai tach thanh hai luong:

- TC001 Plus SDK doc ma tran nhiet that `256x192`.
- OpenCV `cam1` doc RGB/visual stream cho AI detect.

AI dang dung/ho tro:

- MediaPipe Face Mesh de detect mat va uoc luong vung tran.
- Ultralytics YOLO optional de gan nhan vat the gan hotspot.

Vung detect tren RGB can duoc map sang ma tran nhiet bang calibration 4 diem thu
cong, de lay nhiet do that tu TC001 Plus thay vi do tu anh mau.
