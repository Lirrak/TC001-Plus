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

Tuy chon: cai MediaPipe de face detection tot hon. Neu khong cai, chuong trinh
se fallback sang Haar cascade cua OpenCV:

```powershell
py -m pip install mediapipe
```

## Lenh chay chinh

Mo viewer voi nhiet do that tu TC001 Plus:

```powershell
py .\tc001_thermal_viewer_v5.py --sdk-raw
```

Lenh nay la che do nen dung khi can do nhiet do that.

Mo viewer SDK raw va detect mat bang nguon AI mac dinh `sdk-top`:

```powershell
py .\tc001_thermal_viewer_v5.py --sdk-raw --face-detect --show-digital-debug
```

Mac dinh AI dung `--digital-source sdk-top`, tuc la decode nua tren cua frame
SDK `256x384 YUY2` thay vi mo them OpenCV camera1. Neu `sdk-top` khong ra anh
digital that, thu fallback OpenCV/MSMF:

Frame digital duoc xoay rieng bang `--digital-rotate 90` theo mac dinh. Neu anh
bi xoay nguoc huong, thu `--digital-rotate 270`.

Face/head detection hien tai dung nhieu tang:

- `FACE`: detect mat chinh dien bang MediaPipe hoac Haar frontal.
- `PROFILE`: detect mat nghieng bang profile cascade.
- `HEAD TRACK`: fallback theo vung dau/than tren sang trong anh `sdk-top`.
- `HELD`: tam giu box cu khi mat detector miss trong vai frame.
- `CANDIDATE`: box moi/chua du on dinh, chi hien trong debug va chua ve len thermal.
- `NO FACE`: chua co box hop le.

Viewer ho tro nhieu nguoi trong cung khung hinh. Moi nguoi duoc gan nhan `P1`,
`P2`, ... va co box/nhiet do ROI rieng. So nguoi hien thi toi da mac dinh la 5,
co the doi bang `--max-faces`.

De giam nhieu ROI, `HEAD TRACK` mac dinh dang tat (`--head-fallback off`).
Fallback nay de bat mat nghieng/anh xam kho detect, nhung de nhiem hon `FACE`
va `PROFILE`, nen chi nen bat lai bang `--head-fallback auto` khi can. Box head
phai du on dinh qua `--head-confirm-frames` truoc khi hien tren thermal view.

Voi nguon `sdk-top`, anh digital co the la grayscale/IR-like, khong phai RGB
webcam binh thuong. Mat nghieng, deo kinh den, bi cat mat hoac qua toi/sang deu
co the lam detector mat chinh dien miss, nen fallback `PROFILE` va `HEAD TRACK`
duoc dung de tracking dau on dinh hon.

```powershell
py .\tc001_thermal_viewer_v5.py --sdk-raw --face-detect --digital-source opencv --digital-backend msmf --digital-device 1 --digital-split right --show-digital-debug
```

Neu chua co file calibration, box mat van co the hien thi bang mapping tam thoi.
De map chinh xac hon, hay calibration 4 diem:

```powershell
py .\tc001_thermal_viewer_v5.py --sdk-raw --calibrate-alignment
```

Sau khi tao `tc001_alignment.json`, chay lai:

```powershell
py .\tc001_thermal_viewer_v5.py --sdk-raw --face-detect
```

Face detection mac dinh khong mo webcam `camera0`. Neu dung fallback
`--digital-source opencv` va truyen `--digital-device 0`, chuong trinh se tu choi
de tranh dung nham webcam laptop.

Neu khong thay face detection:

- Nen bat debug de xem dung camera digital:

```powershell
py .\tc001_thermal_viewer_v5.py --sdk-raw --face-detect --show-digital-debug
```

Gioi han toi da 2 nguoi neu muon giao dien gon hon:

```powershell
py .\tc001_thermal_viewer_v5.py --sdk-raw --face-detect --show-digital-debug --max-faces 2
```

Lenh khuyen nghi khi can giam nhieu box ROI:

```powershell
py .\tc001_thermal_viewer_v5.py --sdk-raw --face-detect --show-digital-debug --max-faces 2 --head-fallback off --cascade-fallback auto --min-face-hits 2 --max-face-area-ratio 0.20 --debug-detections
```

Luu log JSONL de xem data bbox/ROI/nhiet do tra ra:

```powershell
py .\tc001_thermal_viewer_v5.py --sdk-raw --face-detect --show-digital-debug --max-faces 2 --head-fallback off --debug-detections --save-detection-debug .\logs\detections\detections_debug.jsonl
```

Doc report tu file JSONL:

```powershell
py .\tools\analyze_detections.py .\logs\detections\detections_debug.jsonl
```

Neu can thu bat mat nghieng bang fallback head:

```powershell
py .\tc001_thermal_viewer_v5.py --sdk-raw --face-detect --show-digital-debug --max-faces 2 --head-fallback auto --head-confirm-frames 3
```

Neu co file MediaPipe Tasks face detector `.tflite`, co the dung backend AI moi:

```powershell
py .\tc001_thermal_viewer_v5.py --sdk-raw --face-detect --face-model tasks --face-task-model .\models\face_detector.tflite --show-digital-debug --max-faces 2
```

- Neu debug khong hien anh digital that voi `sdk-top`, thu fallback OpenCV/MSMF.
- Neu debug co anh mat nhung hien `NO FACE`, detector dang khong nhan mat trong
  frame do. Hay kiem tra huong xoay `--digital-rotate`, anh co bi cat dau/mat
  khong, va thu nhin thang camera de xac nhan nhanh. Sau ban moi, mat nghieng
  co the hien `PROFILE` hoac `HEAD TRACK` thay vi `NO FACE`.
- Neu co bbox tren debug nhung bbox tren thermal bi lech, can chay calibration
  4 diem de tao `tc001_alignment.json`.

## Cac lenh debug huu ich

Liet ke camera OpenCV:

```powershell
py .\tc001_thermal_viewer_v5.py --list
```

Mo camera OpenCV index 1 de xem visual stream:

```powershell
py .\tc001_thermal_viewer_v5.py --device 1
```

Thu scan raw/radiometric qua OpenCV:

```powershell
py .\tc001_thermal_viewer_v5.py --find-raw --scan-max 5
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
tc001_thermal_viewer_v5.py       Entry point, UI/render/HUD/calibration flow
tc001_sdk.py                     TOPDON SDK/libiruvc radiometric reader
tc001_face.py                    Face detector va face tracking
tc001_alignment.json             Local calibration file, khong commit vao repo
tools/probes/                    Script probe SDK/libusb/TopView
tools/experiments/               Script thu nghiem raw frame
tests/                           Unit tests cho face tracking/ROI
docs/                            Ghi chu va lenh chay bo sung
logs/detections/                 Log JSONL debug face/ROI/detection
archive/legacy_viewers/          Cac ban viewer cu
captures/                        Anh/raw frame mau de debug
downloads/                       File tai ve phuc vu nghien cuu
```

## Huong phat trien AI tiep theo

Huong thiet ke du kien la tach thanh hai luong:

- TC001 Plus SDK doc ma tran nhiet that `256x192`.
- OpenCV `cam1` doc RGB/visual stream cho AI detect.

Trang thai hien tai:

- `--face-detect`: detect mat tren digital stream cua TC001 Plus.
- `--show-digital-debug`: mo cua so debug rieng cho camera digital va bbox mat.
- `--digital-device 1`: mac dinh, khong dung webcam camera0.
- `--digital-split right`: lay nua phai cua frame ghep lam anh digital.
- `--max-faces 5`: so nguoi/face tracks toi da hien thi.
- `--head-fallback off`: mac dinh tat fallback head de giam nhieu ROI.
- `--head-confirm-frames 2`: so frame detect can co truoc khi `HEAD TRACK`
  duoc ve len thermal view.
- `--cascade-fallback auto`: Haar/profile chi la fallback, khong phai model tin
  cay cao tren anh `sdk-top`.
- `--min-face-hits 2`: Haar/profile can lap lai nhieu frame truoc khi ve thermal.
- `--max-face-area-ratio 0.20`: reject box qua lon de tranh ROI phu vai/nguc/nen.
- `--max-box-overlap 0.30`: merge/suppress box moi neu trung track cu.
- `--debug-detections`: in data bbox/ROI/nhiet do ra console.
- `--save-detection-debug .\logs\detections\detections_debug.jsonl`: luu data audit dang JSONL.
- `--face-model tasks --face-task-model ...`: dung MediaPipe Tasks face detector
  neu da co model `.tflite`.
- Label `P1 FACE xx% | max yy.y degC`: hien thi nhiet do cao nhat trong ROI
  tung nguoi da map sang ma tran nhiet. Label co the la `FACE`, `PROFILE`,
  `HEAD TRACK` hoac `HELD` tuy detector dang dung.
- `--calibrate-alignment`: click 4 diem digital/thermal de luu homography.
- `--alignment-file tc001_alignment.json`: file map digital -> thermal.
- `--alignment simple-scale`: mapping tam thoi khi chua calibration.

AI co the dung:

- MediaPipe Tasks Face Detector de detect nhieu mat on dinh hon khi co model
  `.tflite`.
- MediaPipe Face Landmarker de detect mat va vung tran.
-  11n/YOLO11s de detect vat the co rui ro nhu o dien, adapter, day dien,
  pin, o cam keo dai.

Vung detect tren RGB can duoc map sang ma tran nhiet bang calibration 4 diem thu
cong, de lay nhiet do that tu TC001 Plus thay vi do tu anh mau.

