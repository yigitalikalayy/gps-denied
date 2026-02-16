# Raspberry Pi 5 Optical Flow (PX4Flow-compatible)

Bu klasör PX4Flow Firmware reposuna, Raspberry Pi 5 üzerinde çalışan ve CUAV V5+ (PX4/ArduPilot) otopilota **MAVLink `OPTICAL_FLOW_RAD`** gönderen bir uygulama ekler.

Hedef donanım:
- Raspberry Pi 5
- OV9281 global shutter kamera (CSI / Cam0)
- Lightware LW20b lidar (I2C)
- Otopilot: CUAV V5 Plus (UART4, `921600`)

## Çalıştırma

1) (Raspberry Pi üzerinde) bağımlılıklar:
- Python 3.10+
- `python3-opencv`, `python3-numpy`, `python3-serial`
- Kamera için önerilen: `python3-picamera2` (libcamera)
- I2C için: `python3-smbus` veya `python3-smbus2`

2) I2C ve seri port izinleri:
- I2C: `sudo raspi-config` → Interfaces → I2C enable
- UART: ilgili `/dev/tty*` cihazına erişim (örn. `dialout` grubu)

3) Konfigürasyon:
- `config.json` içindeki `serial.port`, `camera.*`, `flow.*`, `lidar.*` alanlarını kendi sistemine göre ayarla.
- `serial.request_imu_stream.message_ids` içinde `30`/`31` (ATTITUDE/ATTITUDE_QUATERNION) bulunduğundan emin ol.
- Hareket-benzerliğine dayalı zaman senkronizasyonu için `time_sync.enabled=true` kullan.
- `time_sync.nominal_fps_source="runtime"` ile nominal FPS başlangıcı gerçek çalışma FPS’inden bootstrap edilir (`bootstrap_frames`, `bootstrap_timeout_s`).

## Kamera kalibrasyonu (Zhang)

`OV9281` için pinhole intrinsics (`K`) ve distorsiyon (`k1,k2,p1,p2,k3`) hesabı `camera.calibration` ile yapılır.

1) Satranç tahtası görüntülerini aynı çözünürlükte topla (örn. `rpi_flow/calib/*.png`).
2) `config.json` içinde:
- `camera.calibration.enabled=true`
- `camera.calibration.mode="zhang"`
- `camera.calibration.images_glob="calib/*.png"`
- `camera.calibration.cache_file="calib/calibration_ov9281.json"`
3) Uygulama ilk açılışta kalibrasyonu hesaplar, cache’e yazar ve tüm frame’leri undistort eder.

Alternatif olarak `mode="precomputed"` ile `camera_matrix` ve `dist_coeffs` doğrudan verilebilir.

4) Çalıştır:
```bash
cd rpi_flow
PYTHONPATH=./src python3 -m px4flow_rpi.main --config ./config.json
```

Alternatif:
```bash
cd rpi_flow
python3 ./run.py --config ./config.json
```

## Terminal çıktıları

Uygulama stdout’a şu metrikleri basar:
- frame sayısı ve kamera FPS
- optik flow quality (0..255)
- per-frame pixel displacement ve gyro (rad/s)
- lidar distance (m)
- `time_sync` sağlık durumu: `status=warming_up/healthy/degraded/stale`, `score`, `reason`, `lag_ms`, `corr`

Not: En pratik kullanım `px4flow_rpi.sh` çalıştırmaktır; bu şekilde terminalde çıktıları sürekli izlersin:
```bash
cd rpi_flow
./px4flow_rpi.sh ./config.json
```

## PC üzerinde test

Evet, bilgisayarda da test edip terminal çıktısını görebilirsin. Bunun için MAVLink seri çıkışını ve lidar’ı kapat:
- `config.json` → `serial.enabled=false`
- `config.json` → `lidar.enabled=false`
- `config.json` → `camera.backend="opencv"` (PC webcam için)

Sonra:
```bash
cd rpi_flow
PYTHONPATH=./src python3 -m px4flow_rpi.main --config ./config.json
```

## Sistem servisi

`px4flow_rpi.sh` kamera portu “in use” durumlarını azaltmak için önce kamera ile ilgili açık süreçleri kapatır, sonra uygulamayı başlatır.

Örnek systemd servisi: `systemd/px4flow-rpi.service`
