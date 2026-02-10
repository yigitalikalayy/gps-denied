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
