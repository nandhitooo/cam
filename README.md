# Webcam Filters 🎥

Aplikasi filter webcam real-time dengan Python + OpenCV. Ambil frame dari kamera,
terapkan 17 filter artistik, jalankan **face tracker** dengan efek wajah, lalu
simpan snapshot.

## Fitur

- **17 filter** real-time (grayscale, sepia, kartun, sketsa pensil, termal, glitch, dll.)
- **Face tracker** berbasis YuNet (`cv2.FaceDetectorYN`): kotak + label + 5 landmark per wajah
- **4 efek wajah**: kotak deteksi, privasi blur, privasi pixelate, kacamata hitam virtual
- **Face auto-zoom (face cam)**: frame dipotong dan diperbesar mengikuti wajah
- **Kamera dibaca di thread terpisah**: filter berat tidak membuat preview tertinggal dari kenyataan
- Ganti filter dengan tombol angka atau panah kiri/kanan tanpa restart
- HUD: nama filter aktif, FPS, jumlah wajah, mode wajah, daftar filter, status mirror
- Snapshot PNG dengan tombol `s`, nama file otomatis memakai nama filter + timestamp
- Preview mirror (seperti cermin) dan mode fullscreen
- `--self-test`: memverifikasi filter, geometri/efek wajah, logika tracker, dan pembaca
  kamera **tanpa kamera dan tanpa wajah asli**

## Instalasi

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Model wajah sudah disertakan di `models/face_detection_yunet_2023mar.onnx` (232 KB),
jadi fitur wajah langsung jalan tanpa internet. Kalau file itu hilang, aplikasi
tetap berjalan, hanya mematikan fitur wajah sambil memberi tahu cara mengunduhnya.

## Menjalankan

```bash
python webcam_filters.py                         # kamera default, 1280x720
python webcam_filters.py --camera 1              # pilih perangkat kamera lain
python webcam_filters.py --width 640 --height 480
python webcam_filters.py --filter cartoon        # mulai dengan filter tertentu
python webcam_filters.py --face-effect glasses   # mulai dengan kacamata hitam
python webcam_filters.py --face-zoom             # mulai dalam mode face cam
python webcam_filters.py --face-effect blur --filter normal
python webcam_filters.py --list-filters
python webcam_filters.py --self-test             # uji semuanya tanpa kamera
```

Snapshot disimpan di folder `snapshots/` (bisa diubah dengan `--snapshot-dir`).

## Kontrol keyboard

| Tombol | Fungsi |
| --- | --- |
| `1`–`9`, `0` | Pilih filter 1–10 |
| `n` atau `→` | Filter berikutnya |
| `p` atau `←` | Filter sebelumnya |
| `v` | Ganti mode wajah: off → box → blur → pixelate → glasses |
| `z` | Nyalakan/matikan auto-zoom ke wajah |
| `s` | Simpan snapshot PNG |
| `m` | Nyalakan/matikan mirror |
| `h` | Sembunyikan/tampilkan HUD |
| `f` | Toggle fullscreen |
| `q` atau `Esc` | Keluar |

## Filter

| # | Slug | Filter | Efek |
| --- | --- | --- | --- |
| 1 | `normal` | Normal | Tanpa efek |
| 2 | `grayscale` | Grayscale | Hitam putih |
| 3 | `sepia` | Sepia | Ton cokelat klasik |
| 4 | `invert` | Invert | Negatif warna |
| 5 | `blur` | Blur | Buram (Gaussian) |
| 6 | `sharpen` | Sharpen | Pertegas detail |
| 7 | `edges` | Edge | Garis tepi (Canny) |
| 8 | `cartoon` | Cartoon | Warna rata + garis tinta |
| 9 | `sketch` | Sketch | Sketsa pensil |
| 10 | `warm` | Warm | Nuansa hangat |
| 11 | `cool` | Cool | Nuansa dingin |
| 12 | `vintage` | Vintage | Sepia pudar + vignette |
| 13 | `noir` | Noir | Hitam putih kontras tinggi |
| 14 | `dream` | Dream | Glow lembut |
| 15 | `pixelate` | Pixelate | Blok piksel |
| 16 | `thermal` | Thermal | Kamera termal (colormap) |
| 17 | `glitch` | Glitch | Pergeseran kanal + scanline |

## Mode wajah

| Mode | Efek |
| --- | --- |
| `off` | Face tracker tidak dijalankan sama sekali (tidak ada biaya) |
| `box` | Kotak deteksi, id, skor keyakinan, dan titik landmark |
| `blur` | Wajah diburamkan, sisa frame tetap tajam (privasi) |
| `pixelate` | Wajah dijadikan blok piksel, sisa frame tetap tajam (privasi) |
| `glasses` | Kacamata hitam digambar mengikuti 5 landmark, termasuk saat kepala miring |

Mode wajah bisa digabung dengan filter dasar apa pun, dan auto-zoom (`z`) juga
bisa digabung dengan semuanya.

### Bagaimana face tracker bekerja

Deteksi YuNet memakan ~17 ms per frame pada frame 720p, jadi menjalankannya tiap
frame akan memakan seluruh anggaran real-time. Karena itu `FaceTracker`:

- menjalankan detektor hanya tiap **N frame** (`--face-detect-every`, default 3),
- menahan kotak di antara deteksi, jadi overlay tidak berkedip,
- menghaluskan kotak dan landmark dengan EMA agar tidak gemetar,
- mencocokkan deteksi ke track memakai IoU lalu memberi **id stabil**, sehingga
  wajah yang sempat menoleh tidak langsung kehilangan id-nya,
- pensiun setelah `max_missed` deteksi gagal berturut-turut.

Deteksi dijalankan pada frame yang diperkecil ke 320 px (`--face-detect-width`);
kotak dan landmark-nya diskalakan kembali ke resolusi penuh.

## Performa

Biaya per frame pada 1280x720 (diukur tanpa kamera, hanya komputasi). `+ box`,
`+ blur`, dan `+ glasses` termasuk tracker (`detect_every=3`) dan HUD:

| Filter dasar | + box | + blur | + glasses |
| --- | --- | --- | --- |
| normal | 19 ms (53 fps) | 15 ms (67 fps) | 14 ms (70 fps) |
| sepia | 21 ms (48 fps) | 21 ms (48 fps) | 19 ms (52 fps) |
| vintage | 30 ms (33 fps) | 22 ms (45 fps) | 21 ms (47 fps) |
| sketch | 41 ms (25 fps) | 25 ms (41 fps) | 25 ms (40 fps) |
| cartoon | 54 ms (19 fps) | 58 ms (17 fps) | 51 ms (20 fps) |

Biaya face tracker ter-amortisasi pada 720p: **17 ms/frame** kalau deteksi tiap
frame, **9 ms** tiap 2 frame, **6 ms** tiap 3 frame (default), **2 ms** tiap 5 frame.
Saat fitur wajah mati, detektor tidak dijalankan sama sekali.

Filter terberat tetap `cartoon` (~32 ms). Kalau perangkat kamu kurang cepat,
turunkan resolusi dan/atau jarakkan deteksi:

```bash
python webcam_filters.py --width 640 --height 480 --face-detect-every 5
```

## Catatan teknis

- Frame diproses sebagai array BGR `uint8` dengan NumPy; filter yang berat
  (cartoon, dream) menghitung efek pada resolusi lebih kecil lalu di-upscale.
- `warm`/`cool` memakai lookup table per kanal, dan `vintage` memakai perkalian
  integer agar tetap cepat di frame besar.
- Kartun di-posterize setelah upscale supaya blok warnanya tidak kembali menjadi
  gradien halus.
- Sketsa memakai color-dodge lalu meregangkan persentil 2–98 agar garis pensilnya
  terlihat (tanpa ini hasilnya nyaris putih seluruhnya).
- Privasi blur bekerja dengan menurunkan resolusi wajah ke ~30 px lalu
  meng-upscale-nya: hasilnya blur yang jauh lebih kuat daripada kernel Gaussian
  besar di resolusi penuh, dengan biaya jauh lebih kecil.
- Efek wajah digambar sebelum auto-zoom, jadi keduanya memakai satu sistem koordinat.
- Pembaca kamera berjalan di thread terpisah. Driver terus dikuras, jadi hanya
  frame terbaru yang disimpan: saat filter berat, frame lama dibuang alih-alih
  ditampilkan terlambat. `read()` lalu menunggu frame **berikutnya**, sehingga
  aplikasi tidak merender ulang frame yang sama (yang akan membuang satu core
  penuh dan membuat angka FPS di HUD tidak berarti). Jumlah frame yang dibuang
  dilaporkan saat keluar.
- Tanpa `--no-thread`, kamera dibaca inline (berguna untuk debugging).

## Troubleshooting

**"could not open camera index 0"**
Kamera sedang dipakai aplikasi lain, atau indeksnya berbeda — coba `--camera 1`.
Di WSL/container, perangkat kamera harus di-passthrough ke dalam environment.

**"warning: face features are disabled"**
File model wajah tidak ditemukan. Unduh ke path yang dilaporkan, atau tunjuk
lokasi lain dengan `--face-model`. Sisa aplikasi tetap berjalan tanpa fitur wajah.

**Wajah lambat terdeteksi atau kotaknya tertinggal**
Turunkan `--face-detect-every` ke 2 (atau 1) agar deteksi lebih sering, dengan
catatan biaya per frame naik.

**Jendela tidak muncul**
Aplikasi butuh display grafis. Di server tanpa GUI, gunakan `--self-test`.

**Preview patah-patah**
Pakai resolusi lebih kecil (`--width 640 --height 480`), matikan mode wajah
dengan `v`, dan tutup aplikasi lain yang memakai kamera. FPS di HUD adalah laju
nyata frame yang ditampilkan, sedangkan angka "dropped" saat keluar menunjukkan
berapa frame kamera yang dibuang karena aplikasi belum selesai memprosesnya.

## Sumber model

`models/face_detection_yunet_2023mar.onnx` berasal dari
[opencv_zoo](https://github.com/opencv/opencv_zoo) (lisensi Apache-2.0), model
deteksi wajah YuNet. Integritas file bisa dicek dengan SHA-256:

```
8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4
```
