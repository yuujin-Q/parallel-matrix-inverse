# Tugas Kecil 2 - OpenMP

Tugas Kecil 2 IF3230 Sistem Paralel dan Terdistribusi

## Cara Kerja Paralelisasi Program
Pada perhitungan inverse menggunakan metode Gauss-Jordan, tiap baris akan menjadi pivot mulai dari baris paling atas. Pada tiap pemilihan pivot, tiap baris lainnya akan melakukan perhitungan untuk membuat komponen selain pivot menjadi 0. Cara kerja paralelisasi program yaitu dengan **membuat perhitungan tiap baris terhadap pivot menjadi paralel menggunakan thread.** Pembangkitan thread dilakukan secara otomatis melalui *pragma directive* OpenMP. Berikut adalah langkah detail yang digunakan pada program:
1. Program menyiapkan sebuah buffer matriks yang akan diakses bersama oleh thread OpenMP. Kondisi awal buffer adalah matriks augmentasi dari hasil pembacaan matriks masukan.
2. Program membangkitkan thread sejumlah `thread_count` yang disediakan pengguna melalui argumen pemanggilan program.
3. Program memulai iterasi untuk mereduksi sisi matriks augmentasi menjadi matriks diagonal. Pada setiap pergantian baris pivot, program mengalokasikan (secara statik) sekumpulan iterasi loop yang harus dikerjakan oleh masing-masing thread. Masing-masing thread kemudian mengurangi baris iterasi masing-masing dengan baris pivot saat itu. Operasi ini memodifikasi buffer secara langsung (*in-place*).
4. Setelah melakukan reduksi matriks diagonal, program akan menjalankan proses reduksi matriks identitas pada sisi kiri matriks augmentasi. Operasi reduksi ini dibagi secara rata berdasarkan jumlah thread dan iterasi (yaitu sejumlah baris pada matriks). Masing-masing thread melaksanakan operasi reduksi secara *in-place*.
5. Hasil akhir operasi invers matriks adalah sisi kanan matriks augmentasi yang tersimpan pada buffer. Karena perhitungan dilakukan secara *in-place* pada buffer yang sama, hasil akhir tidak memerlukan proses kombinasi solusi.

## Penjelasan Penggunaan Directive Pragma OMP
Berikut adalah *directive pragma* OMP yang digunakan dalam implementasi solusi.
* ``parallel num_threads(thread_count)``: `parallel` menginstruksikan compiler untuk melakukan paralelisasi pada sebuah blok kode yang dituju. Argumen `num_threads` digunakan untuk memspesifikasi jumlah thread yang digunakan pada blok kode tersebut.
* ``for schedule(static)``: `for` menginstruksikan compiler untuk membagi iterasi yang dilakukan dalam sebuah blok loop for secara rata kepada jumlah thread yang digunakan. Argumen `schedule(static)` menginstruksikan compiler untuk melakukan alokasi pembagian iterasi secara statik (round-robin).

## Alasan Pemilihan Skema Pembagian Data
- Efisiensi Memori dan Akses Data: Karena semua thread berbagi akses ke memori yang sama (shared memory), penggunaan memori menjadi efisien. 
- Pengurangan Overhead: Dengan membagi matriks ke dalam baris-baris yang dihandle oleh berbagai thread, komunikasi antar-thread dapat diminimalkan. Setiap thread melakukan operasi secara independen pada bagian matriks yang ditugaskan, yang mengurangi kebutuhan untuk sinkronisasi.
- Penjadwalan Static: Pemilihan penjadwalan static dalam OpenMP di sini dimaksudkan untuk memberikan kepastian tentang bagian kerja yang ditangani oleh setiap thread. Ini mengurangi latensi yang berasal dari pembagian kerja runtime dan menjadikan eksekusi lebih cepat.

## Cara Menjalankan

Untuk melakukan build dari kode serial dan paralel tuliskan command berikut pada root folder
```
$ make
```
Untuk melakukan run dan menyimpan output kode serial untuk test case `32.txt`
```
$ cat ./test_cases/32.txt | ./bin/serial > serial32.txt
```
Untuk melakukan run dan menyimpan output kode paralel untuk test case `32.txt`
```
$ ./bin/mp `<thread_count>` < 32.txt > paralel32.txt
```


## Perbandingan Waktu
Compiled in: Win11 - WSL2 Ubuntu 22.04 - AMD Ryzen 5 5600U
| Matrix n-size | Serial   | OpenMP - 4 Threads | OpenMPI - 8 Threads |
|---------------|----------|--------------------|---------------------|
| 32            | 0.011s   |        0.005s      |        0.020s       |
| 64            | 0.014s   |        0.0014s     |        0.022s       |
| 128           | 0.047s   |        0.0026s     |        0.043s       |
| 256           | 0.246s   |        0.079s      |        0.078s       |
| 512           | 1.636s   |        0.384s      |        0.330s       |
| 1024          | 12.089s  |        2.952s      |        2.013s       |
| 2048          | 106.112s |        23.58s      |        19.076s      |