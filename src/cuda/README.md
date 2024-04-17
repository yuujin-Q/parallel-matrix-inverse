# Tugas Kecil 3 - CUDA Parallelism

Tugas Kecil 3 IF3230 Sistem Paralel dan Terdistribusi

## Cara Kerja Paralelisasi Program
Program ini mengimplementasikan algoritma inversi matriks menggunakan CUDA untuk paralelisasi komputasi. Matriks augmentasi dibuat dengan menambahkan matriks identitas pada matriks asli, yang memungkinkan operasi baris elementer untuk menghasilkan invers pada bagian akhir matriks. Paralelisasi dilakukan dengan mendistribusikan operasi pada baris matriks ke thread dalam blok CUDA:

1. Program menyiapkan 2 buffer matriks yang akan digunakan. Matriks `mat` adalah matriks augmentasi dari hasil pembacaan matriks masukan. Kemudian matriks tersebut akan diduplikasi isinya ke dalam matriks `d_mat` yaitu sebuah matriks di GPU.
2. Program lalu menginisialiasi `dimGrid` dan `dimBlock`. `dimGrid` yang digunakan berukuran 1x1 dan `dimBlock` yang digunakan bergantung ukuran matriks.
3. Program lalu memulai iterasi pivot berdasarkan jumlah baris. Setiap iterasi, program melakukan normalisasi baris pivot dan pengurangan baris non-pivot.
4. Baris pivot dinormalisasi sehingga elemen diagonal menjadi satu, dan elemen lainnya di baris tersebut disesuaikan. Setiap elemen di baris pivot diupdate oleh sebuah thread. Kernel yang bertanggung jawab memodifikasi baris menjalankan kernel internal untuk membagi operasi modifikasi untuk setiap kolom.
5. Setiap baris non-pivot diupdate dengan mengurangkan kelipatannya dengan baris pivot untuk membuat kolom di bawah elemen pivot menjadi nol. Operasi pada setiap baris non-pivot dijalankan oleh thread secara paralel. Kernel yang bertanggung jawab memodifikasi baris menjalankan kernel internal untuk membagi operasi modifikasi untuk setiap kolom.
6. Setelah iterasi selesai, hasil akhir matriks `d_mat` di GPU diduplikasi kembali ke `mat` di CPU.
7. Hasil akhir operasi invers matriks adalah sisi kanan matriks augmentasi yang tersimpan pada buffer. 

## Proses Pembagian Data
- Memori Shared: Setiap iterasi modifkasi baris, setiap block dalam kernel menyimpan hasil modifikasi baris dalam shared memori. Hal ini dilakukan untuk sinkronisasi proses baca-tulis terhadap matriks pada memori global dalam sebuah block.
- Memori Global: Seluruh matriks disimpan dalam memori global GPU, di mana setiap thread dapat mengakses data yang diperlukan. Akses global diperlukan untuk memungkinkan pengaksesan data antar baris yang diolah oleh blok berbeda.

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
$ ./bin/cuda < 32.txt > paralel32.txt
```

Kode parallel CUDA juga dapat dijalankan melalui `cuda_colab.ipynb`


## Perbandingan Waktu
Test environment: Google Colab
- Python 3 Google Compute Engine backend (GPU)
- nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Tue_Aug_15_22:02:13_PDT_2023
Cuda compilation tools, release 12.2, V12.2.140
Build cuda_12.2.r12.2/compiler.33191640_0

| Matrix n-size | Serial    | CUDA      |
|---------------|-----------|-----------|
| 32            | 0m0.357s  | 0m0.478s  |
| 64            | 0m0.529s  | 0m0.305s  |
| 128           | 0m0.615s  | 0m0.443s  |
| 256           | 0m1.279s  | 0m0.722s  |
| 512           | 0m2.335s  | 0m2.502s  |
| 1024          | 0m20.865s | 0m7.911s  |
| 2048          | 2m34.759s | 0m24.686s |