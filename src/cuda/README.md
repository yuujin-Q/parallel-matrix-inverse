# Tugas Kecil 3 - CUDA Parallelism

Tugas Kecil 3 IF3230 Sistem Paralel dan Terdistribusi

## Cara Kerja Paralelisasi Program
todo: isi readme

## Pemetaan Program Serial pada Algoritma Paralel CUDA
todo: isi readme (hapus bagian ini jika tidak diperlukan)

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