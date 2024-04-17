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

TODO: isi waktu
| Matrix n-size | Serial   | CUDA    |
|---------------|----------|---------|
| 32            | 0.011s   | 0.005s  |
| 64            | 0.014s   | 0.0014s |
| 128           | 0.047s   | 0.0026s |
| 256           | 0.246s   | 0.079s  |
| 512           | 1.636s   | 0.384s  |
| 1024          | 12.089s  | 2.952s  |
| 2048          | 106.112s | 23.58s  |