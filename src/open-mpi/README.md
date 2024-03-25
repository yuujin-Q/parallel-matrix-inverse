# Tugas Kecil 1 - OpenMPI

Tugas Kecil 1 IF3230 Sistem Paralel dan Terdistribusi

## Cara Kerja Paralelisasi Program
Pada perhitungan inverse menggunakan metode Gauss-Jordan, tiap baris akan menjadi pivot mulai dari baris paling atas. Pada tiap pemilihan pivot, tiap baris lainnya akan melakukan perhitungan untuk membuat komponen selain pivot menjadi 0. Cara kerja paralelisasi program yaitu degan **membuat perhitungan tiap baris terhadap pivot menjadi paralel.** Berikut adalah langkah detail yang digunakan pada program:
1. Program membagi baris-baris matriks sebanyak proses yang digunakan
2. Masing-masing proses menghitung ``local_start_row`` dan ``local_end_row`` berdasarkan ``rank`` dan ``comm_sz``
3. Program melakukan iterasi baris dimulai dari baris yang paling atas
4. Ketika iterasi baris sampai di row yang dipegang oleh suatu proses, proses tersebut akan melakukan MPI_Bcast row yang dipegang
5. Masing-masing proses melakukan perhitungan berdasarkan baris pivot
6. Setelah iteasi baris mencapai baris paling akhir, dilakukan MPI_Gather untuk matriks hasil

## Cara Program Membagikan Data Antar-Proses
* ``MPI_Bcast`` digunakan pada variabel matriks yang dibaca dari masukan. Hal ini dikarenakan setiap proses membutuhkan data matriks tersebut, sedangkan pembacaan matriks hanya dilakukan pada proses 0.
* ``MPI_Bcast`` digunakan juga proses untuk broadcast baris saat baris tersebut menjadi pivot row. Hal ini dikarenakan setiap proses membutuhkan akses terhadap pivot row sedangkan nilai suatu pivot row yang paling terbaru hanya dimiliki oleh satu proses.
* ``MPI_Gather`` digunakan untuk menyatukan hasil matriks. Hal ini dikarenakan baris-baris matriks yang terbagi di beberapa proses perlu dijadikan satu proses di proses 0 untuk keperluan output.


## How to Run

Contoh build, run, dan menyimpan output untuk test case `32.txt`.

```console
user@user:~/kit-tucil-sister-2024$ make
user@user:~/kit-tucil-sister-2024$ cat ./test_cases/32.txt | ./bin/serial > 32.txt
```

## Perbandingan Waktu
Compiled in: Win11 - WSL2 Ubuntu 22.04 - AMD Ryzen 7 5700U
| Matrix n-size | Serial   | OpenMPI - 4 Core | OpenMPI - 8 Core |
|---------------|----------|------------------|------------------|
| 32            | 0.01s    |        0.0002s   | 0.0003s          |
| 64            | 0.018s   |        0.0008s   | 0.0008s          |
| 128           | 0.097s   |        0.0039s   | 0.0044s          |
| 256           | 0.440s   |        0.0369s   | 0.0303s          |
| 512           | 2.678s   |        0.338s    | 0.235s           |
| 1024          | 18.111s  |        2.95s     | 2.599s           |
| 2048          | 124.010s |        21.99s    | 21.36s           |