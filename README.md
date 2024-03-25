# Tugas Kecil - Paralel Inverse Matrix

Template Tugas Kecil IF3230 Sistem Paralel dan Terdistribusi

## How to Run

### Serial

Contoh build, run, dan menyimpan output untuk test case `32.txt`.

```console
user@user:~/kit-tucil-sister-2024$ make
user@user:~/kit-tucil-sister-2024$ cat ./test_cases/32.txt | ./bin/serial > 32.txt
```

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