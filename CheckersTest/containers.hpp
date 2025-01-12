#include <cstddef> 
#include<map>
#include<string>
#include "cuda_runtime.h"

#define CAPACITY 48

// Template class for CUDA-compatible Vector
template <typename T>
class CUDA_Vector {
private:
    T data[CAPACITY]; // Stała maksymalna pojemność
    size_t length; // Aktualna liczba elementów

public:
    // Konstruktor
    __host__ __device__ CUDA_Vector() : length(0) {}

    // Dodanie elementu
    __host__ __device__ void push_back(const T& value) {
        if (length < CAPACITY) {
            data[length++] = value;
        }
        else {
            // Obsługa błędu: przekroczenie pojemności
            printf("Error: CUDA_Vector capacity exceeded.\n");
        }
    }

    // Dostęp do elementu na indeksie
    __host__ __device__ T& at(size_t index) {
        return data[index];
    }

    // Dostęp do elementu na indeksie (const)
    __host__ __device__ const T& at(size_t index) const {
        return data[index];
    }

    // Aktualny rozmiar
    __host__ __device__ size_t size() const {
        return length;
    }

    // Indeksowanie
    __host__ __device__ T& operator[](size_t index) {
        return data[index];
    }

    __host__ __device__ const T& operator[](size_t index) const {
        return data[index];
    }

    // Iteratory
    __host__ __device__ T* begin() { return data; }
    __host__ __device__ const T* begin() const { return data; }
    __host__ __device__ T* end() { return data + length; }
    __host__ __device__ const T* end() const { return data + length; }

    // Wyczyszczenie wektora
    __host__ __device__ void clear() {
        length = 0;
    }
};


template <typename T>
class CUDA_Queue {
private:
    T data[CAPACITY]; // Stała maksymalna pojemność
    size_t frontIndex; // Indeks początku
    size_t backIndex;  // Indeks końca
    size_t count;      // Liczba elementów

public:
    // Konstruktor
    __host__ __device__ CUDA_Queue() : frontIndex(0), backIndex(0), count(0) {}

    // Dodanie elementu
    __host__ __device__ void push(const T& value) {
        if (count < CAPACITY) {
            data[backIndex] = value;
            backIndex = (backIndex + 1) % CAPACITY;
            ++count;
        }
        else {
            // Obsługa błędu: przekroczenie pojemności
            printf("Error: CUDA_Queue capacity exceeded.\n");
        }
    }

    // Usunięcie elementu z początku
    __host__ __device__ void pop() {
        if (!empty()) {
            frontIndex = (frontIndex + 1) % CAPACITY;
            --count;
        }
    }

    // Dostęp do pierwszego elementu
    __host__ __device__ T& front() {
        return data[frontIndex];
    }

    // Dostęp do pierwszego elementu (const)
    __host__ __device__ const T& front() const {
        return data[frontIndex];
    }

    // Sprawdzenie, czy kolejka jest pusta
    __host__ __device__ bool empty() const {
        return count == 0;
    }

    // Aktualny rozmiar kolejki
    __host__ __device__ size_t size() const {
        return count;
    }

    // Wyczyszczenie kolejki
    __host__ __device__ void clear() {
        frontIndex = 0;
        backIndex = 0;
        count = 0;
    }
};

const std::map<uint32_t, std::string> boardMap = {
    {0x00000001,"b8"},
    {0x00000002,"d8"},
    {0x00000004,"f8"},
    {0x00000008,"h8"},
    {0x00000010,"a7"},
    {0x00000020,"c7"},
    {0x00000040,"e7"},
    {0x00000080,"g7"},
    {0x00000100,"b6"},
    {0x00000200,"d6"},
    {0x00000400,"f6"},
    {0x00000800,"h6"},
    {0x00001000,"a5"},
    {0x00002000,"c5"},
    {0x00004000,"e5"},
    {0x00008000,"g5"},
    {0x00010000,"b4"},
    {0x00020000,"d4"},
    {0x00040000,"f4"},
    {0x00080000,"h4"},
    {0x00100000,"a3"},
    {0x00200000,"c3"},
    {0x00400000,"e3"},
    {0x00800000,"g3"},
    {0x01000000,"b2"},
    {0x02000000,"d2"},
    {0x04000000,"f2"},
    {0x08000000,"h2"},
    {0x10000000,"a1"},
    {0x20000000,"c1"},
    {0x40000000,"e1"},
    {0x80000000,"g1"}
};

const std::map<uint32_t, std::string>boardMapReverse = {
    {0x00000001,"g1"},
    {0x00000002,"e1"},
    {0x00000004,"c1"},
    {0x00000008,"a1"},
    {0x00000010,"h2"},
    {0x00000020,"f2"},
    {0x00000040,"d2"},
    {0x00000080,"b2"},
    {0x00000100,"g3"},
    {0x00000200,"e3"},
    {0x00000400,"c3"},
    {0x00000800,"a3"},
    {0x00001000,"h4"},
    {0x00002000,"f4"},
    {0x00004000,"d4"},
    {0x00008000,"b4"},
    {0x00010000,"g5"},
    {0x00020000,"e5"},
    {0x00040000,"c5"},
    {0x00080000,"a5"},
    {0x00100000,"h6"},
    {0x00200000,"f6"},
    {0x00400000,"d6"},
    {0x00800000,"b6"},
    {0x01000000,"h7"},
    {0x02000000,"f7"},
    {0x04000000,"d7"},
    {0x08000000,"b7"},
    {0x10000000,"g8"},
    {0x20000000,"e8"},
    {0x40000000,"c8"},
    {0x80000000,"a8"}
};