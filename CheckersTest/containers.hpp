#include <cstddef> 
#include "cuda_runtime.h"

#define CAPACITY 64

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
