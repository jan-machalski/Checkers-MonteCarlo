#include <cstddef> 
#include "cuda_runtime.h"

// Template class for CUDA-compatible Vector
template <typename T>
class CUDA_Vector {
private:
    T* data;
    size_t capacity;
    size_t length;

public:
    // Constructor with optional initial capacity
    __host__ __device__ CUDA_Vector(size_t initialCapacity = 16)
        : capacity(initialCapacity), length(0) {
        data = new T[capacity];
    }

    // Destructor
    __host__ __device__ ~CUDA_Vector() {
        delete[] data;
    }

    // Copy constructor
    __host__ __device__ CUDA_Vector(const CUDA_Vector& other)
        : capacity(other.capacity), length(other.length) {
        data = new T[capacity];
        for (size_t i = 0; i < length; ++i) {
            data[i] = other.data[i];
        }
    }

    // Copy assignment operator
    __host__ __device__ CUDA_Vector& operator=(const CUDA_Vector& other) {
        if (this != &other) {
            delete[] data; // Zwolnij poprzednie zasoby
            capacity = other.capacity;
            length = other.length;
            data = new T[capacity];
            for (size_t i = 0; i < length; ++i) {
                data[i] = other.data[i];
            }
        }
        return *this;
    }

    // Move constructor
    __host__ __device__ CUDA_Vector(CUDA_Vector&& other) noexcept
        : data(other.data), capacity(other.capacity), length(other.length) {
        other.data = nullptr;
        other.capacity = 0;
        other.length = 0;
    }

    // Move assignment operator
    __host__ __device__ CUDA_Vector& operator=(CUDA_Vector&& other) noexcept {
        if (this != &other) {
            delete[] data; // Zwolnij poprzednie zasoby
            data = other.data;
            capacity = other.capacity;
            length = other.length;

            other.data = nullptr;
            other.capacity = 0;
            other.length = 0;
        }
        return *this;
    }

    // Add an element to the end
    __host__ __device__ void push_back(const T& value) {
        if (length >= capacity) {
            resize(capacity * 2);
        }
        data[length++] = value;
    }

    // Access element at index
    __host__ __device__ T& at(size_t index) {
        return data[index];
    }

    // Access element at index (const)
    __host__ __device__ const T& at(size_t index) const {
        return data[index];
    }

    // Get the current size
    __host__ __device__ size_t size() const {
        return length;
    }

    // Overloaded operator[] for accessing elements
    __host__ __device__ T& operator[](size_t index) {
        return data[index];
    }

    // Overloaded operator[] for accessing elements (const)
    __host__ __device__ const T& operator[](size_t index) const {
        return data[index];
    }

    // Begin iterator
    __host__ __device__ T* begin() {
        return data;
    }

    // Begin iterator (const)
    __host__ __device__ const T* begin() const {
        return data;
    }

    // End iterator
    __host__ __device__ T* end() {
        return data + length;
    }

    // End iterator (const)
    __host__ __device__ const T* end() const {
        return data + length;
    }

private:
    // Resize the internal storage
    __host__ __device__ void resize(size_t newCapacity) {
        T* newData = new T[newCapacity];
        for (size_t i = 0; i < length; ++i) {
            newData[i] = data[i];
        }
        delete[] data;
        data = newData;
        capacity = newCapacity;
    }
};


// Template class for CUDA-compatible Queue
template <typename T>
class CUDA_Queue {
private:
    T* data;
    size_t capacity;
    size_t frontIndex;
    size_t backIndex;
    size_t count;

public:
    // Constructor with optional initial capacity
    __host__ __device__ CUDA_Queue(size_t initialCapacity = 16)
        : capacity(initialCapacity), frontIndex(0), backIndex(0), count(0) {
        data = new T[capacity];
    }

    // Destructor
    __host__ __device__ ~CUDA_Queue() {
        delete[] data;
    }

    // Copy constructor
    __host__ __device__ CUDA_Queue(const CUDA_Queue& other)
        : capacity(other.capacity), frontIndex(other.frontIndex), backIndex(other.backIndex), count(other.count) {
        data = new T[capacity];
        for (size_t i = 0; i < count; ++i) {
            data[(frontIndex + i) % capacity] = other.data[(frontIndex + i) % capacity];
        }
    }

    // Copy assignment operator
    __host__ __device__ CUDA_Queue& operator=(const CUDA_Queue& other) {
        if (this != &other) {
            delete[] data; // Release current resources

            capacity = other.capacity;
            frontIndex = other.frontIndex;
            backIndex = other.backIndex;
            count = other.count;

            data = new T[capacity];
            for (size_t i = 0; i < count; ++i) {
                data[(frontIndex + i) % capacity] = other.data[(frontIndex + i) % capacity];
            }
        }
        return *this;
    }

    // Move constructor
    __host__ __device__ CUDA_Queue(CUDA_Queue&& other) noexcept
        : data(other.data), capacity(other.capacity), frontIndex(other.frontIndex), backIndex(other.backIndex), count(other.count) {
        other.data = nullptr;
        other.capacity = 0;
        other.frontIndex = 0;
        other.backIndex = 0;
        other.count = 0;
    }

    // Move assignment operator
    __host__ __device__ CUDA_Queue& operator=(CUDA_Queue&& other) noexcept {
        if (this != &other) {
            delete[] data; // Release current resources

            data = other.data;
            capacity = other.capacity;
            frontIndex = other.frontIndex;
            backIndex = other.backIndex;
            count = other.count;

            other.data = nullptr;
            other.capacity = 0;
            other.frontIndex = 0;
            other.backIndex = 0;
            other.count = 0;
        }
        return *this;
    }

    // Add an element to the queue
    __host__ __device__ void push(const T& value) {
        if (count >= capacity) {
            resize(capacity * 2);
        }
        data[backIndex] = value;
        backIndex = (backIndex + 1) % capacity;
        ++count;
    }

    // Remove the front element from the queue
    __host__ __device__ void pop() {
        if (!empty()) {
            frontIndex = (frontIndex + 1) % capacity;
            --count;
        }
    }

    // Get the front element
    __host__ __device__ T& front() {
        return data[frontIndex];
    }

    // Get the front element (const)
    __host__ __device__ const T& front() const {
        return data[frontIndex];
    }

    // Check if the queue is empty
    __host__ __device__ bool empty() const {
        return count == 0;
    }

private:
    // Resize the internal storage
    __host__ __device__ void resize(size_t newCapacity) {
        T* newData = new T[newCapacity];
        for (size_t i = 0; i < count; ++i) {
            newData[i] = data[(frontIndex + i) % capacity];
        }
        delete[] data;
        data = newData;
        frontIndex = 0;
        backIndex = count;
        capacity = newCapacity;
    }
};
