#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/system_error.h>
#include <thrust/sort.h>

#include <cuda.h>

#include <iostream>
#include <chrono>
#include <cstdlib>

using std::chrono::steady_clock;
using std::chrono::steady_clock;

/////////////////////////////////////////////////////////////////

#define BATCH_SIZE 136
#define VERTICES   8

const int imgsize = 500 * (500 / 2 + 1);

#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

/////////////////////////////////////////////////////////////////

typedef int CRSCoord;

class Voxel {
    public:
        __host__ __device__ __forceinline__
        Voxel(double r = 0, double i = 0, double w = 0) {
            real = r; imag = i; weit = w;
        }

        __host__ __device__ __forceinline__
        Voxel operator+(const Voxel& that) const {
            return Voxel(real + that.real,
                         imag + that.imag,
                         weit + that.weit);
        }

        __host__ __device__ __forceinline__
        Voxel& operator+=(const Voxel& that) {
            real += that.real;
            imag += that.imag;
            weit += that.weit;
            
            return *this;
        }

    public:
        double real;
        double imag;
        double weit;
};

/////////////////////////////////////////////////////////////////

int read_data_stream(CRSCoord *coords, Voxel *voxels)
{
    FILE *fcor = fopen("../../data/stream/coords.dat", "rb");
    FILE *fvxl = fopen("../../data/stream/voxels.dat", "rb");

    const int total_size = BATCH_SIZE * imgsize * VERTICES;

    steady_clock::time_point begin = steady_clock::now();

    if (fcor && fvxl) {
        if (fread(coords,
                  sizeof(CRSCoord),
                  BATCH_SIZE * imgsize * VERTICES,
                  fcor)
            == total_size) {
            fclose(fcor);
        } else {
            std::cout << "coords.dat read error!\n";
            exit(1);
        }

        if (fread(voxels,
                  sizeof(Voxel),
                  BATCH_SIZE * imgsize * VERTICES,
                  fvxl)
            == total_size) {
            fclose(fvxl);
        } else {
            std::cout << "voxels.dat read error!\n";
            exit(1);
        }

    } else {
        std::cout << "file open error!\n";
        exit(2);
    }

    steady_clock::time_point end = steady_clock::now();
    std::cout << "Data reading done! time consumed: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()
              << "ms" << std::endl;

    return total_size;
}

void statistic(CRSCoord *coords, Voxel *voxels, int length)
{
    int segment = 0, counter = 1;

    int current = 0, next;
    while (current < length) {
        next = current < length - 1 ? current + 1 : current;

        if (next != current && coords[next] == coords[current]) {
            counter++;
        } else {
            printf("Segment No.%8d: length %8d\n", segment, counter);

            segment++;
            counter = 1;
        }

        current = next;
    }
}

/////////////////////////////////////////////////////////////////

__global__ void kernel_reduce(CRSCoord *dev_cors,
                              Voxel *dev_vxls,
                              CRSCoord *dev_r_cors,
                              Voxel *dev_r_vxls)
{
    ;
}

/////////////////////////////////////////////////////////////////

void partial_reduce(CRSCoord *raw_dev_cors,   Voxel *raw_dev_vxls,
                    CRSCoord *raw_r_dev_cors, Voxel *raw_r_dev_vxls)
{
    ;
}

void thrust_reduce(CRSCoord *raw_dev_cors,   Voxel *raw_dev_vxls,
                   CRSCoord *raw_r_dev_cors, Voxel *raw_r_dev_vxls)
{
    thrust::device_ptr<CRSCoord> dev_cors(raw_dev_cors);
    thrust::device_ptr<Voxel>    dev_vxls(raw_dev_vxls);

    thrust::device_ptr<CRSCoord> reduced_cors(raw_r_dev_cors);
    thrust::device_ptr<Voxel>    reduced_vxls(raw_r_dev_vxls);

    thrust::pair<thrust::device_ptr<CRSCoord>,
                 thrust::device_ptr<Voxel> > reduce_end;
    reduce_end.first  = reduced_cors;
    reduce_end.second = reduced_vxls;

    cudaEvent_t time_start, time_end;

    cudaEventCreate(&time_start);
    cudaEventCreate(&time_end);

    try {
        cudaEventRecord(time_start);

        reduce_end = thrust::reduce_by_key(dev_cors,
                                           dev_cors + BATCH_SIZE * imgsize * VERTICES,
                                           dev_vxls,
                                           reduced_cors,
                                           reduced_vxls);

        cudaEventRecord(time_end);

    } catch (thrust::system_error e) {
        std::cout << "Error detected in reduce by key: "
                  << e.what() << std::endl;
        exit(1);
    }
    
    cudaEventSynchronize(time_end);
    cudaCheckErrors("event sync");
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, time_start, time_end); 

    std::cout << "Thrust reduce done with time consumed: "
              << milliseconds << "ms" << std::endl;
}

/////////////////////////////////////////////////////////////////

int main(int argc, char *argv[])
{
    CRSCoord *coords, *raw_dev_cors, *raw_r_dev_cors;
    Voxel    *voxels, *raw_dev_vxls, *raw_r_dev_vxls;

    /* host buffer */
    cudaHostAlloc((void**)&coords,
                  BATCH_SIZE * imgsize * VERTICES * sizeof(CRSCoord),
                  cudaHostAllocDefault);
    cudaHostAlloc((void**)&voxels,
                  BATCH_SIZE * imgsize * VERTICES * sizeof(Voxel),
                  cudaHostAllocDefault);

    /* on device buffer */
    cudaMalloc((void**)&raw_dev_cors,
               BATCH_SIZE * imgsize * VERTICES * sizeof(CRSCoord));
    cudaMalloc((void**)&raw_dev_vxls,
               BATCH_SIZE * imgsize * VERTICES * sizeof(Voxel));

    cudaMalloc((void**)&raw_r_dev_cors,
               BATCH_SIZE * imgsize * VERTICES * sizeof(CRSCoord));
    cudaMalloc((void**)&raw_r_dev_vxls,
               BATCH_SIZE * imgsize * VERTICES * sizeof(Voxel));

    cudaCheckErrors("Memory allocation");

    /* data reading */
    int length = read_data_stream(coords, voxels);

    /* analysis */
    statistic(coords, voxels, length);

    /* thrust reduce */
    thrust_reduce(raw_dev_cors, raw_dev_vxls, raw_r_dev_cors, raw_r_dev_vxls);

    /* partial reduce */
    // TODO

    /* clean up */
    cudaFreeHost(coords);
    cudaFreeHost(voxels);

    cudaFree(raw_dev_cors);
    cudaFree(raw_dev_vxls);

    cudaFree(raw_r_dev_cors);
    cudaFree(raw_r_dev_vxls);

    return 0;
}
