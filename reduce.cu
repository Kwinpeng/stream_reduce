#include "kernels/reducevoxelbycoord.cuh"

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

using namespace mgpu;

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
//typedef double Voxel;

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

    std::cout << "Data reading begin..." << std::endl;
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

    //for (int i = 0; i < total_size; ++i)
    //    printf("coord:%10d, voxel:%15.6f\n", coords[i], voxels[i]);

    steady_clock::time_point end = steady_clock::now();
    std::cout << "Data reading done with time consumed: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()
              << "ms" << std::endl;

    return total_size;
}

void voxels_reshape(Voxel *voxels)
{
    const int total_size = BATCH_SIZE * imgsize * VERTICES;

    std::cout << "Data reshaping begin..." << std::endl;
    steady_clock::time_point begin = steady_clock::now();

    double *temp = (double*)malloc(total_size * sizeof(Voxel));
    for (int i = 0; i < total_size; ++i) {
        temp[i] = voxels[i].real;
        temp[total_size + i] = voxels[i].imag;
        temp[total_size * 2 + i] = voxels[i].weit;
    }

    memcpy(voxels, temp, total_size * sizeof(Voxel));

    free(temp);

    steady_clock::time_point end = steady_clock::now();
    std::cout << "Data reshaped done with time consumed: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()
              << "ms" << std::endl;
}

void record(CRSCoord *coords, CRSCoord *dev_coords,
            Voxel    *voxels, Voxel    *dev_voxels,
            const int length, const char *filename)
{
    FILE *fp = fopen(filename, "w");

    if (fp == NULL) {
        printf("File %s open error!\n", filename);
        exit(1);
    }

    cudaMemcpy(coords,
               dev_coords,
               length * sizeof(CRSCoord),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(voxels,
               dev_voxels,
               length * sizeof(Voxel),
               cudaMemcpyDeviceToHost);

    for (int i = 0; i < length; ++i) {
        fprintf(fp, "coord:%10d, voxel:%15.6f\n", coords[i], voxels[i].real);
    }

    fclose(fp);
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

        current++;
    }
}

/////////////////////////////////////////////////////////////////

__global__ void kernel_reduce(CRSCoord *dev_cors,
                              Voxel    *dev_vxls,
                              CRSCoord *dev_r_cors,
                              Voxel    *dev_r_vxls)
{
    ;
}

/////////////////////////////////////////////////////////////////

void partial_reduce(CRSCoord *raw_dev_cors,   Voxel *raw_dev_vxls,
                    CRSCoord *raw_r_dev_cors, Voxel *raw_r_dev_vxls)
{
    ;
}

int mgpu_reduce(CRSCoord *raw_dev_cors,   double *raw_dev_vxls,
                CRSCoord *raw_r_dev_cors, double *raw_r_dev_vxls,
                CudaContext& context)
{
    int *counts;
    cudaMalloc((void**)&counts, sizeof(int));

	context.Start();
    ReduceVoxelByCoord(raw_dev_cors,
                       raw_dev_vxls,
                       BATCH_SIZE * imgsize * VERTICES,
                       (double)0,
                       mgpu::plus<double>(),
                       mgpu::equal_to<CRSCoord>(),
                       raw_r_dev_cors,
                       raw_r_dev_vxls,
                       (int*)0,
                       counts,
                       context);
	double milliseconds = context.Split();

    std::cout << "MGPU reduce done with time consumed: "
              << milliseconds << "ms" << std::endl;

    int retrieval = 0;
    cudaMemcpy(&retrieval, counts, sizeof(int), cudaMemcpyDeviceToHost);

    return retrieval;
}

int thrust_reduce(CRSCoord *raw_dev_cors,   Voxel *raw_dev_vxls,
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

    return thrust::distance(reduced_cors, reduce_end.first);
}

/////////////////////////////////////////////////////////////////

int main(int argc, char *argv[])
{
    CRSCoord *coords, *r_coords, *raw_dev_cors, *raw_r_dev_cors;
    Voxel    *voxels, *r_voxels, *raw_dev_vxls, *raw_r_dev_vxls;

    const int size = BATCH_SIZE * imgsize * VERTICES;

    /* host buffer */
    cudaHostAlloc((void**)&coords, size * sizeof(CRSCoord),
                  cudaHostAllocDefault);
    cudaHostAlloc((void**)&voxels, size * sizeof(Voxel),
                  cudaHostAllocDefault);

    /* reduction result for verification */
    cudaHostAlloc((void**)&r_coords, size * sizeof(CRSCoord),
                  cudaHostAllocDefault);
    cudaHostAlloc((void**)&r_voxels, size * sizeof(Voxel),
                  cudaHostAllocDefault);

    /* on device buffer */
    cudaMalloc((void**)&raw_dev_cors, size * sizeof(CRSCoord));
    cudaMalloc((void**)&raw_dev_vxls, size * sizeof(Voxel));

    cudaMalloc((void**)&raw_r_dev_cors, size * sizeof(CRSCoord));
    cudaMalloc((void**)&raw_r_dev_vxls, size * sizeof(Voxel));
    cudaCheckErrors("Memory allocation");

    /* data reading */
    int length = read_data_stream(coords, voxels);

    /* analysis */
    //statistic(coords, voxels, length);
    
    /* upload data */
    cudaMemcpy(raw_dev_cors,
               coords,
               length * sizeof(CRSCoord),
               cudaMemcpyHostToDevice);
    cudaMemcpy(raw_dev_vxls,
               voxels,
               length * sizeof(Voxel),
               cudaMemcpyHostToDevice);
    cudaCheckErrors("Memory copy h2d");

    /* thrust reduce */
    int tlen = thrust_reduce(raw_dev_cors, raw_dev_vxls,
                             raw_r_dev_cors, raw_r_dev_vxls);

    record(r_coords, raw_r_dev_cors,
           r_voxels, raw_r_dev_vxls,
           tlen, "thrust-result.txt");

    /* mgpu reduce */
	ContextPtr context = CreateCudaDevice(argc, argv, true);

    voxels_reshape(voxels);

    int mlen = mgpu_reduce(raw_dev_cors,
                           (double*)raw_dev_vxls,
                           raw_r_dev_cors,
                           (double*)raw_r_dev_vxls,
                           *context);

    record(r_coords, raw_r_dev_cors,
           r_voxels, raw_r_dev_vxls,
           mlen, "mgpu-result.txt");

    if (tlen != mlen)
        printf("reduced length not equal, thrust: %d, mgpu: %d\n", tlen, mlen);

    /* partial reduce */
    // TODO

    /* clean up */
    cudaFreeHost(coords);
    cudaFreeHost(voxels);

    cudaFreeHost(r_coords);
    cudaFreeHost(r_voxels);

    cudaFree(raw_dev_cors);
    cudaFree(raw_dev_vxls);

    cudaFree(raw_r_dev_cors);
    cudaFree(raw_r_dev_vxls);

    return 0;
}
