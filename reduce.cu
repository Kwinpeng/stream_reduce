#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/system_error.h>
#include <thrust/sort.h>

#include <cuda.h>

#include <iostream>
#include <cstdlib>

/////////////////////////////////////////////////////////////////

#define BATCH_SIZE 136
#define VERTICES   8

const int imgsize = 500 * (500 / 2 + 1);

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

void read_data_stream(CRSCoord *coords, Voxel *voxels)
{
    FILE *fcor = fopen("../../data/stream/coords.dat", "rb");
    FILE *fvxl = fopen("../../data/stream/voxels.dat", "rb");

    if (fcor && fvxl) {
        if (fread(coords,
                  sizeof(CRSCoord),
                  BATCH_SIZE * imgsize * VERTICES,
                  fcor)
            == BATCH_SIZE * imgsize * VERTICES) {
            fclose(fcor);
        } else {
            std::cout << "coords.dat read error!\n";
            exit(1);
        }

        if (fread(voxels,
                  sizeof(Voxel),
                  BATCH_SIZE * imgsize * VERTICES,
                  fvxl)
            == BATCH_SIZE * imgsize * VERTICES) {
            fclose(fvxl);
        } else {
            std::cout << "voxels.dat read error!\n";
            exit(1);
        }
    } else {
        std::cout << "file open error!\n";
        exit(2);
    }
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

    try {
        reduce_end = thrust::reduce_by_key(dev_cors,
                                           dev_cors + BATCH_SIZE * imgsize * VERTICES,
                                           dev_vxls,
                                           reduced_cors,
                                           reduced_vxls);
    } catch (thrust::system_error e) {
        std::cout << "Error detected in reduce by key: "
                  << e.what() << std::endl;
        exit(1);
    }
}

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

    /* data reading */
    read_data_stream(coords, voxels);

    /* thrust reduce */
    thrust_reduce(raw_dev_cors, raw_dev_vxls, raw_r_dev_cors, raw_r_dev_vxls);

    /* clean up */
    cudaFreeHost(coords);
    cudaFreeHost(voxels);

    cudaFree(raw_dev_cors);
    cudaFree(raw_dev_vxls);

    cudaFree(raw_r_dev_cors);
    cudaFree(raw_r_dev_vxls);

    return 0;
}
