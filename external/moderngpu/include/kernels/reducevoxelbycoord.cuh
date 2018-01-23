#include "../kernels/reducebykey.cuh"

namespace mgpu {

////////////////////////////////////////////////////////////////////////////////
// ReduceByKeyPreprocess

template<typename ValType, typename KeyType, typename KeysIt, typename Comp>
MGPU_HOST void ReduceVoxelByCoordPreprocess(int count, KeysIt keys_global, 
    KeyType* keysDest_global, Comp comp, int* count_host, int* count_global,
    std::unique_ptr<ReduceByKeyPreprocessData>* ppData, CudaContext& context) {

    typedef typename SegReducePreprocessTuning<sizeof(ValType)>::Tuning Tuning;
    int2 launch = Tuning::GetLaunchParams(context);
    int NV = launch.x * launch.y;

    const bool AsyncTransfer = true;

    std::unique_ptr<ReduceByKeyPreprocessData> 
        data(new ReduceByKeyPreprocessData);

    int numBlocks = MGPU_DIV_UP(count, NV);
    data->count = count;
    data->numBlocks = numBlocks;
    data->limitsDevice = context.Malloc<int>(numBlocks + 1);
    data->threadCodesDevice = context.Malloc<int>(numBlocks * launch.x);

    // Fill out thread codes for each thread in the processing CTAs.
    KernelReduceByKeyPreprocess<Tuning>
        <<<numBlocks, launch.x, 0, context.Stream()>>>(keys_global, count, 
        data->threadCodesDevice->get(), data->limitsDevice->get(), comp);
    MGPU_SYNC_CHECK("KernelReduceByKeyPreprocess");

    // Scan the output counts.
    Scan<MgpuScanTypeExc>(data->limitsDevice->get(), numBlocks, 0,
        mgpu::plus<int>(), data->limitsDevice->get() + numBlocks, (int*)0,
        data->limitsDevice->get(), context);

    // Queue up a transfer of the row total.
    if(count_global)
        copyDtoD(count_global, data->limitsDevice->get() + numBlocks, 1,
            context.Stream());
    if(count_host) {
        if(AsyncTransfer) {
            cudaError_t error = cudaMemcpyAsync(context.PageLocked(), 
                data->limitsDevice->get() + numBlocks, sizeof(int),
                cudaMemcpyDeviceToHost, context.Stream());
            error = cudaEventRecord(context.Event(), context.Stream());
        } else
            copyDtoH(count_host, data->limitsDevice->get() + numBlocks, 1);
    }

    // Output one key per segment.
    if(keysDest_global) {
        KernelReduceByKeyEmit<Tuning>
            <<<numBlocks, launch.x, 0, context.Stream()>>>(keys_global,
            count, data->threadCodesDevice->get(), data->limitsDevice->get(),
            keysDest_global);
        MGPU_SYNC_CHECK("KernelReduceByKeyEmit");
    }

    // Retrieve the number of rows.
    if(AsyncTransfer && count_host) {
        cudaError_t error = cudaEventSynchronize(context.Event());
        *count_host = *context.PageLocked();
    }
    data->numSegments = count_host ? *count_host : -1;

    //*ppData = std::move(data);
    std::swap(*ppData, data);
}

////////////////////////////////////////////////////////////////////////////////
// ReduceByKey host function.

template<typename KeysIt, typename InputIt, typename DestIt,
    typename KeyType, typename ValType, typename Op, typename Comp>
MGPU_HOST void ReduceVoxelByCoord(KeysIt keys_global, InputIt data_global, int count,
    ValType identity, Op op, Comp comp, KeyType* keysDest_global, 
    DestIt dest_global, int* count_host, int* count_global, 
    CudaContext& context) {
        
    std::unique_ptr<ReduceByKeyPreprocessData> data;
    MGPU_MEM(int) countsDevice = context.Malloc<int>(1);
    if(count_host && !count_global) count_global = countsDevice->get();

    // Preprocess the keys and emit the first key in each segment.
    ReduceVoxelByCoordPreprocess<ValType>(count, keys_global, keysDest_global, comp, 
        (int*)0, count_global, &data, context);
    
    const bool AsyncTransfer = true;
    if(count_host) {
        if(AsyncTransfer) {
            cudaError_t error = cudaMemcpyAsync(context.PageLocked(), 
                count_global, sizeof(int), cudaMemcpyDeviceToHost, 
                context.Stream());
            error = cudaEventRecord(context.Event(), context.Stream());
        } else
            copyDtoH(count_host, count_global, 1);
    }

    // Evaluate the segmented reduction.
    SegReduceApply(*data, data_global, identity, op, dest_global, context);

    // Retrieve the number of segments.
    if(AsyncTransfer && count_host) {
        cudaError_t error = cudaEventSynchronize(context.Event());
        *count_host = *context.PageLocked();
    }
}

} // namespace mgpu
