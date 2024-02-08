 __kernel void Sum (__global float* src, int iNumElements, __global float* dst)
{
    // find position in global arrays
    int iGID = get_global_id(0);

    // process
    int iInOffset = iGID * iNumElements;
    dst[iGID] = 0;
    for (int i=0; i < iNumElements; i++) {
        dst[iGID] += src[i + iInOffset];
    }
}
