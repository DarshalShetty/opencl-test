 __kernel void Mult(__global float* src1, __global float* src2, __global float* dst)
{
    // find position in global arrays
    int iGID = get_global_id(0);

    // process
    dst[iGID] = src1[iGID] * src2[iGID];
}
