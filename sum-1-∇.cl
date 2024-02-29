 __kernel void Kernel (__global float* g0,
                       __global float* v0,
                       int stride0,
                       __global float* vz,
                        int stridez)
{

    // compute indices passed into sum-1-∇ which were originally computed in
    // flat-ext1-∇
    int iz = get_global_id(0);
    // offset is handled by the platform API
    int i0 = 0 + (iz / stridez) * stride0;

    // transliterate sum-1-∇
    float z = vz[iz];
    for (int i=i0; i < i0+stride0; i++) {
        g0[i] += z;
    }
}
