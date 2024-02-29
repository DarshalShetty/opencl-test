 __kernel void Kernel (__global float* v0,
                       int stride0,
                       __global float* v_out,
                        int stride_out)
{

    // compute indices passed into sum-1-ρ which were originally computed in
    // flat-ext1-ρ
    int i_out = get_global_id(0);
    // offset is handled by the platform API
    int i0 = 0 + (i_out / stride_out) * stride0;

    // transliterate sum-1-ρ
    float sum = 0;
    for (int i=i0; i < i0+stride0; i++) {
        sum += v0[i];
    }
    v_out[i_out] = sum;
}
