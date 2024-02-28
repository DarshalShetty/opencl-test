 __kernel void Kernel (__global float* v0,
                       int off0,
                       int stride0,
                       __global float* v_out,
                        int stride_out)
{

    int i_out = get_global_id(0) * stride_out;
    int i0 = off0 + (i_out / stride_out) * stride0;

    // transliterate dup-1-ρ
    for (int i=0; i < stride_out; i++) {
        v_out[i_out+i] = v0[i0 + (i % stride0)];
    }
}
