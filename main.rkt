#lang racket
(require ffi/cvector
         ffi/unsafe
         rackunit
         opencl/c
         "utils.rkt"
         malt/uniform-tensors/no-overrides
         (relative-in "../malt/uniform-tensors/tensors/."
                      "0-vectors.rkt"
                      "1-flats.rkt"
                      "A-equality.rkt"
                      "D-extend.rkt")
         string-interpolation)

(define context (make-parameter #f))
(define command-queue (make-parameter #f))
(define shape-scale-factor (make-parameter 1))

(define (clSetKernelArgs kernel args)
  (for ([i (in-range (length args))])
    (match-define (list type-fn arg) (list-ref args i))
    #;(printf "###Kernel arg ~a: ~a~n" i arg)
    (type-fn kernel i arg)))

(define (scale-shape s)
  (map (λ (x) (* (shape-scale-factor) x)) s))

(define (debug-print-tensor t message)
  (define store (flat-store t))
  (println "### @{message}: (flat '@{(flat-shape t)} @{(with-output-to-string (λ ()
           (print-vec store)))} @{(flat-offset t)})" ))

(define calc-repeats
  (λ (s0 s1 r0 r1 s-out r-out)
    (define size-rep0 (size-of (drop-right s0 r0)))
    (define size-rep1 (size-of (drop-right s1 r1)))
    (define size-rep-out (size-of (drop-right s-out r-out)))
    #;(printf "size-rep0=~a size-rep1=~a size-rep-out-out=~a r-out=~a~n"
            size-rep0 size-rep1 size-rep-out r-out)
    (values (/ size-rep-out size-rep0)
            (/ size-rep-out size-rep1))))

(define idxs-inv
  (λ (strides i-in off-out i-rep repeats s-out left-in?)
    (for/fold ([i-out off-out]
               [dividend-rep i-rep]
               [predivisor-rep repeats]
               [x i-in] #:result i-out)
              ([desc-out s-out] ;; s-out = (append descents-out sf-out)
               [stride strides])
      #;(printf "i-out=~a dividend-rep=~a predivisor-rep=~a x=~a desc-out=~a stride=~a~n" i-out dividend-rep predivisor-rep x desc-out stride)
      (let ((stride-out (vector-ref stride 0))
            (stride-in (vector-ref stride (if left-in? 1 2))))
        (cond
          ((zero? stride-in)
           (let* ((divisor-rep (quotient predivisor-rep desc-out))
                  (scaling (quotient dividend-rep divisor-rep))
                  (next-dividend (remainder dividend-rep divisor-rep)))
             (values (+ i-out (* scaling stride-out))
                     next-dividend
                     divisor-rep
                     x)))
          (else
           (let ((idx (quotient x stride-in))
                 (next-x (remainder x stride-in)))
             (values (+ i-out (* idx stride-out))
                     dividend-rep
                     predivisor-rep
                     next-x))))))))

(define (print-ext2-index-info s0 s1 r0 r1 s-out r-out strides idxs0 idxs1)
  (displayln "####################################")
  (printf "input shape 0 =~a~n" s0)
  (printf "input shape 1 =~a~n" s1)
  (printf "output shape =~a~n" s-out)
  (define-values (strides-out strides0 strides1)
    (for/lists (st-o st0 st1)
               ((stride strides))
      (values (vector-ref stride 0)
              (vector-ref stride 1)
              (vector-ref stride 2))))
  (printf "strides-out=~a~nstrides0=~a~nstrides1=~a~n"
          strides-out strides0 strides1)
  (define-values (repeats0 repeats1) (calc-repeats s0 s1 r0 r1 s-out r-out))
  (displayln "Output indices corresponding to every index of input 0:")
  (define prim-stride0 (size-of (min-shape r0 s0)))
  (for ([i0 (in-range 0 (size-of s0) prim-stride0)])
    (printf "\tExpected\t~a\t=> ~a~n" i0 (reverse (vector-ref idxs0 i0)))
    (printf "\tActual\t\t~a\t=> ~a~n"
            i0
            (for/list ([i (in-range repeats0)])
              (idxs-inv strides i0 0 i repeats0 s-out #t))))
  (displayln "Output indices corresponding to every index of input 1:")
  (define prim-stride1 (size-of (min-shape r1 s1)))
  (for ([i1 (in-range 0 (size-of s1) prim-stride1)])
    (printf "\tExpected\t~a\t=> ~a~n" i1 (reverse (vector-ref idxs1 i1)))
    (printf "\tActual\t\t~a\t=> ~a~n"
            i1
            (for/list ([i (in-range repeats1)])
              (idxs-inv strides i1 0 i repeats1 s-out #f))))
  (displayln "####################################"))

(define (check-idxs-in s0 s1 r0 r1 s-out r-out strides idxs0 idxs1)
  (define-values (repeats0 repeats1) (calc-repeats s0 s1 r0 r1 s-out r-out))
  (define prim-stride0 (size-of (min-shape r0 s0)))
  (for ([i0 (in-range 0 (size-of s0) prim-stride0)])
    (define expected (reverse (vector-ref idxs0 i0)))
    (define actual
      (for/list ([i (in-range repeats0)])
        (idxs-inv strides i0 0 i repeats0 s-out #t)))
    (check-equal? actual expected))
  (define prim-stride1 (size-of (min-shape r1 s1)))
  (for ([i1 (in-range 0 (size-of s1) prim-stride1)])
    (define expected (reverse (vector-ref idxs1 i1)))
    (define actual
      (for/list ([i (in-range repeats1)])
        (idxs-inv strides i1 0 i repeats1 s-out #f)))
    (check-equal? actual expected)))

(define (ext1-ρ-kernel prim1-ρ-f)
  #<<EOF
__kernel void Kernel (__global float* v0,
                      int stride0,
                      __global float* v_out,
                      int stride_out)
{

    int i_out = get_global_id(0) * stride_out;
    // offset is handled by the platform API
    int i0 = (i_out / stride_out) * stride0;

@{(prim1-ρ-f "v0" "i0" "stride0" "v_out" "i_out" "stride_out")}

}
EOF
  )

(define (sum-1-ρ-kernel v0 i0 stride0 v-out i-out stride-out)
  #<<EOF
    float sum = 0;
    for (int i=@{i0}; i < @{i0}+@{stride0}; i++) {
        sum += @{v0}[i];
    }
    @{v-out}[@{i-out}] = sum;
EOF
  )

(define (dup-1-ρ-kernel v0 i0 stride0 v-out i-out stride-out)
  #<<EOF
    for (int i=0; i < @{stride-out}; i++) {
        @{v-out}[@{i-out}+i] = @{v0}[@{i0} + (i % @{stride0})];
    }
EOF
)

(define (ext1-∇-kernel prim1-∇-f)
  #<<EOF
__kernel void Kernel (__global float* g0,
                      __global float* v0,
                      int stride0,
                      __global float* vz,
                      int stridez)
{

    int iz = get_global_id(0) * stridez;
    // offset is handled by the platform API
    int i0 = 0 + (iz / stridez) * stride0;

@{(prim1-∇-f "g0" "v0" "i0" "stride0"
                  "vz" "iz" "stride-z")}
}
EOF
  )

(define (sum-1-∇-kernel g0 v0 i0 stride0 vz iz stride-z)
  #<<EOF
    float z = @{vz}[@{iz}];
    for (int i=@{i0}; i < @{i0}+@{stride0}; i++) {
        @{g0}[i] += z;
    }
EOF
  )


(define (ext2-ρ-kernel prim2-ρ-f generate-idxs)
  (let-values (((i0-expr i1-expr) (generate-idxs "i_out")))
    #<<EOF
__kernel void Kernel (__global float* v0,
                      int stride0,
                      __global float* v1,
                      int stride1,
                      __global float* v_out,
                      int stride_out)
{

    int i_out = get_global_id(0) * stride_out;
    int i0 = @{i0-expr};
    int i1 = @{i1-expr};

@{(prim2-ρ-f "v0" "i0" "stride0"
             "v1" "i1" "stride1"
             "v_out" "i_out" "stride_out")}
}
EOF
    ))

(define (*-0-0-ρ-kernel v0 i0 stride0
                        v1 i1 stride1
                        v-out i-out stride-out)
  #<<EOF
    @{v-out}[@{i-out}] = @{v0}[@{i0}] * @{v1}[@{i1}];
EOF
  )

(define (*-2-1-ρ-kernel v0 i0 stride0
                        v1 i1 stride1
                        v-out i-out stride-out)
  #<<EOF
    for(int i=0; i < @{stride-out}; i++) {
        @{v-out}[@{i-out}+i] = @{v0}[@{i0}+i] * @{v1}[@{i1}+i%@{stride1}];
    }
EOF
  )

(define (+-0-0-ρ-kernel v0 i0 stride0
                        v1 i1 stride1
                        v-out i-out stride-out)
  #<<EOF
    @{v-out}[@{i-out}] = @{v0}[@{i0}] + @{v1}[@{i1}];
EOF
  )

(define (concat-base-ρ-kernel v0 i0 stride0
                              v1 i1 stride1
                              v-out i-out stride-out)
  #<<EOF
    for(int i=0; i < @{stride-out}; i++) {
        if (i < @{stride0}) {
            @{v-out}[i+@{i-out}] = @{v0}[i+@{i0}];
        } else {
            @{v-out}[i+@{i-out}] = @{v1}[(i-@{stride0})+@{i1}];
        }
    }
EOF
  )

(define (ext2-∇-kernel prim2-∇-f strides
                       s0 s1 r0 r1 s-out r-out)
  (let*-values (((prim-effect0 prim-effect1) (prim2-∇-f "g"
                                                        "v0" "i0" "stride0"
                                                        "v1" "i1" "stride1"
                                                        "vz" "iz" "stride_z"))
                ((repeats0 repeats1) (calc-repeats s0 s1 r0 r1 s-out r-out))
                ((generate-idxs) (idx-exprs strides 0 0))
                ((generate-idxs-inv) (idx-exprs-inv strides 0
                                                    repeats0 repeats1 s-out))
                ((i0-expr i1-expr) (generate-idxs "iz"))
                ((iz-expr0 iz-expr1) (generate-idxs-inv "i0" "i1" "i_rep")))
    #<<EOF
__kernel void Kernel (__global float* g0,
                      __global float* g1,
                      __global float* v0,
                      int stride0,
                      int size0,
                      __global float* v1,
                      int stride1,
                      int size1,
                      __global float* vz,
                      int stride_z)
{
    int g_id = get_global_id(0);
    int i0_g = g_id * stride0;
    int i1_g = g_id * stride1;
    __global float *g;
    int i0, i1, iz;

    if (i0_g < size0) {
        g = g0;
        i0 = i0_g;
        for(int i_rep=0; i_rep<@{repeats0}; i_rep++) {
            iz = @{iz-expr0};
            i1 = @{i1-expr};

@{prim-effect0}
        }
    }

    if (i1_g < size1) {
        g = g1;
        i1 = i1_g;
        for(int i_rep=0; i_rep<@{repeats1}; i_rep++) {
            iz = @{iz-expr1};
            i0 = @{i0-expr};

@{prim-effect1}
        }
    }
}
EOF
    ))

(define (ext2-∇-kernel-atomic prim2-∇-f strides)
  (let*-values (((prim-effect0 prim-effect1) (prim2-∇-f "g"
                                                        "v0" "i0" "stride0"
                                                        "v1" "i1" "stride1"
                                                        "vz" "iz" "stride_z"))
                ((generate-idxs) (idx-exprs strides 0 0))
                ((i0-expr i1-expr) (generate-idxs "iz")))
    #<<EOF
__kernel void Kernel (__global float* g0,
                      __global float* g1,
                      __global float* v0,
                      int stride0,
                      __global float* v1,
                      int stride1,
                      __global float* vz,
                      int stride_z)
{

    int iz = get_global_id(0) * stride_z;
    int i0 = @{i0-expr};
    int i1 = @{i1-expr};
    __global float *g;

    g = g0;
@{prim-effect0}

    g = g1;
@{prim-effect1}
}
EOF
    ))

(define (ext2-∇-kernel-split prim2-∇-f strides
                             s0 s1 r0 r1 s-out r-out)
  (let*-values (((prim-effect0 prim-effect1) (prim2-∇-f "g"
                                                        "v0" "i0" "stride0"
                                                        "v1" "i1" "stride1"
                                                        "vz" "iz" "stride_z"))
                ((repeats0 repeats1) (calc-repeats s0 s1 r0 r1 s-out r-out))
                ((generate-idxs) (idx-exprs strides 0 0))
                ((generate-idxs-inv) (idx-exprs-inv strides 0
                                                    repeats0 repeats1 s-out))
                ((i0-expr i1-expr) (generate-idxs "iz"))
                ((iz-expr0 iz-expr1) (generate-idxs-inv "i0" "i1" "i_rep")))
    (values
     #<<EOF
__kernel void Kernel (__global float* g,
                      __global float* v0,
                      int stride0,
                      __global float* v1,
                      int stride1,
                      __global float* vz,
                      int stride_z)
{
    int i0 = get_global_id(0) * stride0;

    for(int i_rep=0; i_rep<@{repeats0}; i_rep++) {
        int iz = @{iz-expr0};
        int i1 = @{i1-expr};

@{prim-effect0}
    }
}
EOF

     #<<EOF
__kernel void Kernel (__global float* g,
                      __global float* v0,
                      int stride0,
                      __global float* v1,
                      int stride1,
                      __global float* vz,
                      int stride_z)
{
    int i1 = get_global_id(0) * stride1;

    for(int i_rep=0; i_rep<@{repeats1}; i_rep++) {
        int iz = @{iz-expr1};
        int i0 = @{i0-expr};

@{prim-effect1}
    }
}
EOF
     )))

(define (*-0-0-∇-kernel g
                        v0 i0 stride0
                        v1 i1 stride1
                        vz iz stride-z)
  (values
   #<<EOF
    @{g}[@{i0}] += @{v1}[@{i1}] * @{vz}[@{iz}];
EOF

   #<<EOF
    @{g}[@{i1}] += @{v0}[@{i0}] * @{vz}[@{iz}];
EOF
   ))

(define (*-2-1-∇-kernel g
                        v0 i0 stride0
                        v1 i1 stride1
                        vz iz stride-z)
  (values
   #<<EOF
    for(int i=0; i<@{stride-z}; i++) {
        float b = @{v1}[@{i1}+i%@{stride1}];
        float z = @{vz}[@{iz}+i];
        @{g}[@{i0}+i] += z * b;
    }
EOF

   #<<EOF
    for(int i=0; i<@{stride-z}; i++) {
        float a = @{v0}[@{i0}+i];
        float z = @{vz}[@{iz}+i];
        @{g}[@{i1}+i%@{stride1}] += z * a;
    }
EOF
   )
  )

(define (+-0-0-∇-kernel g
                        v0 i0 stride0
                        v1 i1 stride1
                        vz iz stride-z)
  (values
   #<<EOF
    @{g}[@{i0}] += @{vz}[@{iz}];
EOF

   #<<EOF
    @{g}[@{i1}] += @{vz}[@{iz}];
EOF
   ))


(define (concat-base-∇-kernel g
                              v0 i0 stride0
                              v1 i1 stride1
                              vz iz stride-z)
  (values
   #<<EOF
    for(int i=0; i < @{stride0}; i++) {
        @{g}[i+@{i0}] += @{vz}[i+@{iz}];
    }
EOF

   #<<EOF
    for(int i=@{stride0}; i < @{stride-z}; i++) {
        @{g}[i-@{stride0}+@{i1}] += @{vz}[i+@{iz}];
    }
EOF
   ))

(define (initialize devices device-idx)
  (define device (cvector-ref devices device-idx))
  (displayln "#### Device Info ####")
  (printDeviceInfo device)
  (displayln "#####################\n")
  (context (clCreateContext #f (cvector->vector devices)))
  (command-queue (clCreateCommandQueue (context) device '())))

(define (cleanup)
  (when (command-queue)
    (clReleaseCommandQueue (command-queue)))
  (when (context)
    (clReleaseContext (context))))

(define tensor->clBuffer
  (λ (t)
    (clCreateBuffer (context) '(CL_MEM_USE_HOST_PTR CL_MEM_READ_ONLY)
                    (* (ctype-sizeof _cl_float)
                       (size-of (shape t)))
                    (vec->cpointer (flat-store t)))))

(define (run/opencl ds in-refs out-shape ker-source ker-name)
  (let* ([out-shape out-shape]
         [out-size (size-of out-shape)]
         [out-store (new-vec out-size 0.0)]
         [num-in-refs (vector-length in-refs)]
         [inputs #f]
         [out-buf #f]
         [program #f]
         [kernel #f]
         [event #f])
     (dynamic-wind
       (λ ()
         (set! inputs (for/vector ([ref in-refs])
                          (let ([data (vector-ref ds ref)])
                            (cond
                              [(flat? data) (tensor->clBuffer data)]
                              [else data]))))
         (set! out-buf (clCreateBuffer (context) 'CL_MEM_WRITE_ONLY
                                       (* (ctype-sizeof _cl_float)
                                          out-size)
                                       #f))
         (set! program (clCreateProgramWithSource (context)
                                                  (make-vector
                                                   1
                                                   ker-source)))
         (clBuildProgram program (make-vector 0) (make-bytes 0))
         (set! kernel (clCreateKernel program ker-name))
         (for ([i (in-range num-in-refs)]
               [in inputs])
           (cond
             [(integer? in) (clSetKernelArg:_cl_int kernel i in)]
             [(cpointer? in)
              (clSetKernelArg:_cl_mem kernel i in)]
             [else (error 'cl-kernel-arg-type "Cannot handle kernel argument: ~a"
                          in)]))
         (clSetKernelArg:_cl_mem kernel num-in-refs out-buf))
       (λ ()
         (set! event (clEnqueueNDRangeKernel (command-queue) kernel 1
                                             (make-vector 1 out-size)
                                             (make-vector 0)
                                             (make-vector 0)))
         (set! event (clEnqueueReadBuffer (command-queue) out-buf 'CL_TRUE 0
                                          (* (ctype-sizeof _cl_float)
                                             out-size)
                                          (vec->cpointer out-store) (vector event)))
         (flat out-shape out-store 0))
       (λ ()
         (when kernel
           (clReleaseKernel kernel))
         (when program
           (clReleaseProgram program))
         (when out-buf
           (clReleaseMemObject out-buf))
         (when inputs
           (for ([in-buf inputs])
             (when (and (cpointer? in-buf)
                        (cpointer-has-tag? in-buf 'cl_mem))
               (clReleaseMemObject in-buf))))))))

(define (*-ρ/opencl t1 t2)
  (flat-ext2-ρ *-0-0-ρ-kernel 0 0 (λ _ '()) t1 t2))

(define *-2-1-shape
  (λ (s t)
    s))

(define (*-2-1-ρ/opencl t1 t2)
  (flat-ext2-ρ *-2-1-ρ-kernel 2 1 *-2-1-shape t1 t2))

(define (+-ρ/opencl t1 t2)
  (flat-ext2-ρ +-0-0-ρ-kernel 0 0 (λ _ '()) t1 t2))

(define (*-∇/opencl^ t1 t2 z)
  (flat-ext2-∇^ *-0-0-∇-kernel 0 0 (λ _ '()) t1 t2 z))

(define (*-∇/opencl t1 t2 z)
  (flat-ext2-∇ *-0-0-∇-kernel 0 0 (λ _ '()) t1 t2 z))

(define (*-2-1-∇/opencl t1 t2 z)
  (flat-ext2-∇ *-2-1-∇-kernel 2 1 *-2-1-shape t1 t2 z))

(define (+-∇/opencl t1 t2 z)
  (flat-ext2-∇ +-0-0-∇-kernel 0 0 (λ _ '()) t1 t2 z))

(define concat-shape
  (λ (st su)
    (cons (+ (ref st 0) (ref su 0))
          (cdr st))))

(define (concat-ρ/opencl t1 t2)
  (flat-ext2-ρ concat-base-ρ-kernel 1 1 concat-shape t1 t2))

(define (concat-∇/opencl t1 t2 z)
  (flat-ext2-∇ concat-base-∇-kernel 1 1 concat-shape t1 t2 z))

(define (binary-expr rator rand1 rand2)
  (string-append "(" rand1 " " rator " " rand2 ")"))

(define idx-exprs
  (λ (strides i0 i1)
    (λ (i-out-var-str)
      (for/fold ([i0 (number->string i0)]
                 [i1 (number->string i1)]
                 [x i-out-var-str] #:result (values i0 i1))
                ([stride strides])
        (let ((stride-out (number->string (vector-ref stride 0)))
              (stride0 (number->string (vector-ref stride 1)))
              (stride1 (number->string (vector-ref stride 2))))
          (let ((idx (binary-expr "/" x stride-out))
                (next-x (binary-expr "%" x stride-out)))
            (values (binary-expr "+" i0 (binary-expr "*" idx stride0))
                    (binary-expr "+" i1 (binary-expr "*" idx stride1))
                    next-x)))))))

(define idx-exprs-inv
  (λ (strides i-out repeats0 repeats1 s-out)
    (λ (i0-var-str i1-var-str i-rep-var-str)
      (let ((gen-expr
             (λ (i-in-var-str stride-i repeats)
               (for/fold ([i-out (number->string i-out)]
                          [dividend-rep i-rep-var-str]
                          [predivisor-rep repeats]
                          [x i-in-var-str] #:result i-out)
                         ([desc-out s-out] ;; s-out = (append descents-out sf-out)
                          [stride strides])
                 (let ((stride-out (vector-ref stride 0))
                       (stride-in (vector-ref stride stride-i)))
                   (cond
                     ((zero? stride-in)
                      (let* ((divisor-rep (quotient predivisor-rep desc-out))
                             (divisor-rep-str (number->string divisor-rep))
                             (scaling (binary-expr "/" dividend-rep divisor-rep-str))
                             (next-dividend (binary-expr "%"
                                                         dividend-rep
                                                         divisor-rep-str)))
                        (values (binary-expr "+" i-out
                                             (binary-expr "*"
                                                          scaling
                                                          (number->string
                                                           stride-out)))
                                next-dividend
                                divisor-rep
                                x)))
                     (else
                      (let ((stride-in-str (number->string stride-in)))
                        (let ((idx (binary-expr "/" x stride-in-str))
                              (next-x (binary-expr "%" x stride-in-str)))
                          (values (binary-expr "+" i-out
                                               (binary-expr "*" idx
                                                            (number->string
                                                             stride-out)))
                                  dividend-rep
                                  predivisor-rep
                                  next-x))))))))))
        (values (gen-expr i0-var-str 1 repeats0)
                (gen-expr i1-var-str 2 repeats1))))))

(define flat-ext2-ρ
  (λ (f r0 r1 shape-fn t0 t1)
    (let* ((s0 (flat-shape t0))
           (v0 (flat-store t0))
           (off0 (flat-offset t0))
           (size0 (size-of s0))
           (sf0 (min-shape r0 s0))

           (s1 (flat-shape t1))
           (v1 (flat-store t1))
           (off1 (flat-offset t1))
           (size1 (size-of s1))
           (sf1 (min-shape r1 s1))

           (sf-out (shape-fn sf0 sf1))
           (stride0 (size-of sf0))
           (stride1 (size-of sf1))
           (stride-out (size-of sf-out)))
      (ext2-shapes
       s0 s1 r0 r1 sf-out
       (λ (s-out size-out q0 q1 strides parallel-desc?)
         #;(printf "###sz=~a, size-z=~a, q0=~a, q1=~a, strides=~a~n" s-out size-out q0 q1 strides)
         (let ((out-v (new-vec size-out 0.0)))
           #;
           (begin
             (printf "General Expressions:~n")
             (define-values (i0-expr i1-expr) (gen-exprs #"i-out"))
             (printf "i0 = ~a~n" i0-expr)
             (printf "i1 = ~a~n~n" i1-expr)
             (for ([out-i (in-range 0 size-out stride-out)])
               (let-values (((i0 i1)
                             (idxs strides out-i off0 off1))
                            ((i0-expr i1-expr)
                             (gen-exprs (number->string out-i))))
                 (printf "~a = ~a ~n" i0 i0-expr)
                 (printf "~a = ~a ~n" i1 i1-expr)
                 #;
                 (f v0 i0 stride0 v1 i1 stride1 out-v (+ 0 out-i) stride-out))))
           (run-prim2-ρ! (ext2-ρ-kernel f (idx-exprs strides 0 0))
                         v0 off0 size0 stride0
                         v1 off1 size1 stride1
                         out-v size-out stride-out)
           (flat s-out out-v 0)))))))

(define (run-prim2-ρ! kernel-code
                      v0 off0 size0 stride0
                      v1 off1 size1 stride1
                      v-out size-out stride-out)
  (let* ([buf0 #f]
         [buf1 #f]
         [buf-out #f]
         [program #f]
         [kernel #f]
         [event #f])
     (dynamic-wind
       (λ ()
         ;; Exclude memory consumed by elements before offset of input vector v0
         (set! buf0 (clCreateBuffer (context)
                                    '(CL_MEM_USE_HOST_PTR CL_MEM_READ_ONLY)
                                    (* (ctype-sizeof _cl_float)
                                       size0)
                                    (vref-cpointer v0 off0)))
         (set! buf1 (clCreateBuffer (context)
                                    '(CL_MEM_USE_HOST_PTR CL_MEM_READ_ONLY)
                                    (* (ctype-sizeof _cl_float)
                                       size1)
                                    (vref-cpointer v1 off1)))
         (set! buf-out (clCreateBuffer (context) 'CL_MEM_WRITE_ONLY
                                       (* (ctype-sizeof _cl_float)
                                          size-out)
                                       #f))
         (set! program (clCreateProgramWithSource
                        (context)
                        (make-vector
                         1
                         (string->bytes/utf-8 kernel-code))))
         (clBuildProgram program (make-vector 0) (make-bytes 0))
         (set! kernel (clCreateKernel program #"Kernel"))
         (clSetKernelArg:_cl_mem kernel 0 buf0)
         (clSetKernelArg:_cl_int kernel 1 stride0)
         (clSetKernelArg:_cl_mem kernel 2 buf1)
         (clSetKernelArg:_cl_int kernel 3 stride1)
         (clSetKernelArg:_cl_mem kernel 4 buf-out)
         (clSetKernelArg:_cl_int kernel 5 stride-out))
       (λ ()
         (set! event (clEnqueueNDRangeKernel (command-queue) kernel 1
                                             (make-vector 1 (/ size-out stride-out))
                                             (make-vector 0)
                                             (make-vector 0)))
         (set! event (clEnqueueReadBuffer (command-queue) buf-out 'CL_TRUE 0
                                          (* (ctype-sizeof _cl_float)
                                             size-out)
                                          (vec->cpointer v-out) (vector event))))
       (λ ()
         (when kernel
           (clReleaseKernel kernel))
         (when program
           (clReleaseProgram program))
         (when buf-out
           (clReleaseMemObject buf-out))
         (when buf1
           (clReleaseMemObject buf1))
         (when buf0
           (clReleaseMemObject buf0))))))


(define ext2-shapes
  (λ (s0 s1 r0 r1 sf-out k)
    (let ((l0 (length s0))
          (l1 (length s1)))
      (cond
        ((and (= r0 l0) (= r1 l1))
           (k sf-out
              (size-of sf-out)
              (size-of s0)
              (size-of s1)
              '()
              #t))

        ((= r0 l0)
         (ext2-shapes s0 (cdr s1) r0 r1 sf-out
           (desc-right (car s1) k)))

        ((= r1 l1)
         (ext2-shapes (cdr s0) s1 r0 r1 sf-out
           (desc-left (car s0) k)))

        ((and (not (null? s0))
              (not (null? s1))
              (= (car s0) (car s1)))
         (ext2-shapes (cdr s0) (cdr s1) r0 r1 sf-out
           (desc-both (car s0) k)))

        ((> l1 l0)
         (ext2-shapes s0 (cdr s1) r0 r1 sf-out
           (desc-right (car s1) k)))

        ((> l0 l1)
         (ext2-shapes (cdr s0) s1 r0 r1 sf-out
           (desc-left (car s0) k)))

        (else (error 'ext
               "Shapes are incompatible for ext2: ~a, and ~a for min ranks ~a, and ~a~%"
               s0 s1 r0 r1))))))

(define desc-both
  (λ (d k)
    (λ (s-out qout q0 q1 strides parallel-desc?)
      (k (cons d s-out)
         (* qout d)
         (* q0 d)
         (* q1 d)
         (cons (vector qout q0 q1) strides)
         parallel-desc?))))

(define desc-left
  (λ (d k)
    (λ (s-out qout q0 q1 strides parallel-desc?)
      (k (cons d s-out)
         (* qout d)
         (* q0 d)
         q1
         (cons (vector qout q0 0) strides)
         #f))))

(define desc-right
  (λ (d k)
    (λ (s-out qout q0 q1 strides parallel-desc?)
      (k (cons d s-out)
         (* qout d)
         q0
         (* q1 d)
         (cons (vector qout 0 q1) strides)
         #f))))

(define flat-ext2-∇^
  (λ (fᵈ r0 r1 shape-fn t0 t1 z)
    (let* ((s0 (flat-shape t0))
           (v0 (flat-store t0))
           (off0 (flat-offset t0))
           (size0 (size-of s0))
           (sf0 (min-shape r0 s0))
           (stride0 (size-of sf0))

           (s1 (flat-shape t1))
           (v1 (flat-store t1))
           (off1 (flat-offset t1))
           (size1 (size-of s1))
           (sf1 (min-shape r1 s1))
           (stride1 (size-of sf1))

           (sf-z (shape-fn sf0 sf1))
           (stride-z (size-of sf-z))
           (vz (flat-store z))
           (offz (flat-offset z)))
      (ext2-shapes s0 s1 r0 r1 sf-z
        (λ (sz size-z q0 q1 strides parallel-desc?)
          (let ((g0 (new-vec (size-of s0) 0.0))
                (g1 (new-vec (size-of s1) 0.0)))
            #;
            (printf "###sz=~a, size-z=~a, q0=~a, q1=~a, strides=~a~n" sz size-z q0 q1 strides)
            #|
            (define idxs0 (make-vector (size-of s0) '()))
            (define idxs1 (make-vector (size-of s1) '()))
            (for ([iz (in-range 0 size-z stride-z)])
              (let-values (((i0 i1)
                            (idxs strides iz off0 off1)))
                (vector-set! idxs0 i0 (cons iz (vector-ref idxs0 i0)))
                (vector-set! idxs1 i1 (cons iz (vector-ref idxs1 i1)))
                #;(fᵈ g0 g1 v0 i0 stride0 v1 i1 stride1 vz (+ offz iz) stride-z)))
            (check-idxs-in
             s0 s1 r0 r1 sz (length sf-z)
             strides idxs0 idxs1)
            |#
            (let ((kernel-code (ext2-∇-kernel fᵈ strides s0 s1 r0 r1 sz
                                              (length sf-z))))
              (run-prim2-∇! kernel-code
                            g0 g1
                            v0 off0 size0 stride0
                            v1 off1 size1 stride1
                            vz offz size-z stride-z))

            (values (flat s0 g0 0)
                    (flat s1 g1 0))))))))

(define (run-prim2-∇! kernel-code g0 g1
                      v0 off0 size0 stride0
                      v1 off1 size1 stride1
                      vz offz size-z stride-z)
  (let* ([global-work-size (max (/ size0 stride0)
                                (/ size1 stride1))]
         [buf0 #f]
         [buf1 #f]
         [buf-z #f]
         [buf-g0 #f]
         [buf-g1 #f]
         [program #f]
         [kernel #f]
         [event #f])
    (dynamic-wind
     (λ ()
       ;; Exclude memory consumed by elements before offset of input vector v0
       (set! buf0 (clCreateBuffer (context)
                                  '(CL_MEM_USE_HOST_PTR CL_MEM_READ_ONLY)
                                  (* (ctype-sizeof _cl_float)
                                     size0)
                                  (vref-cpointer v0 off0)))
       (set! buf1 (clCreateBuffer (context)
                                  '(CL_MEM_USE_HOST_PTR CL_MEM_READ_ONLY)
                                  (* (ctype-sizeof _cl_float)
                                     size1)
                                  (vref-cpointer v1 off1)))
       (set! buf-z (clCreateBuffer (context)
                                   '(CL_MEM_USE_HOST_PTR CL_MEM_READ_ONLY)
                                   (* (ctype-sizeof _cl_float)
                                      size-z)
                                   (vref-cpointer vz offz)))
       (set! buf-g0 (clCreateBuffer (context) 'CL_MEM_WRITE_ONLY
                                   (* (ctype-sizeof _cl_float)
                                      size0)
                                   #f))
       (set! buf-g1 (clCreateBuffer (context) 'CL_MEM_WRITE_ONLY
                                   (* (ctype-sizeof _cl_float)
                                      size1)
                                   #f))
       #;(printf "###Source:~n~a~n" kernel-code)
       (set! program (clCreateProgramWithSource
                      (context)
                      (make-vector 1 (string->bytes/utf-8 kernel-code))))
       (clBuildProgram program (make-vector 0) (make-bytes 0))
       (set! kernel (clCreateKernel program #"Kernel"))
       (clSetKernelArgs kernel
                        `((,clSetKernelArg:_cl_mem ,buf-g0)
                          (,clSetKernelArg:_cl_mem ,buf-g1)
                          (,clSetKernelArg:_cl_mem ,buf0)
                          (,clSetKernelArg:_cl_int ,stride0)
                          (,clSetKernelArg:_cl_int ,size0)
                          (,clSetKernelArg:_cl_mem ,buf1)
                          (,clSetKernelArg:_cl_int ,stride1)
                          (,clSetKernelArg:_cl_int ,size1)
                          (,clSetKernelArg:_cl_mem ,buf-z)
                          (,clSetKernelArg:_cl_int ,stride-z))))
     (λ ()
       (set! event (clEnqueueNDRangeKernel (command-queue) kernel 1
                                           (make-vector 1 global-work-size)
                                           (make-vector 0)
                                           (make-vector 0)))
       (set! event (clEnqueueReadBuffer (command-queue) buf-g0 'CL_TRUE 0
                                        (* (ctype-sizeof _cl_float)
                                           size0)
                                        (vec->cpointer g0) (vector event)))
       (set! event (clEnqueueReadBuffer (command-queue) buf-g1 'CL_TRUE 0
                                        (* (ctype-sizeof _cl_float)
                                           size1)
                                        (vec->cpointer g1) (vector event))))
     (λ ()
       (when kernel
         (clReleaseKernel kernel))
       (when program
         (clReleaseProgram program))
       (when buf-g0
         (clReleaseMemObject buf-g0))
       (when buf-g1
         (clReleaseMemObject buf-g1))
       (when buf-z
         (clReleaseMemObject buf-z))
       (when buf1
         (clReleaseMemObject buf1))
       (when buf0
         (clReleaseMemObject buf0))))))

(define flat-ext2-∇
  (λ (fᵈ r0 r1 shape-fn t0 t1 z)
    (let* ((s0 (flat-shape t0))
           (v0 (flat-store t0))
           (off0 (flat-offset t0))
           (size0 (size-of s0))
           (sf0 (min-shape r0 s0))
           (stride0 (size-of sf0))

           (s1 (flat-shape t1))
           (v1 (flat-store t1))
           (off1 (flat-offset t1))
           (size1 (size-of s1))
           (sf1 (min-shape r1 s1))
           (stride1 (size-of sf1))

           (sf-z (shape-fn sf0 sf1))
           (stride-z (size-of sf-z))
           (vz (flat-store z))
           (offz (flat-offset z)))
      (ext2-shapes s0 s1 r0 r1 sf-z
        (λ (sz size-z q0 q1 strides parallel-desc?)
          (let ((g0 (new-vec (size-of s0) 0.0))
                (g1 (new-vec (size-of s1) 0.0)))
            #;
            (printf "###sz=~a, size-z=~a, q0=~a, q1=~a, strides=~a~n" sz size-z q0 q1 strides)
            #|
            (define idxs0 (make-vector (size-of s0) '()))
            (define idxs1 (make-vector (size-of s1) '()))
            (for ([iz (in-range 0 size-z stride-z)])
              (let-values (((i0 i1)
                            (idxs strides iz off0 off1)))
                (vector-set! idxs0 i0 (cons iz (vector-ref idxs0 i0)))
                (vector-set! idxs1 i1 (cons iz (vector-ref idxs1 i1)))
                #;(fᵈ g0 g1 v0 i0 stride0 v1 i1 stride1 vz (+ offz iz) stride-z)))
            (check-idxs-in
             s0 s1 r0 r1 sz (length sf-z)
             strides idxs0 idxs1)
            |#
            (cond
                 (parallel-desc?
                  (run-prim2-∇-atomic! (ext2-∇-kernel-atomic fᵈ strides)
                                       g0 g1
                                       v0 off0 size0 stride0
                                       v1 off1 size1 stride1
                                       vz offz size-z stride-z))
                 (else
                  (let*-values (((kernel-code0 kernel-code1)
                                 (ext2-∇-kernel-split fᵈ strides s0 s1 r0 r1 sz
                                                      (length sf-z))))
                    (run-prim2-∇-split! kernel-code0 kernel-code1
                                        g0 g1
                                        v0 off0 size0 stride0
                                        v1 off1 size1 stride1
                                        vz offz size-z stride-z))))

            (values (flat s0 g0 0)
                    (flat s1 g1 0))))))))

(define (run-prim2-∇-atomic! kernel-code g0 g1
                             v0 off0 size0 stride0
                             v1 off1 size1 stride1
                             vz offz size-z stride-z)
  (let* ([buf0 #f]
         [buf1 #f]
         [buf-z #f]
         [buf-g0 #f]
         [buf-g1 #f]
         [program #f]
         [kernel #f]
         [event #f])
    #;(printf "###offz=~a, size-z=~a, stride-z=~a~n" offz size-z stride-z)
    (dynamic-wind
     (λ ()
       ;; Exclude memory consumed by elements before offset of input vector v0
       (set! buf0 (clCreateBuffer (context)
                                  '(CL_MEM_USE_HOST_PTR CL_MEM_READ_ONLY)
                                  (* (ctype-sizeof _cl_float)
                                     size0)
                                  (vref-cpointer v0 off0)))
       (set! buf1 (clCreateBuffer (context)
                                  '(CL_MEM_USE_HOST_PTR CL_MEM_READ_ONLY)
                                  (* (ctype-sizeof _cl_float)
                                     size1)
                                  (vref-cpointer v1 off1)))
       (set! buf-z (clCreateBuffer (context)
                                   '(CL_MEM_USE_HOST_PTR CL_MEM_READ_ONLY)
                                   (* (ctype-sizeof _cl_float)
                                      size-z)
                                   (vref-cpointer vz offz)))
       (set! buf-g0 (clCreateBuffer (context) 'CL_MEM_WRITE_ONLY
                                    (* (ctype-sizeof _cl_float)
                                       size0)
                                    #f))
       (set! buf-g1 (clCreateBuffer (context) 'CL_MEM_WRITE_ONLY
                                    (* (ctype-sizeof _cl_float)
                                       size1)
                                    #f))
       (set! program (clCreateProgramWithSource
                      (context)
                      (make-vector 1 (string->bytes/utf-8 kernel-code))))
       (clBuildProgram program (make-vector 0) (make-bytes 0))
       (set! kernel (clCreateKernel program #"Kernel"))
       (clSetKernelArg:_cl_mem kernel 0 buf-g0)
       (clSetKernelArg:_cl_mem kernel 1 buf-g1)
       (clSetKernelArg:_cl_mem kernel 2 buf0)
       (clSetKernelArg:_cl_int kernel 3 stride0)
       (clSetKernelArg:_cl_mem kernel 4 buf1)
       (clSetKernelArg:_cl_int kernel 5 stride1)
       (clSetKernelArg:_cl_mem kernel 6 buf-z)
       (clSetKernelArg:_cl_int kernel 7 stride-z))
     (λ ()
       (set! event (clEnqueueNDRangeKernel (command-queue) kernel 1
                                           (make-vector 1 (/ size-z stride-z))
                                           (make-vector 0)
                                           (make-vector 0)))
       (set! event (clEnqueueReadBuffer (command-queue) buf-g0 'CL_TRUE 0
                                        (* (ctype-sizeof _cl_float)
                                           size0)
                                        (vec->cpointer g0) (vector event)))
       (set! event (clEnqueueReadBuffer (command-queue) buf-g1 'CL_TRUE 0
                                        (* (ctype-sizeof _cl_float)
                                           size1)
                                        (vec->cpointer g1) (vector event))))
     (λ ()
       (when kernel
         (clReleaseKernel kernel))
       (when program
         (clReleaseProgram program))
       (when buf-g1
         (clReleaseMemObject buf-g1))
       (when buf-g0
         (clReleaseMemObject buf-g0))
       (when buf-z
         (clReleaseMemObject buf-z))
       (when buf1
         (clReleaseMemObject buf1))
       (when buf0
         (clReleaseMemObject buf0))))))

(define (run-prim2-∇-split! kernel-code0 kernel-code1 g0 g1
                            v0 off0 size0 stride0
                            v1 off1 size1 stride1
                            vz offz size-z stride-z)
  (define (run! kernel-code g size-in stride-in)
    (let* ([buf0 #f]
         [buf1 #f]
         [buf-z #f]
         [buf-g #f]
         [program #f]
         [kernel #f]
         [event #f])
    (dynamic-wind
     (λ ()
       ;; Exclude memory consumed by elements before offset of input vector v0
       (set! buf0 (clCreateBuffer (context)
                                  '(CL_MEM_USE_HOST_PTR CL_MEM_READ_ONLY)
                                  (* (ctype-sizeof _cl_float)
                                     size0)
                                  (vref-cpointer v0 off0)))
       (set! buf1 (clCreateBuffer (context)
                                  '(CL_MEM_USE_HOST_PTR CL_MEM_READ_ONLY)
                                  (* (ctype-sizeof _cl_float)
                                     size1)
                                  (vref-cpointer v1 off1)))
       (set! buf-z (clCreateBuffer (context)
                                   '(CL_MEM_USE_HOST_PTR CL_MEM_READ_ONLY)
                                   (* (ctype-sizeof _cl_float)
                                      size-z)
                                   (vref-cpointer vz offz)))
       (set! buf-g (clCreateBuffer (context) 'CL_MEM_WRITE_ONLY
                                   (* (ctype-sizeof _cl_float)
                                      size-in)
                                   #f))
       (set! program (clCreateProgramWithSource
                      (context)
                      (make-vector 1 (string->bytes/utf-8 kernel-code))))
       (clBuildProgram program (make-vector 0) (make-bytes 0))
       (set! kernel (clCreateKernel program #"Kernel"))
       (clSetKernelArgs kernel
                        `((,clSetKernelArg:_cl_mem ,buf-g)
                          (,clSetKernelArg:_cl_mem ,buf0)
                          (,clSetKernelArg:_cl_int ,stride0)
                          (,clSetKernelArg:_cl_mem ,buf1)
                          (,clSetKernelArg:_cl_int ,stride1)
                          (,clSetKernelArg:_cl_mem ,buf-z)
                          (,clSetKernelArg:_cl_int ,stride-z))))
     (λ ()
       (set! event (clEnqueueNDRangeKernel (command-queue) kernel 1
                                           (make-vector 1 (/ size-in stride-in))
                                           (make-vector 0)
                                           (make-vector 0)))
       (set! event (clEnqueueReadBuffer (command-queue) buf-g 'CL_TRUE 0
                                        (* (ctype-sizeof _cl_float)
                                           size-in)
                                        (vec->cpointer g) (vector event))))
     (λ ()
       (when kernel
         (clReleaseKernel kernel))
       (when program
         (clReleaseProgram program))
       (when buf-g
         (clReleaseMemObject buf-g))
       (when buf-z
         (clReleaseMemObject buf-z))
       (when buf1
         (clReleaseMemObject buf1))
       (when buf0
         (clReleaseMemObject buf0))))))
  (run! kernel-code0 g0 size0 stride0)
  (run! kernel-code1 g1 size1 stride1))

;; TODO: For any tensor t0, a shape function shape-fn, minimum rank
;; min-rank and associated values derived from them, show that:
;;
;; ∃ fn . i0s ≡ (map fn i-outs)
;;
;; where,
;; i-outs = (in-range 0 size-out stride-out)
;; i0s = (in-range off0 (+ off0 size0) stride0)
;;
;; i-outs and i0s are the sequences which are looped over in the original
;; definition of flat-ext1-ρ.
;;
;; Proof:
;;
;; From the semantics of in-range, we can infer that:
;; (in-range start end step) ≡ (map (λ (i) (+ start (* i step)))
;;                                  (in-range (/ (- end start) step)))
;;
;; We will also be using the map fusion property:
;; (map f (map g ls)) ≡ (map (compose f g) ls)
;;
;; Set fn = (λ (i) (+ off0 (* (/ i stride-out) stride0)))
;;
;; i0s ≡ (in-range off0 (+ off0 size0) stride0)
;;     ≡ (map (λ (i) (+ off0 (* i stride0))) (in-range (/ size0 stride0)))
;;     ≡ (map (λ (i) (+ off0 (* i stride0)))
;;            (in-range (/ (size-of s0)
;;                         (size-of sf0))))
;;     ≡ (map (λ (i) (+ off0 (* i stride0)))
;;            (in-range (/ (size-of s0)
;;                         (size-of (min-shape min-rank s0))))
;;
;; (map (λ (i) (+ off0 (* (/ i stride-out) stride0))) i-outs)
;; ≡ (map (λ (i) (+ off0 (* (/ i stride-out) stride0)))
;;        (in-range 0 size-out stride-out))
;; ≡ (map (λ (i) (+ off0 (* (/ i stride-out) stride0)))
;;        (map (λ (i) (* i stride-out))
;;             (in-range (/ size-out stride-out))))
;; ≡ (map (compose (λ (i) (+ off0 (* (/ i stride-out) stride0)))
;;                 (λ (i) (* i stride-out)))
;;        (in-range (/ size-out stride-out)))
;; ≡ (map (compose (λ (i) (+ off0 (* (/ i stride-out) stride0)))
;;                 (λ (i) (* i stride-out)))
;;        (in-range (/ (size-of s-out)
;;                     (size-of sf-out))))
;; ≡ (map (compose (λ (i) (+ off0 (* (/ i stride-out) stride0)))
;;                 (λ (i) (* i stride-out)))
;;        (in-range (/ (size-of (merge-shapes s0 min-rank sf-out))
;;                     (size-of (shape-fn sf0)))))
;; ≡ (map (compose (λ (i) (+ off0 (* (/ i stride-out) stride0)))
;;                 (λ (i) (* i stride-out)))
;;        (in-range (/ (size-of (merge-shapes s0 min-rank (shape-fn sf0)))
;;                     (size-of (shape-fn sf0)))))
;; ≡ (map (λ (i) (+ off0 (* (/ (* i stride-out) stride-out) stride0)))
;;        (in-range (/ (size-of (merge-shapes s0 min-rank (shape-fn sf0)))
;;                     (size-of (shape-fn sf0)))))
;; ≡ (map (λ (i) (+ off0 (* i stride0)))
;;        (in-range (/ (size-of (merge-shapes s0 min-rank (shape-fn sf0)))
;;                     (size-of (shape-fn sf0)))))
;; ≡ (map (λ (i) (+ off0 (* i stride0)))
;;        (in-range (/ (size-of
;;                      (merge-shapes s0 min-rank
;;                                    (shape-fn (min-shape min-rank s0))))
;;                     (size-of (shape-fn (min-shape min-rank s0))))))
;;
;; We now need to show that:
;; ∀ s0.
;; (/ (size-of s0) (size-of (min-shape min-rank s0)))
;; ≡ (/ (size-of
;;       (merge-shapes s0 min-rank
;;                     (shape-fn (min-shape min-rank s0))))
;;      (size-of (shape-fn (min-shape min-rank s0))))
;;
(define flat-ext1-ρ
  (λ (f min-rank shape-fn t0)
    (let* ((s0 (flat-shape t0))
           (v0 (flat-store t0))
           (off0 (flat-offset t0))
           (sf0 (min-shape min-rank s0))
           (stride0 (size-of sf0))
           (size0 (size-of s0))

           (sf-out (shape-fn sf0))
           (stride-out (size-of sf-out))
           (s-out (merge-shapes s0 min-rank sf-out))
           (size-out (size-of s-out))
           (v-out (new-vec size-out 0.0)))
      #;
      (printf "sf-out=~a~nstride-out=~a~ns-out=~a~nsize-out=~a~nsf0=~a~n"
              sf-out stride-out s-out size-out sf0)
      #;
      (for ([i-out (in-range 0 size-out stride-out)]
            [i0 (in-range off0 (+ off0 size0) stride0)])
        (f v0 i0 stride0 v-out i-out stride-out))
      (run-prim1-ρ! (ext1-ρ-kernel f)
                    v0 off0 size0 stride0
                    v-out size-out stride-out)
      (flat s-out v-out 0))))

(define (run-prim1-ρ! kernel-code
                      v0 off0 size0 stride0
                      v-out size-out stride-out)
  (let* ([buf0 #f]
         [buf-out #f]
         [program #f]
         [kernel #f]
         [event #f])
    (dynamic-wind
     (λ ()
       ;; Exclude memory consumed by elements before offset of input vector v0
       (set! buf0 (clCreateBuffer (context)
                                  '(CL_MEM_USE_HOST_PTR CL_MEM_READ_ONLY)
                                  (* (ctype-sizeof _cl_float)
                                     size0)
                                  (vref-cpointer v0 off0)))
       (set! buf-out (clCreateBuffer (context) 'CL_MEM_WRITE_ONLY
                                     (* (ctype-sizeof _cl_float)
                                        size-out)
                                     #f))
       (set! program (clCreateProgramWithSource (context)
                                                (make-vector
                                                 1
                                                 (string->bytes/utf-8
                                                  kernel-code))))
       (clBuildProgram program (make-vector 0) (make-bytes 0))
       (set! kernel (clCreateKernel program #"Kernel"))
       (clSetKernelArg:_cl_mem kernel 0 buf0)
       (clSetKernelArg:_cl_int kernel 1 stride0)
       (clSetKernelArg:_cl_mem kernel 2 buf-out)
       (clSetKernelArg:_cl_int kernel 3 stride-out))
     (λ ()
       (set! event (clEnqueueNDRangeKernel (command-queue) kernel 1
                                           (make-vector 1 (/ size-out stride-out))
                                           (make-vector 0)
                                           (make-vector 0)))
       (set! event (clEnqueueReadBuffer (command-queue) buf-out 'CL_TRUE 0
                                        (* (ctype-sizeof _cl_float)
                                           size-out)
                                        (vec->cpointer v-out) (vector event))))
     (λ ()
       (when kernel
         (clReleaseKernel kernel))
       (when program
         (clReleaseProgram program))
       (when buf-out
         (clReleaseMemObject buf-out))
       (when buf0
         (clReleaseMemObject buf0))))))

(define (sum-shape st)
  (refr st 1))

(define (sum-1-ρ/opencl t)
  (flat-ext1-ρ sum-1-ρ-kernel
               1 sum-shape t))

(define (sum-1-∇/opencl t z)
  (flat-ext1-∇ sum-1-∇-kernel
               1 sum-shape t z))

(define sum-1-∇
  (λ (g0 v0 i0 stride0
      vz iz stride-z)
    (let ((z (vref vz iz)))
      (for ([i (in-range i0 (+ i0 stride0))])
        (vset! g0 i
          (+ (vref g0 i) z))))))

(define (sum-test)
  (define t-shape (scale-shape '(10 10 500)))
  (printf "Shape of tensor to be summed: ~a~n" t-shape)
  (define t (random-tensor 0 100 t-shape))
  (printf "Timing for CPU computation (in ms):~n")
  (define golden-ρ (time (sum-ρ t)))
  (printf "Timing for GPU computation (in ms):~n")
  (define result-ρ (time (sum-1-ρ/opencl t)))
  (check-tensor-equal? result-ρ golden-ρ)
  (printf "Timing for CPU computation (in ms):~n")
  (define golden-∇ (time ((ext1-∇ sum-1-∇ 1 sum-shape) t (one-like golden-ρ))))
  (printf "Timing for GPU computation (in ms):~n")
  (define result-∇ (time (sum-1-∇/opencl t (one-like result-ρ))))
  (check-tensor-equal? result-∇ golden-∇))

(define dup-f
  (λ (in-v iᵢ sᵢ out-v iₒ sₒ)
    (for ([i (in-range 0 sₒ)])
      (vset! out-v (+ iₒ i)
             (vref in-v (+ iᵢ (modulo i sᵢ)))))))

(define dup-shape-f
  (λ (in-f-shape)
    (list (* 2 (car in-f-shape)))))

(define dup-ρ (ext1-ρ dup-f 1 dup-shape-f))

(define (dup/opencl t)
  (flat-ext1-ρ dup-1-ρ-kernel
               1 dup-shape-f t))
(define (dup-test)
  (define t-shape (scale-shape '(100 100 50)))
  (printf "Shape of tensor to be duplicated: ~a~n" t-shape)
  (define t (random-tensor 0 100 t-shape))
  (printf "Timing for CPU computation (in ms):~n")
  (define golden (time (dup-ρ t)))
  (printf "Timing for GPU computation (in ms):~n")
  (define result (time (dup/opencl t)))
  (check-tensor-equal? result golden))

(define flat-ext1-∇
  (λ (fᵈ min-rank shape-fn t0 z)
    ;; z has the same shape as the output
    (let* ((s0 (flat-shape t0))
           (v0 (flat-store t0))
           (off0 (flat-offset t0))
           (sf0 (min-shape min-rank s0))
           (stride0 (size-of sf0))
           (size0 (size-of s0))

           (sz (flat-shape z))
           (size-z (size-of sz))
           (sf-z (shape-fn sf0))
           (stride-z (size-of sf-z))
           (vz (flat-store z))
           (offz (flat-offset z))

           (g0 (new-vec size0 0.0)))
      #;
      (for ([iz (in-range 0 size-z stride-z)]
            [i0 (in-range off0 (+ off0 size0) stride0)])
        (fᵈ g0 v0 i0 stride0 vz iz stride-z))
      (run-prim1-∇! (ext1-∇-kernel fᵈ) g0
                    v0 off0 size0 stride0
                    vz offz size-z stride-z)
      (flat s0 g0 0))))

(define (run-prim1-∇! kernel-code g0
                      v0 off0 size0 stride0
                      vz offz size-z stride-z)
  (let* ([buf0 #f]
         [buf-z #f]
         [buf-g #f]
         [program #f]
         [kernel #f]
         [event #f])
     (dynamic-wind
       (λ ()
         ;; Exclude memory consumed by elements before offset of input vector v0
         (set! buf0 (clCreateBuffer (context)
                                    '(CL_MEM_USE_HOST_PTR CL_MEM_READ_ONLY)
                                    (* (ctype-sizeof _cl_float)
                                       size0)
                                    (vref-cpointer v0 off0)))
         (set! buf-z (clCreateBuffer (context)
                                         '(CL_MEM_USE_HOST_PTR CL_MEM_READ_ONLY)
                                         (* (ctype-sizeof _cl_float)
                                            size-z)
                                         (vref-cpointer vz offz)))
         (set! buf-g (clCreateBuffer (context) 'CL_MEM_WRITE_ONLY
                                       (* (ctype-sizeof _cl_float)
                                          size0)
                                       #f))

         (set! program (clCreateProgramWithSource (context)
                                                  (make-vector
                                                   1
                                                   (string->bytes/utf-8
                                                    kernel-code))))
         (clBuildProgram program (make-vector 0) (make-bytes 0))
         (set! kernel (clCreateKernel program #"Kernel"))
         (clSetKernelArg:_cl_mem kernel 0 buf-g)
         (clSetKernelArg:_cl_mem kernel 1 buf0)
         (clSetKernelArg:_cl_int kernel 2 stride0)
         (clSetKernelArg:_cl_mem kernel 3 buf-z)
         (clSetKernelArg:_cl_int kernel 4 stride-z))
       (λ ()
         (set! event (clEnqueueNDRangeKernel (command-queue) kernel 1
                                             (make-vector 1 (/ size-z stride-z))
                                             (make-vector 0)
                                             (make-vector 0)))
         (set! event (clEnqueueReadBuffer (command-queue) buf-g 'CL_TRUE 0
                                          (* (ctype-sizeof _cl_float)
                                             size0)
                                          (vec->cpointer g0) (vector event))))
       (λ ()
         (when kernel
           (clReleaseKernel kernel))
         (when program
           (clReleaseProgram program))
         (when buf-g
           (clReleaseMemObject buf-g))
         (when buf-z
           (clReleaseMemObject buf-z))
         (when buf0
           (clReleaseMemObject buf0))))))

(define *-2-1-∇
  (ext2-∇ (λ (g0 g1 v0 i0 stride0
                 v1 i1 stride1
                 vz iz stride-z)
            (for ([i (in-range 0 stride-z)])
              (let ((a (vref v0 (+ i0 i)))
                    (b (vref v1 (+ i1 (modulo i stride1))))
                    (z (vref vz (+ iz i))))
                (vset! g0 (+ i0 i)
                       (+ (vref g0 (+ i0 i)) (* z b)))
                (vset! g1 (+ i1 (modulo  i stride1))
                       (+ (vref g1 (+ i1 (modulo i stride1))) (* z a))))))
          2 1 *-2-1-shape))

(define *-∇
  (ext2-∇ (λ (a b z) (values (* b z) (* a z))) 0 0))

(define +-∇
  (ext2-∇ (λ (a b z) (values z z)) 0 0))

(define concat-∇
  (ext2-∇ (λ (g0 g1 v0 i0 stride0
                 v1 i1 stride1
                 vz iz stride-z)
            (for ([i (in-range 0 stride-z)])
              (cond
                ((< i stride0)
                 (vset! g0 (+ i0 i)
                        (+ (vref g0 (+ i0 i))
                           (vref vz (+ iz i)))))
                (else
                 (vset! g1 (+ i1 (- i stride0))
                        (+ (vref g1 (+ i1 (- i stride0)))
                           (vref vz (+ iz i))))))))
          1 1 concat-shape))

(define (*-2-1-test t-shape1^ t-shape2^)
  (define t-shape1 (scale-shape t-shape1^))
  (define t-shape2 (scale-shape t-shape2^))
  (printf "Shape of tensors to be *-2-1: ~a ~a~n" t-shape1 t-shape2)
  (define t1 (random-tensor 0 100 t-shape1))
  (define t2 (random-tensor 0 100 t-shape2))
  (printf "Timing for CPU computation (in ms):~n")
  (define golden-ρ (time (*-2-1-ρ t1 t2)))
  (printf "Timing for GPU computation (in ms):~n")
  (define result-ρ (time (*-2-1-ρ/opencl t1 t2)))
  (check-tensor-equal? result-ρ golden-ρ)
  (printf "Timing for CPU computation (in ms):~n")
  (define-values (golden0-∇ golden1-∇) (time (*-2-1-∇ t1 t2 (one-like golden-ρ))))
  (printf "Timing for GPU computation (in ms):~n")
  (define-values (result0-∇ result1-∇) (time (*-2-1-∇/opencl t1 t2 (one-like result-ρ))))
  (check-tensor-equal? result0-∇ golden0-∇)
  (check-tensor-equal? result1-∇ golden1-∇))

(define (*-test^ t-shape1^ t-shape2^)
  (define t-shape1 (scale-shape t-shape1^))
  (define t-shape2 (scale-shape t-shape2^))
  (printf "Shape of tensors to be multiplied: ~a ~a~n" t-shape1 t-shape2)
  (define t1 (random-tensor 0 100 t-shape1))
  (define t2 (random-tensor 0 100 t-shape2))
  (define result-ρ (*-ρ/opencl t1 t2))
  (printf "Timing for GPU computation (in ms):~n")
  (define-values (result0-∇ result1-∇) (time (*-∇/opencl t1 t2 (one-like result-ρ))))
  (printf "Timing for GPU computation (in ms):~n")
  (define-values (result0-∇^ result1-∇^) (time (*-∇/opencl^ t1 t2 (one-like result-ρ))))
  (check-tensor-equal? result0-∇^ result0-∇)
  (check-tensor-equal? result1-∇^ result1-∇))

(define (*-test t-shape1^ t-shape2^)
  (define t-shape1 (scale-shape t-shape1^))
  (define t-shape2 (scale-shape t-shape2^))
  (printf "Shape of tensors to be multiplied: ~a ~a~n" t-shape1 t-shape2)
  (define t1 (random-tensor 0 100 t-shape1))
  (define t2 (random-tensor 0 100 t-shape2))
  (printf "Timing for CPU computation (in ms):~n")
  (define golden-ρ (time (*-ρ t1 t2)))
  (printf "Timing for GPU computation (in ms):~n")
  (define result-ρ (time (*-ρ/opencl t1 t2)))
  (check-tensor-equal? result-ρ golden-ρ)
  (printf "Timing for CPU computation (in ms):~n")
  (define-values (golden0-∇ golden1-∇) (time (*-∇ t1 t2 (one-like golden-ρ))))
  (printf "Timing for GPU computation (in ms):~n")
  (define-values (result0-∇ result1-∇) (time (*-∇/opencl t1 t2 (one-like result-ρ))))
  #;(debug-print-tensor result0-∇ "result0-∇")
  #;(debug-print-tensor golden0-∇ "golden0-∇")
  (check-tensor-equal? result0-∇ golden0-∇)
  #;(debug-print-tensor result1-∇ "result1-∇")
  #;(debug-print-tensor golden1-∇ "golden1-∇")
  (check-tensor-equal? result1-∇ golden1-∇))

(define (concat-test t-shape1^ t-shape2^)
  (define t-shape1 (scale-shape t-shape1^))
  (define t-shape2 (scale-shape t-shape2^))
  (printf "Shape of tensors to be concatenated: ~a ~a~n" t-shape1 t-shape2)
  (define t1 (random-tensor 0 100 t-shape1))
  (define t2 (random-tensor 0 100 t-shape2))
  (printf "Timing for CPU computation (in ms):~n")
  (define golden-ρ (time (concat-ρ t1 t2)))
  (printf "Timing for GPU computation (in ms):~n")
  (define result-ρ (time (concat-ρ/opencl t1 t2)))
  (check-tensor-equal? result-ρ golden-ρ)
  (printf "Timing for CPU computation (in ms):~n")
  (define-values (golden0-∇ golden1-∇) (time (concat-∇ t1 t2 (one-like golden-ρ))))
  (printf "Timing for GPU computation (in ms):~n")
  (define-values (result0-∇ result1-∇) (time (concat-∇/opencl t1 t2 (one-like result-ρ))))
  (check-tensor-equal? result0-∇ golden0-∇)
  (check-tensor-equal? result1-∇ golden1-∇))


(define (+-test t-shape1^ t-shape2^)
  (define t-shape1 (scale-shape t-shape1^))
  (define t-shape2 (scale-shape t-shape2^))
  (printf "Shape of tensors to be added: ~a ~a~n" t-shape1 t-shape2)
  (define t1 (random-tensor 0 100 t-shape1))
  (define t2 (random-tensor 0 100 t-shape2))
  (printf "Timing for CPU computation (in ms):~n")
  (define golden-ρ (time (+-ρ t1 t2)))
  (printf "Timing for GPU computation (in ms):~n")
  (define result-ρ (time (+-ρ/opencl t1 t2)))
  (check-tensor-equal? result-ρ golden-ρ)
  (printf "Timing for CPU computation (in ms):~n")
  (define-values (golden0-∇ golden1-∇) (time (+-∇ t1 t2 (one-like golden-ρ))))
  (printf "Timing for GPU computation (in ms):~n")
  (define-values (result0-∇ result1-∇) (time (+-∇/opencl t1 t2 (one-like result-ρ))))
  (check-tensor-equal? result0-∇ golden0-∇)
  (check-tensor-equal? result1-∇ golden1-∇))

(define (main)
  (let* ([platform (cvector-ref (clGetPlatformIDs:vector) 0)]
         [devices (clGetDeviceIDs:vector platform 'CL_DEVICE_TYPE_GPU)]
         [device 0])
    (parameterize* ([context #f]
                    [command-queue #f])
      (dynamic-wind
        (λ () (initialize devices device))
        (λ ()
          (sum-test)
          (dup-test)
          (+-test '(100 100 50) '(100 100 50))
          (+-test '(50) '(100 100 50))
          (+-test '(100 100 50) '(100 50))
          (+-test '(30) '(20 30))
          (+-test '(20 30) '(20 30))
          (+-test '(20 30) '(20 20 30))
          (+-test '(20 30) '(30 20 30))
          (+-test '(40) '(30 20 40))
          (+-test '(20 30 20) '(20 30 30 20))
          (+-test '(20 30 20) '(20 20 30 30 20))
          (+-test '(4 20 30 20) '(4 4 20 20 30 30 20))
          (*-test '(100 100 50) '(100 100 50))
          (*-test '(50) '(100 100 50))
          (*-test '(100 100 50) '(100 50))
          (*-test '(30) '(20 30))
          (*-test '(20 30) '(20 30))
          (*-test '(20 30) '(20 20 30))
          (*-test '(20 30) '(30 20 30))
          (*-test '(40) '(30 20 40))
          (*-test '(20 30 20) '(20 30 30 20))
          (*-test '(20 30 20) '(20 20 30 30 20))
          (*-test '(4 20 30 20) '(4 4 20 20 30 30 20))
          (for ((i (in-range 10)))
            (*-test^ '(4 20 30 20) '(4 4 20 20 30 30 20)))
          (*-2-1-test '(30 20 30) '(20 30 10 30))
          (concat-test '(100 100 50) '(100 50))
          (concat-test '(20 30) '(40 20 30)))
        cleanup))))

(main)
