#lang racket
(require ffi/cvector
         ffi/unsafe
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

(define (scale-shape s)
  (map (λ (x) (* (shape-scale-factor) x)) s))

(define (debug-print-tensor t message)
  (define store (flat-store t))
  (printf "### ~a: (flat ~a '(" message (flat-shape t))
  (for ((i (in-range (vlen store))))
    (printf "~a " (vref store i)))
  (printf ") ~a)~n" (flat-offset t)))

(define (ext1-ρ-kernel prim1-ρ-f)
  #<<EOF
__kernel void Kernel (__global float* v0,
                    int stride0,
                    __global float* v_out,
                    int stride_out)
{

    int i_out = get_global_id(0) * stride_out;
    // offset is handled by the platform API
    int i0 = 0 + (i_out / stride_out) * stride0;

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

(define (ext2-∇-kernel prim2-∇-f generate-idxs)
  (let-values (((i0-expr i1-expr) (generate-idxs "iz")))
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

@{(prim2-∇-f "g0" "g1"
             "v0" "i0" "stride0"
             "v1" "i1" "stride1"
             "vz" "iz" "stride_z")}
}
EOF
    ))

(define (*-0-0-∇-kernel g0 g1
                        v0 i0 stride0
                        v1 i1 stride1
                        vz iz stride-z)
  #<<EOF
    @{g0}[@{i0}] = @{v1}[@{i1}] * @{vz}[@{iz}];
    @{g1}[@{i1}] = @{v0}[@{i0}] * @{vz}[@{iz}];
EOF
  )

(define (concat-base-∇-kernel g0 g1
                              v0 i0 stride0
                              v1 i1 stride1
                              vz iz stride-z)
  #<<EOF
    for(int i=0; i < @{stride-z}; i++) {
        if (i < @{stride0}) {
            @{g0}[i+@{i0}] += @{vz}[i+@{iz}];
        } else {
            @{g1}[i-@{stride0}+@{i1}] += @{vz}[i+@{iz}];
        }
    }
EOF
  )

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

(define (*-∇/opencl t1 t2 z)
  (flat-ext2-∇ *-0-0-∇-kernel 0 0 (λ _ '()) t1 t2 z))

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

(define idx-exprs-gen
  (λ (strides i0 i1)
    (λ (out-i)
      (for/fold ([i0 (number->string i0)]
                 [i1 (number->string i1)]
                 [x out-i] #:result (values i0 i1))
                ([stride strides])
        (let ((stride-out (number->string (vector-ref stride 0)))
              (stride0 (number->string (vector-ref stride 1)))
              (stride1 (number->string (vector-ref stride 2))))
          (let ((idx (binary-expr "/" x stride-out))
                (next-x (binary-expr "%" x stride-out)))
            (values (binary-expr "+" i0 (binary-expr "*" idx stride0))
                    (binary-expr "+" i1 (binary-expr "*" idx stride1))
                    next-x)))))))

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
       (λ (s-out size-out q0 q1 strides)
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
           (run-prim2-ρ! f strides
                         v0 off0 size0 stride0
                         v1 off1 size1 stride1
                         out-v size-out stride-out)
           (flat s-out out-v 0)))))))

(define (run-prim2-ρ! prim-kernel-f strides
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
                         (string->bytes/utf-8
                          (ext2-ρ-kernel prim-kernel-f
                                         (idx-exprs-gen strides 0 0))))))
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
        (λ (sz size-z q0 q1 strides)
          (let ((g0 (new-vec (size-of s0) 0.0))
                (g1 (new-vec (size-of s1) 0.0)))
            #;
            (for ([iz (in-range 0 size-z stride-z)])
              (let-values (((i0 i1)
                            (idxs strides iz off0 off1)))
                (fᵈ g0 g1 v0 i0 stride0 v1 i1 stride1 vz (+ offz iz) stride-z)))
            (run-prim2-∇! fᵈ strides g0 g1
                          v0 off0 size0 stride0
                          v1 off1 size1 stride1
                          vz offz size-z stride-z)
            (values (flat s0 g0 0)
                    (flat s1 g1 0))))))))

(define (run-prim2-∇! prim-kernel-f strides g0 g1
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
                        (make-vector
                         1
                         (string->bytes/utf-8
                          (ext2-∇-kernel prim-kernel-f
                                         (idx-exprs-gen strides 0 0))))))
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
      (run-prim1-ρ! f
                    v0 off0 size0 stride0
                    v-out size-out stride-out)
      (flat s-out v-out 0))))

(define (run-prim1-ρ! prim-kernel-f
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
                                                    (ext1-ρ-kernel prim-kernel-f)))))
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
  (define golden-∇ (time ((ext1-∇ sum-1-∇ 1 sum-shape) t (zeroes golden-ρ))))
  (printf "Timing for GPU computation (in ms):~n")
  (define result-∇ (time (sum-1-∇/opencl t (zeroes result-ρ))))
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

           (g0 (new-vec size0 0.0)))
      #;
      (for ([iz (in-range 0 size-z stride-z)]
            [i0 (in-range off0 (+ off0 size0) stride0)])
        (fᵈ g0 v0 i0 stride0 vz iz stride-z))
      (run-prim1-∇! fᵈ g0
                    v0 off0 size0 stride0
                    vz size-z stride-z)
      (flat s0 g0 0))))

(define (run-prim1-∇! prim-kernel-f g0
                      v0 off0 size0 stride0
                      vz size-z stride-z)
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
         (set! buf-z (clCreateBuffer (context) 'CL_MEM_WRITE_ONLY
                                       (* (ctype-sizeof _cl_float)
                                          size-z)
                                       #f))
         (set! buf-g (clCreateBuffer (context) 'CL_MEM_WRITE_ONLY
                                       (* (ctype-sizeof _cl_float)
                                          size0)
                                       #f))
         (set! program (clCreateProgramWithSource (context)
                                                  (make-vector
                                                   1
                                                   (string->bytes/utf-8
                                                    (ext1-∇-kernel prim-kernel-f)))))
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

(define *-∇
  (ext2-∇ (λ (a b z) (values (* b z) (* a z))) 0 0))

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
          1 1))

(define (*-test t-shape1 t-shape2)
  (define t-shape1 (scale-shape t-shape1))
  (define t-shape2 (scale-shape t-shape2))
  (printf "Shape of tensors to be multiplied: ~a ~a~n" t-shape1 t-shape2)
  (define t1 (random-tensor 0 100 t-shape1))
  (define t2 (random-tensor 0 100 t-shape2))
  (printf "Timing for CPU computation (in ms):~n")
  (define golden-ρ (time (*-ρ t1 t2)))
  (printf "Timing for GPU computation (in ms):~n")
  (define result-ρ (time (*-ρ/opencl t1 t2)))
  (check-tensor-equal? result-ρ golden-ρ)
  (printf "Timing for CPU computation (in ms):~n")
  (define-values (golden0-∇ golden1-∇) (time (*-∇ t1 t2 (zeroes golden-ρ))))
  (printf "Timing for GPU computation (in ms):~n")
  (define-values (result0-∇ result1-∇) (time (*-∇/opencl t1 t2 (zeroes result-ρ))))
  (check-tensor-equal? result0-∇ golden0-∇)
  (check-tensor-equal? result1-∇ golden1-∇))

(define (concat-test t-shape1 t-shape2)
  (define t-shape1 (scale-shape t-shape1))
  (define t-shape2 (scale-shape t-shape2))
  (printf "Shape of tensors to be concatenated: ~a ~a~n" t-shape1 t-shape2)
  (define t1 (random-tensor 0 100 t-shape1))
  (define t2 (random-tensor 0 100 t-shape2))
  (printf "Timing for CPU computation (in ms):~n")
  (define golden-ρ (time (concat-ρ t1 t2)))
  (printf "Timing for GPU computation (in ms):~n")
  (define result-ρ (time (concat-ρ/opencl t1 t2)))
  (check-tensor-equal? result-ρ golden-ρ)
  (printf "Timing for CPU computation (in ms):~n")
  (define-values (golden0-∇ golden1-∇) (time (concat-∇ t1 t2 (zeroes golden-ρ))))
  (printf "Timing for GPU computation (in ms):~n")
  (define-values (result0-∇ result1-∇) (time (concat-∇/opencl t1 t2 (zeroes result-ρ))))
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
          (*-test '(100 100 50) '(100 100 50))
          (*-test '(50) '(100 100 50))
          (*-test '(100 100 50) '(100 50))
          (concat-test '(100 100 50) '(100 50)))
        cleanup))))

(main)
