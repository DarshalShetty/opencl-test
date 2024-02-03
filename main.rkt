#lang racket
(require ffi/cvector
         ffi/unsafe
         opencl/c
         "utils.rkt"
         malt/uniform-tensors/no-overrides
         (relative-in "../malt/uniform-tensors/tensors/."
                      "0-vectors.rkt"
                      "1-flats.rkt"
                      "A-equality.rkt")
         racket/runtime-path)

(define-runtime-path mult-kernel-file "mult.cl")
(define-runtime-path sum-kernel-file "sum.cl")

(define context (make-parameter #f))
(define command-queue (make-parameter #f))

(define (initialize devices device-idx)
  (context (clCreateContext #f (cvector->vector devices)))
  (command-queue (clCreateCommandQueue (context)
                                       (cvector-ref devices device-idx)
                                       '())))

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

(define (*/opencl t1 t2)
  (run/opencl (vector t1 t2)
              (vector 0 1)
              (shape t1)
              (file->bytes mult-kernel-file)
              #"Mult"))

(define (sum/opencl t)
  (run/opencl (vector t (last (shape t)))
              (vector 0 1)
              (take (shape t) (sub1 (len (shape t))))
              (file->bytes sum-kernel-file)
              #"Sum"))

(define (sum-test)
  (define t-shape '(10 10 500))
  (printf "Shape of tensor to be summed: ~a~n" t-shape)
  (define t (random-tensor 0 100 t-shape))
  (printf "Timing for CPU computation (in ms):~n")
  (define golden (time (sum-ρ t)))
  (printf "Timing for GPU computation (in ms):~n")
  (define result (time (sum/opencl t)))
  (check-tensor-equal? result golden))

(define (*-test)
  ;; TODO: make this work for tensors of different ranks. Refer to
  ;; ext2 code in flat tensors
  (define t-shape '(100 100 50))
  (printf "Shape of tensors to be multiplied: ~a~n" t-shape)
  (define t1 (random-tensor 0 100 t-shape))
  (define t2 (random-tensor 0 100 t-shape))
  (printf "Timing for CPU computation (in ms):~n")
  (define golden (time (*-ρ t1 t2)))
  (printf "Timing for GPU computation (in ms):~n")
  (define result (time (*/opencl t1 t2)))
  (check-tensor-equal? result golden))

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
          (*-test))
        cleanup))))

(main)
