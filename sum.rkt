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

(define (sum/opencl t)
  (let* ([src-size (size-of (shape t))]
         [dst-shape (take (shape t) (sub1 (len (shape t))))]
         [dst-size (size-of dst-shape)]
         [dst-store (new-vec dst-size 0.0)]
         [src-buf #f]
         [dst-buf #f]
         [program #f]
         [kernel #f]
         [event #f])
     (dynamic-wind
       (λ ()
         (set! src-buf (clCreateBuffer (context) 'CL_MEM_READ_ONLY
                                       (* (ctype-sizeof _cl_float)
                                          src-size)
                                       #f))
         (set! dst-buf (clCreateBuffer (context) 'CL_MEM_WRITE_ONLY
                                       (* (ctype-sizeof _cl_float)
                                          dst-size)
                                       #f))
         (set! program (clCreateProgramWithSource (context)
                                                  (make-vector
                                                   1
                                                   (file->bytes sum-kernel-file))))
         (clBuildProgram program (make-vector 0) (make-bytes 0))
         (set! kernel (clCreateKernel program #"Sum"))
         (clSetKernelArg:_cl_mem kernel 0 src-buf)
         (clSetKernelArg:_cl_mem kernel 1 dst-buf)
         (clSetKernelArg:_cl_int kernel 2 (last (shape t))))
       (λ ()
         (set! event (clEnqueueWriteBuffer (command-queue) src-buf 'CL_FALSE 0
                                           (* (ctype-sizeof _cl_float)
                                              src-size)
                                           (vec->cpointer (flat-store t))
                                           (make-vector 0)))
         (set! event (clEnqueueNDRangeKernel (command-queue) kernel 1
                                                (make-vector 1 dst-size)
                                                (make-vector 1 1)
                                                (make-vector 0)))
         (set! event (clEnqueueReadBuffer (command-queue) dst-buf 'CL_TRUE 0
                                          (* (ctype-sizeof _cl_float)
                                             dst-size)
                                          (vec->cpointer dst-store) (vector event)))
         (flat dst-shape dst-store 0))
       (λ ()
         (when kernel
           (clReleaseKernel kernel))
         (when program
           (clReleaseProgram program))
         (when dst-buf
           (clReleaseMemObject dst-buf))
         (when src-buf
           (clReleaseMemObject src-buf))))))

(define (main)
  (let* ([platform (cvector-ref (clGetPlatformIDs:vector) 0)]
         [devices (clGetDeviceIDs:vector platform 'CL_DEVICE_TYPE_GPU)]
         [device 0])
    (parameterize* ([context #f]
                    [command-queue #f])
      (dynamic-wind
        (λ () (initialize devices device))
        (λ ()
          (define t-shape '(10 10 500))
          (printf "Shape of tensor to be summed: ~a~n" t-shape)
          (define t (random-tensor 0 100 t-shape))
          (printf "Timing for CPU computation (in ms):~n")
          (define golden (time (sum-ρ t)))
          (printf "Timing for GPU computation (in ms):~n")
          (define result (time (sum/opencl t)))
          (check-tensor-equal? result golden))
        cleanup))))

(main)
