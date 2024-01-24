#lang racket
(require ffi/cvector
         ffi/unsafe
         opencl/c
         "utils.rkt"
         malt/flat-tensors/no-overrides
         (relative-in "../malt/flat-tensors/tensors/."
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
         [src-mem #f]
         [dst-mem #f]
         [src-buf #f]
         [dst-buf #f]
         [program #f]
         [kernel #f]
         [event #f])
     (dynamic-wind
       (λ ()
         (set! src-mem (malloc _cl_float src-size 'raw))
         (set! dst-mem (malloc _cl_float dst-size 'raw))
         (fill-with-tensor src-mem t)
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
                                           src-mem (make-vector 0)))
         (set! event (clEnqueueNDRangeKernel (command-queue) kernel 1
                                             (make-vector 1 dst-size)
                                             (make-vector 1 1)
                                             (make-vector 0)))
         (set! event (clEnqueueReadBuffer (command-queue) dst-buf 'CL_TRUE 0
                                          (* (ctype-sizeof _cl_float)
                                             dst-size)
                                          dst-mem (vector event)))
         (reshape dst-shape
                  (build-tensor (list dst-size)
                                (λ (i)
                                  (ptr-ref dst-mem _cl_float (ref i 0))))))
       (λ ()
         (when kernel
           (clReleaseKernel kernel))
         (when program
           (clReleaseProgram program))
         (when dst-buf
           (clReleaseMemObject dst-buf))
         (when src-buf
           (clReleaseMemObject src-buf))
         (free dst-mem)
         (free src-mem)))))

(define (main)
  (let* ([platform (cvector-ref (clGetPlatformIDs:vector) 0)]
         [devices (clGetDeviceIDs:vector platform 'CL_DEVICE_TYPE_GPU)]
         [device 0])
    (parameterize* ([context #f]
                    [command-queue #f])
      (dynamic-wind
        (λ () (initialize devices device))
        (λ ()
          (define t-shape '(1000 1000 500))
          (printf "Shape of tensor to be summed: ~a~n" t-shape)
          (define t (random-tensor 0 100 t-shape))
          (printf "Timing for CPU computation (in ms):~n")
          (define golden (time (sum-ρ t)))
          (printf "Timing for GPU computation (in ms):~n")
          (define result (time (sum/opencl t)))
          (check-tensor-equal? result golden))
        cleanup))))

(main)
