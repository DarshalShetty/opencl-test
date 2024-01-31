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

(define-runtime-path sum-kernel-file "mult.cl")

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

;; (3) TODO: make this driver function more generic by taking data segment (DS),
;; DS refs which are inputs, shape of the result and a kernel source byte string
;; as arguments.
(define (*/opencl t1 t2)
  (let* ([src1-size (size-of (shape t1))]
         [src2-size (size-of (shape t2))]
         [dst-shape (shape t1)]
         [dst-size (size-of dst-shape)]
         [dst-store (new-vec dst-size 0.0)]
         [src1-buf #f]
         [src2-buf #f]
         [dst-buf #f]
         [program #f]
         [kernel #f]
         [event #f])
     (dynamic-wind
       (λ ()
         ;; TODO: extract this operation to a function called tensor->cl-buffer
         (set! src1-buf (clCreateBuffer (context) 'CL_MEM_READ_ONLY
                                       (* (ctype-sizeof _cl_float)
                                          src1-size)
                                       #f))
         (set! src2-buf (clCreateBuffer (context) 'CL_MEM_READ_ONLY
                                        (* (ctype-sizeof _cl_float)
                                           src2-size)
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
         (set! kernel (clCreateKernel program #"Mult"))
         (clSetKernelArg:_cl_mem kernel 0 src1-buf)
         (clSetKernelArg:_cl_mem kernel 1 src2-buf)
         (clSetKernelArg:_cl_mem kernel 2 dst-buf))
       (λ ()
         (set! event (clEnqueueWriteBuffer (command-queue) src1-buf 'CL_FALSE 0
                                           (* (ctype-sizeof _cl_float)
                                              src1-size)
                                           (vec->cpointer (flat-store t1))
                                           (make-vector 0)))
         (set! event (clEnqueueWriteBuffer (command-queue) src2-buf 'CL_FALSE 0
                                           (* (ctype-sizeof _cl_float)
                                              src2-size)
                                           (vec->cpointer (flat-store t2))
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
         (when src1-buf
           (clReleaseMemObject src1-buf))
         (when src2-buf
           (clReleaseMemObject src2-buf))))))

(define (main)
  (let* ([platform (cvector-ref (clGetPlatformIDs:vector) 0)]
         [devices (clGetDeviceIDs:vector platform 'CL_DEVICE_TYPE_GPU)]
         [device 0])
    (parameterize* ([context #f]
                    [command-queue #f])
      (dynamic-wind
        (λ () (initialize devices device))
        (λ ()
          ;; (4) TODO: make this work for tensors of different ranks. Refer to
          ;; ext2 code in flat tensors
          (define t-shape '(10 10 50))
          (printf "Shape of tensors to be multiplied: ~a~n" t-shape)
          (define t1 (random-tensor 0 100 t-shape))
          (define t2 (random-tensor 0 100 t-shape))
          (printf "Timing for CPU computation (in ms):~n")
          (define golden (time (*-ρ t1 t2)))
          (printf "Timing for GPU computation (in ms):~n")
          (define result (time (*/opencl t1 t2)))
          (check-tensor-equal? result golden))
        cleanup))))

(main)
