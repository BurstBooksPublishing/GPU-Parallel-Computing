c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <CL/cl.h>

#define CHECK_CL(err, msg) \
    do { \
        if ((err) != CL_SUCCESS) { \
            fprintf(stderr, "%s: %d\n", (msg), (err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

static const char *kernel_src =
    "__kernel void vec_add(__global const float *a,\n"
    "                      __global const float *b,\n"
    "                      __global float *c) {\n"
    "    size_t i = get_global_id(0);\n"
    "    c[i] = a[i] + b[i];\n"
    "}\n";

static void fill_rand(float *v, size_t n) {
    for (size_t i = 0; i < n; ++i) v[i] = (float)rand() / RAND_MAX;
}

int main(void) {
    const size_t N = 1 << 20;
    const size_t bytes = N * sizeof(float);

    cl_int err;
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_mem bufA, bufB, bufC;

    // Platform & device
    CHECK_CL(clGetPlatformIDs(1, &platform, NULL), "clGetPlatformIDs");
    CHECK_CL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL),
             "clGetDeviceIDs");

    // Context & queue
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    CHECK_CL(err, "clCreateContext");
    queue = clCreateCommandQueueWithProperties(context, device, NULL, &err);
    CHECK_CL(err, "clCreateCommandQueueWithProperties");

    // Program & kernel
    program = clCreateProgramWithSource(context, 1, &kernel_src, NULL, &err);
    CHECK_CL(err, "clCreateProgramWithSource");
    err = clBuildProgram(program, 1, &device, "", NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = malloc(log_size + 1);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        log[log_size] = '\0';
        fprintf(stderr, "Build log:\n%s\n", log);
        free(log);
        CHECK_CL(err, "clBuildProgram");
    }
    kernel = clCreateKernel(program, "vec_add", &err);
    CHECK_CL(err, "clCreateKernel");

    // Buffers
    bufA = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, &err);
    CHECK_CL(err, "clCreateBuffer A");
    bufB = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, &err);
    CHECK_CL(err, "clCreateBuffer B");
    bufC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, NULL, &err);
    CHECK_CL(err, "clCreateBuffer C");

    // Host data
    float *hA = malloc(bytes);
    float *hB = malloc(bytes);
    float *hC = malloc(bytes);
    fill_rand(hA, N);
    fill_rand(hB, N);

    // Transfer & run
    CHECK_CL(clEnqueueWriteBuffer(queue, bufA, CL_TRUE, 0, bytes, hA, 0, NULL, NULL),
             "clEnqueueWriteBuffer A");
    CHECK_CL(clEnqueueWriteBuffer(queue, bufB, CL_TRUE, 0, bytes, hB, 0, NULL, NULL),
             "clEnqueueWriteBuffer B");

    CHECK_CL(clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA), "clSetKernelArg 0");
    CHECK_CL(clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB), "clSetKernelArg 1");
    CHECK_CL(clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufC), "clSetKernelArg 2");

    size_t global = N;
    CHECK_CL(clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, NULL, 0, NULL, NULL),
             "clEnqueueNDRangeKernel");
    CHECK_CL(clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0, bytes, hC, 0, NULL, NULL),
             "clEnqueueReadBuffer");

    // Verify
    int ok = 1;
    for (size_t i = 0; i < N; ++i) {
        if (fabs(hC[i] - (hA[i] + hB[i])) > 1e-5) {
            ok = 0;
            break;
        }
    }
    printf("Result: %s\n", ok ? "PASS" : "FAIL");

    // Cleanup
    free(hA); free(hB); free(hC);
    clReleaseMemObject(bufA);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufC);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    return 0;
}