c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <CL/cl.h>

#define CHECK_CL(ERR, OP) do {                          \
    if ((ERR) != CL_SUCCESS) {                          \
        fprintf(stderr, "OpenCL error %d at %s:%d (%s)\n", \
                (int)(ERR), __FILE__, __LINE__, (OP));  \
        exit(EXIT_FAILURE);                             \
    }                                                   \
} while (0)

static const char *kernel_src =
"__kernel void square(__global const float *in, __global float *out) {\n"
"    size_t gid = get_global_id(0);\n"
"    out[gid] = in[gid] * in[gid];\n"
"}\n";

int main(void) {
    const size_t N = 1 << 20;
    cl_int err;

    float *h_in  = aligned_alloc(64, N * sizeof(float));
    float *h_out = aligned_alloc(64, N * sizeof(float));
    if (!h_in || !h_out) { perror("aligned_alloc"); exit(EXIT_FAILURE); }

    for (size_t i = 0; i < N; ++i) h_in[i] = (float)i;

    cl_uint num_platforms;
    err = clGetPlatformIDs(0, NULL, &num_platforms);
    CHECK_CL(err, "clGetPlatformIDs");
    if (!num_platforms) { fputs("No OpenCL platform\n", stderr); exit(EXIT_FAILURE); }

    cl_platform_id platform;
    err = clGetPlatformIDs(1, &platform, NULL);
    CHECK_CL(err, "clGetPlatformIDs");

    cl_uint num_devices;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
    CHECK_CL(err, "clGetDeviceIDs");
    if (!num_devices) { fputs("No GPU device\n", stderr); exit(EXIT_FAILURE); }

    cl_device_id device;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    CHECK_CL(err, "clGetDeviceIDs");

    cl_context ctx = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    CHECK_CL(err, "clCreateContext");

    cl_command_queue q = clCreateCommandQueue(ctx, device,
                                              CL_QUEUE_PROFILING_ENABLE, &err);
    CHECK_CL(err, "clCreateCommandQueue");

    cl_program prog = clCreateProgramWithSource(ctx, 1, &kernel_src, NULL, &err);
    CHECK_CL(err, "clCreateProgramWithSource");

    err = clBuildProgram(prog, 1, &device, "-cl-fast-relaxed-math", NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = malloc(log_size);
        clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        fprintf(stderr, "Build log:\n%s\n", log);
        free(log);
        exit(EXIT_FAILURE);
    }

    cl_kernel k = clCreateKernel(prog, "square", &err);
    CHECK_CL(err, "clCreateKernel");

    cl_mem d_in  = clCreateBuffer(ctx, CL_MEM_READ_ONLY,  N * sizeof(float), NULL, &err);
    CHECK_CL(err, "clCreateBuffer");
    cl_mem d_out = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, N * sizeof(float), NULL, &err);
    CHECK_CL(err, "clCreateBuffer");

    cl_event write_ev;
    err = clEnqueueWriteBuffer(q, d_in, CL_FALSE, 0, N * sizeof(float), h_in,
                               0, NULL, &write_ev);
    CHECK_CL(err, "clEnqueueWriteBuffer");

    const size_t local  = 64;
    const size_t global = ((N + local - 1) / local) * local;

    err  = clSetKernelArg(k, 0, sizeof(cl_mem), &d_in);
    err |= clSetKernelArg(k, 1, sizeof(cl_mem), &d_out);
    CHECK_CL(err, "clSetKernelArg");

    cl_event kernel_ev;
    err = clEnqueueNDRangeKernel(q, k, 1, NULL, &global, &local,
                                 1, &write_ev, &kernel_ev);
    CHECK_CL(err, "clEnqueueNDRangeKernel");

    cl_event read_ev;
    err = clEnqueueReadBuffer(q, d_out, CL_FALSE, 0, N * sizeof(float), h_out,
                              1, &kernel_ev, &read_ev);
    CHECK_CL(err, "clEnqueueReadBuffer");

    clWaitForEvents(1, &read_ev);

    cl_ulong t0, t1;
    clGetEventProfilingInfo(kernel_ev, CL_PROFILING_COMMAND_START, sizeof(t0), &t0, NULL);
    clGetEventProfilingInfo(kernel_ev, CL_PROFILING_COMMAND_END,   sizeof(t1), &t1, NULL);
    printf("Kernel time: %.3f ms\n", (t1 - t0) * 1e-6);

    clReleaseEvent(write_ev);
    clReleaseEvent(kernel_ev);
    clReleaseEvent(read_ev);
    clReleaseMemObject(d_in);
    clReleaseMemObject(d_out);
    clReleaseKernel(k);
    clReleaseProgram(prog);
    clReleaseCommandQueue(q);
    clReleaseContext(ctx);
    free(h_in);
    free(h_out);
    return 0;
}