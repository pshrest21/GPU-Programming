//Vector Addition

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define DEVICE_NAME_LEN 128
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cl/opencl.h>
#include <cl/cl.h>

static char dev_name[DEVICE_NAME_LEN];

const char* kernelSource =                  "\n"\
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable \n"\
"__kernel void vecAdd(__global double *a,    \n"\
"                     __global double *b,    \n"\
"                     __global double *c,    \n"\
"                     const unsigned int n){ \n"\
"int id = get_global_id(0);                  \n"\
"if(id < n )                                 \n"\
"   c[id] = a[id] + b[id];                   \n"\
"}                                           \n"\
"                                            \n";

int main()
{
    //Length of vectors
    unsigned int n = 100000;

    //Host input vectors
    double* h_a;
    double* h_b;
    //Host output vector
    double* h_c;

    //Device input buffers
    cl_mem bufferA;
    cl_mem bufferB;
    //Device output buffer
    cl_mem bufferC;
    cl_uint platformCount;
    cl_platform_id platforms;
    cl_device_id device_id;
    cl_uint ret_num_devices;
    cl_context context;
    cl_command_queue command_queue;
    cl_program my_program;
    cl_kernel my_kernel;
    size_t localSize, globalSize;
    cl_int ret;

    //Size in bytes, of each vector
    size_t bytes = n * sizeof(double);


    //Allocate memory for each vector on host
    h_a = (double*)malloc(bytes);
    h_b = (double*)malloc(bytes);
    h_c = (double*)malloc(bytes);

    //Initialize vectors on host

    for (int i = 0; i < n; i++) {
        h_a[i] = 2;
        h_b[i] = 4;
        //h_a[i] = sinf(i) * sinf(i);
        //h_b[i] = cosf(i) * cosf(i);
    }

    //Bind to platform
    ret = clGetPlatformIDs(1, &platforms, NULL);

    //Allocate space to store platform IDs
    //platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * platformCount);

    //Get Platform IDs
    //clGetPlatformIDs(platformCount, platforms, NULL);

    //Query for available OpenCL device
    ret = clGetDeviceIDs(platforms, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
    //Optional- To get Information of the device to print it for user
    //ret = clGetDeviceInfo(device_id, CL_DEVICE_NAME, DEVICE_NAME_LEN, dev_name, NULL);
    //printf("device name= %s\n", dev_name);

    //Create OpenCL context
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &ret);

    //Create Command Queue
    command_queue = clCreateCommandQueue(context, device_id, 0, &ret);

    //Declare Buffers (input and output arrays) in the device memory
    bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
    bufferB = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
    bufferC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, NULL, NULL);


    //Copy the host input vectors to the device of respective buffer memory
    ret = clEnqueueWriteBuffer(command_queue, bufferA, CL_TRUE, 0, bytes, h_a, 0, NULL, NULL);
    ret |= clEnqueueWriteBuffer(command_queue, bufferB, CL_TRUE, 0, bytes, h_b, 0, NULL, NULL);

   
    //Create a program
    my_program = clCreateProgramWithSource(context, 1, (const char**)&kernelSource, NULL, &ret);

    //Compile(Build) the program
    ret = clBuildProgram(my_program, 0, NULL, NULL, NULL, NULL);    

    //Create the kernel
    my_kernel = clCreateKernel(my_program, "vecAdd", &ret);

    //Set Kernel Arguments
    ret = clSetKernelArg(my_kernel, 0, sizeof(cl_mem), &bufferA);
    ret |= clSetKernelArg(my_kernel, 1, sizeof(cl_mem), &bufferB);
    ret |= clSetKernelArg(my_kernel, 2, sizeof(cl_mem), &bufferC);
    ret |= clSetKernelArg(my_kernel, 3, sizeof(unsigned int), &n);

    //Important Step!!
    //Set the Local and Global workgroup sizes
    //..............Still a little confused on how to fix the sizes....................
    localSize = 64;
    globalSize = ceil(n / (float)localSize) * localSize;

    //Execute the kernel
    ret = clEnqueueNDRangeKernel(command_queue, my_kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);

    //Wait for the command queue to get serviced before reading back results
    clFinish(command_queue);

    //Obtain Results
    //Read the Output data back to the host

    ret = clEnqueueReadBuffer(command_queue, bufferC, CL_TRUE, 0, bytes, h_c, 0, NULL, NULL);

    //Sum up the vector c and print result divided by n, this should equal 1
    double sum = 0;
    for (int i = 0; i < n; i++) {
        sum += h_c[i];
    }
    printf("final result: %f", sum / n);
  
    //Release OpenCL resources
    clReleaseMemObject(bufferA);
    clReleaseMemObject(bufferB);
    clReleaseMemObject(bufferC);
    clReleaseProgram(my_program);
    clReleaseKernel(my_kernel);
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);

    //Release host memory
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
