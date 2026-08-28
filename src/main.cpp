#include "cldef.hpp"

#include <iostream>
#include <vector>
#include <chrono>
#include <cstring>

std::string opencl_code = R"(
float calculate_euclidean(__global float *vec_1, __global float *vec_2, int dim)
{
	float res = 0.0;

  for(int i=0; i < dim; i++){
    float diff = (vec_2[i] - vec_1[i]);
    res = (res + (diff * diff));
  }

  return (sqrt(res));
}


__kernel void euclidean_distance_kernel(__global float *q, __global float *x, int len_x, int num_dims, __global float *arr_results)
{
	int g_idx = get_global_id(0);
  if((g_idx < len_x))
  {
    int x_idx = (g_idx * num_dims);
    arr_results[g_idx] = calculate_euclidean(q, (x + x_idx), num_dims);
  }
}

)";

#define TARGET_DEVICE CL_DEVICE_TYPE_CPU
#define LEN_X 3'584

int main(int argc, char const *argv[])
{
  if (argc != 2 && argc != 3)
  {
    std::cout << "Usage: " << argv[0] << " NUM_DIMS [-f|--fast]" << std::endl;

    return 1;
  }

  const int num_dims = std::stoi(argv[1]);
  std::string compile_options = "";

  if (argc == 3)
  {
    if (strcmp(argv[2], "-f") == 0 || strcmp(argv[2], "--fast") == 0)
    {
      compile_options = "-cl-fast-relaxed-math";
    }
  }

  cl::Platform platform;
  cl::Device device;
  cl::Context context;
  cl::CommandQueue queue;

  std::vector<cl::Platform> available_platforms;
  std::vector<cl::Device> available_devices;

  cl::Platform::get(&available_platforms);
  for (const cl::Platform &p : available_platforms)
  {
    p.getDevices(TARGET_DEVICE, &available_devices);

    if (!available_devices.empty())
    {
      platform = p;
      device = available_devices.front();
      context = cl::Context(device);
      queue = cl::CommandQueue(context);

      break;
    }
  }

  // Check SVM capabilities
  cl_device_svm_capabilities svm_cap = device.getInfo<CL_DEVICE_SVM_CAPABILITIES>();

  if (!(svm_cap & CL_DEVICE_SVM_COARSE_GRAIN_BUFFER))
  {
    std::cerr << "[Error] This device does not support coarse-grained SVM Buffers." << std::endl;
  }

  // Creating and compiling program

  cl::Program program(context, opencl_code);

  try
  {
    program.build(device, compile_options.c_str());
  }
  catch (const cl::Error &e)
  {
    std::string log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
    std::cerr << "Build log [" << device.getInfo<CL_DEVICE_NAME>() << "]: " << log << std::endl;
  }
  catch (const std::exception &e)
  {
    std::cerr << e.what() << '\n';
  }

  cl::Kernel kernel(program, "euclidean_distance_kernel");

  // Allocating memory
  size_t q_array_size = sizeof(float) * num_dims;
  size_t x_set_size = sizeof(float) * LEN_X * num_dims;
  size_t results_size = sizeof(float) * LEN_X;

  void *q_array = clSVMAlloc(
      context(),
      CL_MEM_READ_WRITE,
      q_array_size,
      0);

  void *x_set = clSVMAlloc(
      context(),
      CL_MEM_READ_WRITE,
      x_set_size,
      0);

  void *results = clSVMAlloc(
      context(),
      CL_MEM_READ_WRITE,
      results_size,
      0);

  // Filling them

  queue.enqueueMapSVM(q_array, CL_FALSE, CL_MAP_WRITE, q_array_size);
  queue.enqueueMapSVM(x_set, CL_TRUE, CL_MAP_WRITE, x_set_size);

  for (uint i = 0; i < num_dims; i++)
  {
    reinterpret_cast<float *>(q_array)[i] = 1.0f;
  }

  for (uint i = 0; i < LEN_X; i++)
  {
    for (uint j = 0; j < num_dims; j++)
    {
      reinterpret_cast<float *>(x_set)[i * num_dims + j] = 1.0f * (i + 1);
    }
  }

  queue.enqueueUnmapSVM(q_array);
  queue.enqueueUnmapSVM(x_set);

  // Launch kernel

  kernel.setArg(0, q_array);
  kernel.setArg(1, x_set);
  kernel.setArg(2, LEN_X);
  kernel.setArg(3, num_dims);
  kernel.setArg(4, results);

  cl::NDRange global(LEN_X);

  auto start_t = std::chrono::steady_clock::now();

  queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange);
  queue.finish();

  auto end_t = std::chrono::steady_clock::now();

  // Print results

  std::chrono::duration<double, std::milli> duration_millis = (end_t - start_t);
  std::cout << "T = " << num_dims << " | " << duration_millis.count() << "ms" << std::endl;

  // Cleanup

  clSVMFree(context(), q_array);
  clSVMFree(context(), x_set);
  clSVMFree(context(), results);

  return 0;
}
