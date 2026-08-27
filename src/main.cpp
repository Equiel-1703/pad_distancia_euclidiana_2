#include "cldef.hpp"

#include <iostream>
#include <vector>

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
)";

#define TARGET_DEVICE CL_DEVICE_TYPE_CPU

int main(int argc, char const *argv[])
{
  cl::Platform platform;
  cl::Device device;

  std::vector<cl::Platform> available_platforms;
  cl::Platform::get(&available_platforms);

  std::vector<cl::Device> available_devices;
  for (const cl::Platform &p : available_platforms)
  {
    p.getDevices(TARGET_DEVICE, &available_devices);

    if (!available_devices.empty())
    {
      platform = p;
      device = available_devices.front();
    }
  }

  std::cout << "Platform: " << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;
  std::cout << "Device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;

  return 0;
}
