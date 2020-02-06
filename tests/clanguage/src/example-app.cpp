#include <iostream>
#include <chrono>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <torch/torch.h>

int main() {

  // Simple tensor with Libtorch
  torch::Tensor tensor = torch::rand({2, 3});
  std::cout << tensor << std::endl;

  // Simple matrix with OpenCV
  cv::Mat A = (cv::Mat_<double>(4,3) << 3., 2., 2., 2., 3., -2., -5., 0., 1., 4., 6., -8.);
  std::cout << A << std::endl;

}