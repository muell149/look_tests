#include <iostream>
#include <chrono>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <torch/torch.h>

int main() {

  torch::Tensor tensor = torch::rand({2, 3});
  std::cout << tensor << std::endl;

  // cv::Mat w,u,v;

  // Simple matrix
  // cv::Mat A = (cv::Mat_<double>(4,3) << 3., 2., 2., 2., 3., -2., -5., 0., 1., 4., 6., -8.);
  
  // Image 
  // cv::Mat A = cv::imread("images/amber.jpg",0);
  // std::cout << A << std::endl;
  // cv::Mat_<double> B;

  // std::vector< cv::String > fn;
  // std::vector< cv::Mat > im;
  // std::vector< cv::Mat_<double> > dat_im;

  // cv::glob("images/*.jpg",fn,true);
  // for (size_t k = 0; k < fn.size(); ++k){
  //     im.push_back(cv::imread(fn[k],0));
  //     dat_im.push_back(cv::imread(fn[k],0));
  // }

  // auto start = std::chrono::high_resolution_clock::now();
  // cv::SVD::compute(B,w,u,v);
  // auto stop = std::chrono::high_resolution_clock::now();

  // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

  // std::cout << duration.count() << " microseconds" << std::endl;    



}