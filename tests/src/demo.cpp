#include <iostream>
#include <opencv2/core.hpp>

int main()
{
    cv::Mat C = (cv::Mat_<double>(3,3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
    std::cout<<"C = " << std::endl << " " << C << std::endl << std::endl;
    return 0;
}