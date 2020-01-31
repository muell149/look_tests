#include <iostream>
#include <chrono>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>


int main()
{
    cv::Mat w,u,v;

    // Simple matrix
    //cv::Mat A = (cv::Mat_<double>(4,3) << 3, 2, 2, 2, 3, -2, -5, 0, 1, 4, 6, -8);
    
    // Image 
    cv::Mat A = cv::imread("amber.jpg",0);
    cv::Mat_<double> B;

    A.convertTo(B,CV_64F);

    auto start = std::chrono::high_resolution_clock::now();
    cv::SVD::compute(B,w,u,v);
    auto stop = std::chrono::high_resolution_clock::now();
    
    /*
    std::cout<<"A = " << std::endl << " " << A << std::endl << std::endl;
    
    std::cout<<"w = " << std::endl << " " << w << std::endl << std::endl;
    
    std::cout<<"u = " << std::endl << " " << u << std::endl << std::endl;
    
    std::cout<<"v = " << std::endl << " " << v << std::endl << std::endl;
    */

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    std::cout << duration.count() << " microseconds" << std::endl;    

    return 0;
}