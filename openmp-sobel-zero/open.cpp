#include <iostream>
#include <stdio.h>

#include <omp.h>
#include <time.h>

// ------ OpenCV includes ------
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

// dimension of kernel
int x[3][3];
int y[3][3];

/*----- OpenMP -----*/
int num_of_threads = 2;
int i, j;
double start, t_end;

int main(int argc, char **argv)
{

  Mat initialImage = imread(argv[1], 0); // imread gray-scale image
  Mat finalImage = Mat::zeros(initialImage.size(), initialImage.type());

  if (finalImage.type() == initialImage.type())
  {
    cout << "YES" << endl;
  }

  if (argc != 2 || !initialImage.data)
  {
    cout << "No image data or Usage: ./sobel imagePath" << endl;
    return -1;
  }
  else
    cout << "Image OK!" << endl;

  // x direction
  x[0][0] = -1; x[0][1] = 0; x[0][2] = 1;
  x[1][0] = -2; x[1][1] = 0; x[1][2] = 2;
  x[2][0] = -1; x[2][1] = 0; x[2][2] = 1;

  // y direction
  y[0][0] = -1; y[0][1] = -2; y[0][2] = -1;
  y[1][0] = 0; y[1][1] = 0; y[1][2] = 0;
  y[2][0] = 1; y[2][1] = 2; y[2][2] = 1;

  num_of_threads = 4;
  omp_set_num_threads(num_of_threads);

  start = omp_get_wtime();
  #pragma omp parallel for private(i)
  for (j = 0; j < initialImage.rows - 2; j++)
  {
    for (i = 0; i < initialImage.cols - 2; i++)
    {
      // kernel x

      int xValOfPixel =
          (x[0][0] * (int)initialImage.at<uchar>(j, i)) + (x[0][1] * (int)initialImage.at<uchar>(j + 1, i)) + (x[0][2] * (int)initialImage.at<uchar>(j + 2, i)) +
          (x[1][0] * (int)initialImage.at<uchar>(j, i + 1)) + (x[1][1] * (int)initialImage.at<uchar>(j + 1, i + 1)) + (x[1][2] * (int)initialImage.at<uchar>(j + 2, i + 1)) +
          (x[2][0] * (int)initialImage.at<uchar>(j, i + 2)) + (x[2][1] * (int)initialImage.at<uchar>(j + 1, i + 2)) + (x[2][2] * (int)initialImage.at<uchar>(j + 2, i + 2));

      // kernel y
      int yValOfPixel =
          (y[0][0] * (int)finalImage.at<uchar>(j, i)) + (y[0][1] * (int)finalImage.at<uchar>(j + 1, i)) + (y[0][2] * (int)finalImage.at<uchar>(j + 2, i)) +
          (y[1][0] * (int)finalImage.at<uchar>(j, i + 1)) + (y[1][1] * (int)finalImage.at<uchar>(j + 1, i + 1)) + (y[1][2] * (int)finalImage.at<uchar>(j + 2, i + 1)) +
          (y[2][0] * (int)finalImage.at<uchar>(j, i + 2)) + (y[2][1] * (int)finalImage.at<uchar>(j + 1, i + 2)) + (y[2][2] * (int)finalImage.at<uchar>(j + 2, i + 2));

      int sum = abs(xValOfPixel) + abs(yValOfPixel);
      if (sum > 255)
        sum = 255;

      finalImage.at<uchar>(j, i) = (uchar)sum;
    }
  }
  t_end = omp_get_wtime();
  cout << "Time: " << t_end - start << endl;
  namedWindow("Minha Imagem", WINDOW_NORMAL);
  resizeWindow("Minha Imagem", 600, 600);
  imshow("Minha Imagem", finalImage);
  waitKey(0);
}