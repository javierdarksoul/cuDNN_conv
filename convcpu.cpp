#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <time.h>       /* clock_t, clock, CLOCKS_PER_SEC */
#include <omp.h>
using namespace cv;

Mat convolucion_sec(Mat inputIm, Mat kernel) {
	Mat outputIm = cv::Mat::zeros(inputIm.rows, inputIm.cols, CV_16SC1);
	for (int i = 0; i < inputIm.rows; i = i + 1) { //RECORRIDO DE FILAS DE LA MATRIZ
		for (int j = 0; j < inputIm.cols; j = j + 1) { //RECORRIDO DE COLUMNAS DE LA IMAGEN
			double sum = 0;
			for (int k = 0; k < kernel.rows; k = k + 1) { //RECORRIDO DE FILAS DEL NUCLEO DE CONV.
				for (int l = 0; l < kernel.cols; l = l + 1) { //RECORRIDO DE COLUMNAS DEL NUCLEO DE CONV.
					if ((i < 1) || (j < 1) || (i + 1 > inputIm.rows)
							|| (j + 1 > inputIm.cols)) {
						outputIm.at<short>(i, j) = 0;
					} else {
						sum += inputIm.at<uchar>(i - 1 + k, j - 1 + l) // uchar - imagen origen
								* kernel.at<double>(k, l);             // double - kernel
					}

				}
			}
			outputIm.at<short>(i, j) = sum; // 16sc1 es un short

		}
	}
	return (outputIm);
}
Mat convolucion_parallel(Mat inputIm, Mat kernel) {
	Mat outputIm = cv::Mat::zeros(inputIm.rows, inputIm.cols, CV_16SC1);
    #pragma omp parallel for
	for (int i = 0; i < inputIm.rows; i = i + 1) { //RECORRIDO DE FILAS DE LA MATRIZ
		for (int j = 0; j < inputIm.cols; j = j + 1) { //RECORRIDO DE COLUMNAS DE LA IMAGEN
			double sum = 0;
			for (int k = 0; k < kernel.rows; k = k + 1) { //RECORRIDO DE FILAS DEL NUCLEO DE CONV.
				for (int l = 0; l < kernel.cols; l = l + 1) { //RECORRIDO DE COLUMNAS DEL NUCLEO DE CONV.
					if ((i < 1) || (j < 1) || (i + 1 > inputIm.rows)
							|| (j + 1 > inputIm.cols)) {
						outputIm.at<short>(i, j) = 0;
					} else {
						sum += inputIm.at<uchar>(i - 1 + k, j - 1 + l) // uchar - imagen origen
								* kernel.at<double>(k, l);             // double - kernel
					}
				}
			}
			outputIm.at<short>(i, j) = sum; // 16sc1 es un short

		}
	}
	
	return (outputIm);
}

using namespace std;

int main(int argc, char const *argv[]) {
    if(argc!=3){
        printf("ejecute como /convcpu mode nt\n mode\n 1-secuencial\n 2-paralelo\n nt=num hilos\n");
        return EXIT_SUCCESS;
    }
    int mode=atoi(argv[1]);
    int nt=atoi(argv[2]);
    omp_set_num_threads(4);
	Mat image, gray_image,salida;
	Mat kernel = (Mat_<double>(3, 3) << 1, 1, 1, 1, -8, 1, 1, 1, 1);
	image = cv::imread("/home/javier/Imágenes/vino.jpg");
	cvtColor(image, gray_image, cv::COLOR_BGR2GRAY); //conversion a escala de grises
    double start,end;
	switch (mode)
	{
	case 1:
		start=omp_get_wtime();
		salida = convolucion_sec(gray_image, kernel);
    	end=omp_get_wtime();
		cout<<"tiempo de ejecucion secuencial: "<< end-start<<" segundos"<<endl;
		break;
		
	case 2:
		start=omp_get_wtime();
		salida = convolucion_parallel(gray_image, kernel);
    	end=omp_get_wtime();
		cout<<"tiempo de ejecucion paralelo: "<< end-start<<" segundos"<<endl;
		break;
	}
    
    
	salida.convertTo(salida, CV_8UC1); // se pone en unsigned chaar otra vez
	cv::imwrite("output_cpu.png",salida);
	return EXIT_SUCCESS;

}