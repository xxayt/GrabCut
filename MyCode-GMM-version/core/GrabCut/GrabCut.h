#ifndef GRABCUT_H_  // 防止头文件重复包含，多次编译(如果没有定义过GRABCUT_H_，则定义GRABCUT_H_，否则跳过)
#define GRABCUT_H_
#include <opencv2\opencv.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\core\core.hpp>
#include <iostream>
#include "GMM/GMM.h"
# include "maxflow/graph.h"
using namespace cv;
using namespace std;

enum
{  // 枚举在编译阶段将名字替换成对应的值
	GC_WITH_RECT  = 0, 
	GC_WITH_MASK  = 1, 
	GC_CUT        = 2  
};
enum
{
	MUST_BGD = 0,
	MUST_FGD = 1,  // 未用过
	MAYBE_BGD = 2,
	MAYBE_FGD = 3
};
// 展示mask掩码的变化 change: colorCompIndex
static void getcolorCompIndex(const Mat& CompIndex, Mat& colorCompIndex)
{
    if(CompIndex.empty() || CompIndex.type() != CV_32SC1)
        CV_Error(Error::StsBadArg, "CompIndex is empty or has incorrect type (not CV_32SC1)");
    if(colorCompIndex.empty() || colorCompIndex.rows != CompIndex.rows || colorCompIndex.cols != CompIndex.cols)
        colorCompIndex.create(CompIndex.size(), CV_8UC3);  // CV_8UC3: Vec3b = vector<uchar, 3>
    for(int i = 0; i < CompIndex.rows; i++){
        for(int j = 0; j < CompIndex.cols; j++){
            if(CompIndex.at<signed int>(i, j) == 0)  // 0表示MUST_BGD，用红色
                colorCompIndex.at<Vec3b>(i, j) = Vec3b(0, 0, 255);
            else if(CompIndex.at<signed int>(i, j) == 1)  // 1表示MUST_FGD，用黄色
                colorCompIndex.at<Vec3b>(i, j) = Vec3b(0, 255, 255);
            else if(CompIndex.at<signed int>(i, j) == 2)  // 2表示MAYBE_BGD，用蓝色
                colorCompIndex.at<Vec3b>(i, j) = Vec3b(255, 0, 0);
            else if(CompIndex.at<signed int>(i, j) == 3)  // 3表示MAYBE_FGD，用绿色
                colorCompIndex.at<Vec3b>(i, j) = Vec3b(0, 255, 0);
            else if(CompIndex.at<signed int>(i, j) == 4)  // 4表示MAYBE_FGD，用粉色
                colorCompIndex.at<Vec3b>(i, j) = Vec3b(255, 0, 255);
        }
    }
}
class GrabCutSegmentation
{
    public:
        static constexpr double gamma = 50.0;
        // 主函数
        void GrabCut(InputArray arrayimg, InputOutputArray arraymask, Rect rect,
                    InputOutputArray bgdModel,InputOutputArray fgdModel,
                    int iterCount, int mode);
        
        // fun(const A& a); ——引用的a对象实例本身及其成员变量值不可改变
        void initMaskWithRect(Mat& mask, Size imgSize, Rect rect);

        // step0. Initialize GMMs by kmeans (change: bgdGMM, fgdGMM)
        void initGMMs(const Mat& img, const Mat& mask, GMM& bgdGMM, GMM& fgdGMM);
        // step1. Assign GMM Components to pixels: for each n in Tu (change: partIndex)
        void AssignGMMComponents(const Mat& img, const Mat& mask, const GMM& bgdGMM, const GMM& fgdGMM, Mat& partIndex);
        // step2. Learn GMM Parameters from data z (change: bgdGMM, fgdGMM)
        void LearnGMMParameters(const Mat& img, const Mat& mask, GMM& bgdGMM, GMM& fgdGMM, const Mat& partIndex);
        // step2-1. Calculate Beta (change: beta)
        double CalcBeta(const Mat& img);
        // step2-2. Calculate Smoothness Energy (change: leftW, upleftW, upW, uprightW)
        void CalcSmoothness(const Mat& img, const double beta, const double gamma, 
                            Mat& leftW, Mat& upleftW, Mat& upW, Mat& uprightW);
        // step2-3. Build Graph based on edge-weights (change: graph)
        void getGraph(const Mat& img, const Mat& mask, 
                    const GMM& bgdGMM, const GMM& fgdGMM,
                    const Mat& leftW, const Mat& upleftW, const Mat& upW, const Mat& uprightW, 
                    double lambda, Graph<double, double, double>& graph);
        // step3. Estimate Segmentation: use min cut to solve (change: mask)
        void EstimateSegmentation(Graph<double, double, double>& graph, Mat& mask);
    private:
        void CalcEneryFunction(const Graph<double, double, double>& graph, const Mat& mask, 
                                const Mat& leftW, const Mat& upleftW, const Mat& upW, const Mat& uprightW);
        vector<double> tweight_wSource, tweight_wSink;
};
#endif




