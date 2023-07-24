// #pragma once
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "GrabCut/GrabCut.h"
#include "BorderMatting/BorderMatting.h"
#include <iostream>
using namespace std;
using namespace cv;

//颜色//
const Scalar RED = Scalar(0, 0, 255);
const Scalar PINK = Scalar(230, 130, 255);
const Scalar BLUE = Scalar(255, 0, 0);
const Scalar LIGHTBLUE = Scalar(255, 255, 160);
const Scalar GREEN = Scalar(0, 255, 0);

//  获取二值掩码 change: binMask
static void getBinMask(const Mat& comMask, Mat& binMask)
{
    if(comMask.empty() || comMask.type() != CV_8UC1)
        CV_Error(Error::StsBadArg, "comMask is empty or has incorrect type (not CV_8UC1)");
    if(binMask.empty() || binMask.rows != comMask.rows || binMask.cols != comMask.cols)
        binMask.create(comMask.size(), CV_8UC1);
    // comMask可能为0,1,2,3
    // 0&1 = 2&1 = 0. 0表示MUST_BGD，2表示MAYBE_BGD
    // 1&1 = 3&1 = 1. 1表示MUST_FGD，3表示MAYBE_FGD
    binMask = comMask & 1;
}
// 展示mask掩码的变化 change: colorMask
static void getcolormask(const Mat& comMask, Mat& colorMask)
{
    if(comMask.empty() || comMask.type() != CV_8UC1)
        CV_Error(Error::StsBadArg, "comMask is empty or has incorrect type (not CV_8UC1)");
    if(colorMask.empty() || colorMask.rows != comMask.rows || colorMask.cols != comMask.cols)
        colorMask.create(comMask.size(), CV_8UC3);  // CV_8UC3: Vec3b = vector<uchar, 3>
    for(int i = 0; i < comMask.rows; i++){
        for(int j = 0; j < comMask.cols; j++){
            if(comMask.at<uchar>(i, j) == 0)  // 0表示MUST_BGD，用黑色
                colorMask.at<Vec3b>(i, j) = Vec3b(0, 0, 0);
            else if(comMask.at<uchar>(i, j) == 1)  // 1表示MUST_FGD，用白色
                colorMask.at<Vec3b>(i, j) = Vec3b(255, 255, 255);
            else if(comMask.at<uchar>(i, j) == 2)  // 2表示MAYBE_BGD，用灰色
                colorMask.at<Vec3b>(i, j) = Vec3b(128, 128, 128);
            else if(comMask.at<uchar>(i, j) == 3)  // 3表示MAYBE_FGD，用浅粉色
                colorMask.at<Vec3b>(i, j) = Vec3b(255, 192, 203);
        }
    }
}
//grab cut应用程序类//
class GCApplication
{
public:
    enum{
        NOT_SET = 0,
        IN_PROCESS = 1,
        SET = 2
    };
    static const int radius = 2;
    static const int thickness = -1;//实心//
    // 重置所有变量状态
    void reset();
    // 设置图像和窗口名
    void setImageAndWinName(const Mat& _image, const string& _winName);
    // 显示图像
    void showImage() const;
    Mat saveRect() const;
    // 鼠标单击事件
    void mouseClick(int event, int x, int y, int flags, void* param);
    // 运行 GrabCut 算法
    int nextIter();
    void borderMatting();
    int getIterCount() const { return iterCount; }
    string getWinName() const { return (*winName); }
    
private:
    // 设置掩码里矩形框像素值
    void setRectInMask();
    // 在掩码上绘制标记点
    void setLblsInMask(int flags, Point p, bool isPr);

    const string* winName;
    const Mat* image;
    Mat mask, alphaMask;
    Mat bgdModel, fgdModel;

    uchar rectState, lblsState, prLblsState;  // 
    bool isInitialized;

    Rect rect;
    vector<Point> fgdPxls, bgdPxls, prFgdPxls, prBgdPxls;
    int iterCount;
    GrabCutSegmentation gc;
    BorderMatting bm;
};