#include "OpenCV-GCApplication.h"
#include <time.h>
#include<iomanip>
using namespace std;
using namespace cv;


static void help(char** argv)
{
    cout << "\nThis program demonstrates GrabCut segmentation -- select an object in a region\n"
        "and then grabcut will attempt to segment it out.\n"
        "Call:\n"
        << argv[0] << " <image_name>\n"
        "\nSelect a rectangular area around the object you want to segment\n" <<
        "\nHot keys: \n"
        "\tESC - quit the program\n"
        "\tr - restore the original image\n"
        "\tn - next iteration\n"
        "\n"
        "\tleft mouse button - set rectangle\n"
        "\n"
        "\tCTRL+left mouse button - set GC_BGD pixels\n"
        "\tSHIFT+left mouse button - set GC_FGD pixels\n"
        "\n"
        "\tCTRL+right mouse button - set GC_PR_BGD pixels\n"
        "\tSHIFT+right mouse button - set GC_PR_FGD pixels\n" << endl;
}

GCApplication gcapp; //grab cut应用程序

/** @brief 鼠标事件的回调函数。 见 cv::setMouseCallback
@param 事件 cv::MouseEventTypes 常量之一。
@param x 鼠标事件的 x 坐标。
@param y 鼠标事件的 y 坐标。
@param 标记 cv::MouseEventFlags 常量之一。
@param userdata 可选参数。
  */
static void on_mouse(int event, int x, int y, int flags, void* param)
{
    gcapp.mouseClick(event, x, y, flags, param);
}


int main(int argc, char** argv){
    help(argv);
    string filename = argc >= 2 ? argv[1] : "../../../data/sheep.jpg";
    Mat image = imread(filename, 1);//加载图片
    
    if(image.empty()){
        cout << "\n Durn, couldn't read image filename " << filename << endl;
        return 1;
    }
    // winName 为 sheep
    string winName = filename.substr(filename.find_last_of('/') + 1);
    winName = winName.substr(0, winName.find_last_of('.'));
    namedWindow(winName, WINDOW_NORMAL);
    setMouseCallback(winName, on_mouse, 0);//设置窗口鼠标回调

    gcapp.setImageAndWinName(image, winName);
    gcapp.showImage();

    for(;;)
    {
        char c = (char)waitKey(0);
        switch (c)
        {
        case '\x1b': // \x1B 表示 ASCII 中的第 1B(27号)字符 ESC（Escape）也就是转义字符
            cout << "Exiting ..." << endl;
            goto exit_main; //退出主程序
        case 'r':
            cout << endl;
            gcapp.reset();//重置图像
            gcapp.showImage();
            break;
        case 'n':
            int iterCount = gcapp.getIterCount();
            cout << "\nBegin " << iterCount << " iterations of GrabCut ..." << endl;
            clock_t start, finish;
            start = clock();
            if(gcapp.nextIter() > iterCount){//确定好矩形
                finish = clock();
                gcapp.showImage();
                cout << "Done! It took " << fixed << setprecision(3) << (double)(finish - start) / CLOCKS_PER_SEC << " seconds" << endl;
            }
            else //矩形必须先确定
                cout << "rect must be determined>" << endl;
            break;
        }
    }
    exit_main:
        destroyWindow(winName);
        return 0;
}