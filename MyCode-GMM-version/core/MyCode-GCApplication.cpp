#include <vector>
#include "MyCode-GCApplication.h"

// 重置
void GCApplication::reset()
{
    if(!mask.empty())
        mask.setTo(Scalar::all(GC_BGD));//黑色掩码
    bgdPxls.clear(); fgdPxls.clear();
    prBgdPxls.clear();  prFgdPxls.clear();

    isInitialized = false;
    rectState = NOT_SET;  //矩形框状态未设置/
    lblsState = NOT_SET;
    prLblsState = NOT_SET;
    iterCount = 0;
}
//设置图像和窗口名
void GCApplication::setImageAndWinName(const Mat& _image, const string& _winName)
{
    if(_image.empty() || _winName.empty())
        return;
    image = &_image;
    winName = &_winName;
    cout << "winName: " << *winName << endl;
    //创建掩码
    mask.create(image->size(), CV_8UC1);  // CV_8UC1: uchar = unsigned char
    reset(); //重置图像
}
//Show image with border matting
void GCApplication::borderMatting()
{
	bm.borderMatting(*image, mask, alphaMask);
}
//显示图像
void GCApplication::showImage() const 
{
    if(image->empty() || winName->empty())
        return;
    // 从image和mask中获取结果图像res
    Mat res, binMask, colorMask;
    if(!isInitialized){  //未选定矩形
        image->copyTo(res);  // res: image内容
    }
    else{
        getBinMask(mask, binMask); // 获取二值掩码
        image->copyTo(res, binMask); // res: 将binMask和image重叠，重叠部分保留image像素值，其余部分为0
    }
    // image->copyTo(res);  // res: image内容
    // getcolormask(mask, colorMask);  // 获取彩色掩码
    // imshow("colormask", colorMask);

    vector<Point>::const_iterator it;
    for(it = bgdPxls.begin(); it != bgdPxls.end(); ++it)//背景像素点：蓝色小圆圈  
        circle(res, *it, radius, BLUE, thickness);// 
    for(it = fgdPxls.begin(); it != fgdPxls.end(); ++it)//前景像素点：红色小圆圈
        circle(res, *it, radius, RED, thickness);//
    for(it = prBgdPxls.begin(); it != prBgdPxls.end(); ++it)//浅蓝色点:鼠标右键ctrl  背景点
        circle(res, *it, radius, LIGHTBLUE, thickness);//
    for(it = prFgdPxls.begin(); it != prFgdPxls.end(); ++it)//粉色点
        circle(res, *it, radius, PINK, thickness);//

    if(rectState == IN_PROCESS || rectState == SET)
        rectangle(res, Point(rect.x, rect.y), Point(rect.x + rect.width, rect.y + rect.height), GREEN, 2);//绘制绿色矩形框//

    imshow(*winName, res);
}
Mat GCApplication::saveRect() const
{
    Mat res, binMask;
    getBinMask(mask, binMask); // 获取二值掩码
    image->copyTo(res, binMask); // res: 将binMask和image重叠，重叠部分保留image像素值，其余部分为0
    Mat roiImg;
    if(rectState == SET){
        Mat roiImg = res(rect);  // 获取矩形区域
        rectangle(roiImg, Point(0, 0), Point(rect.width, rect.height), GREEN, 2);//绘制绿色矩形框//
        imshow("roiImg", roiImg);
        return roiImg;
    }
    return roiImg;
}
// 矩形框外设为GC_BGD，框内设为GC_PR_FGD
void GCApplication::setRectInMask()
{
    CV_Assert(!mask.empty());
    mask.setTo(GC_BGD);//黑色掩码
    rect.x = max(0, rect.x); //确保矩形左上角点的x>0
    rect.y = max(0, rect.y);//确保矩形左上角点的y>0
    rect.width = min(rect.width, image->cols - rect.x);//确保矩形右下角在image上
    rect.height = min(rect.height, image->rows - rect.y);
    (mask(rect)).setTo(Scalar(GC_PR_FGD));//掩码矩形区域设置为像素值3
}
// 针对鼠标左键或右键，设置掩码mask
void GCApplication::setLblsInMask(int flags, Point p, bool isPr)
{
    vector<Point>* bpxls, * fpxls;//背景和前景像素点集  指针
    uchar bvalue, fvalue;//背景和前景像素点的值
    if(!isPr){  // 鼠标左键模式(确定的点)
        bpxls = &bgdPxls;
        fpxls = &fgdPxls;
        bvalue = GC_BGD;
        fvalue = GC_FGD;
    }
    else{  //鼠标右键模式(概率的点)
        bpxls = &prBgdPxls;//获取点集合指针
        fpxls = &prFgdPxls;
        bvalue = GC_PR_BGD;//背景点目标像素值
        fvalue = GC_PR_FGD;//前景点集目标像素值
    }
    if(flags & EVENT_FLAG_CTRLKEY){  //ctrl按下
        bpxls->push_back(p);//背景点
        circle(mask, p, radius, bvalue, thickness);//绘制圆点
    }
    if(flags & EVENT_FLAG_SHIFTKEY){  //shift按下
        fpxls->push_back(p);//前景点
        circle(mask, p, radius, fvalue, thickness);//绘制圆点
    }
}
//鼠标单击事件
void GCApplication::mouseClick(int event, int x, int y, int flags, void*)
{
    // TODO add bad args check
    switch (event){
        case EVENT_LBUTTONDOWN: //左键按下 set rect or GC_BGD(GC_FGD) labels
        {
            bool isb = (flags & EVENT_FLAG_CTRLKEY) != 0;  // ctrl按下表示背景
            bool isf = (flags & EVENT_FLAG_SHIFTKEY) != 0;  // shift按下表示前景
            if(rectState == NOT_SET && !isb && !isf){
                rectState = IN_PROCESS;  //矩形框模式--左键按下
                rect = Rect(x, y, 1, 1);  //初始矩形宽1 高1
            }
            if((isb || isf) && rectState == SET)//没按ctrl,没按shift， 矩形绘制完毕。
                lblsState = IN_PROCESS;//切换到标签模式 
        }
        break;
        case EVENT_RBUTTONDOWN: //右键按下  set GC_PR_BGD(GC_PR_FGD) labels
        {
            bool isb = (flags & EVENT_FLAG_CTRLKEY) != 0;  //按ctrl
            bool isf = (flags & EVENT_FLAG_SHIFTKEY) != 0;  //按shift
            if((isb || isf) && rectState == SET)//矩形绘制完毕
                prLblsState = IN_PROCESS;//标签模式 
        }
        break;
        case EVENT_LBUTTONUP://左键弹起
            if(rectState == IN_PROCESS){  //矩形绘制模式
                rect = Rect(Point(rect.x, rect.y), Point(x, y));
                // 针对特定图像固定矩形区域
                if(*winName == "sheep"){
                    rect = Rect(Point(108, 90), Point(484, 395));
                }
                else if(*winName == "bird"){
                    rect = Rect(Point(214, 118), Point(514, 294));
                }
                else if(*winName == "dog"){
                    rect = Rect(Point(95, 261), Point(293, 545));
                }
                else if(*winName == "table"){
                    rect = Rect(Point(158, 27), Point(537, 399));
                }
                rectState = SET;//矩形绘制完毕
                setRectInMask();//设置矩形掩码区域像素值为3
                CV_Assert(bgdPxls.empty() && fgdPxls.empty() && prBgdPxls.empty() && prFgdPxls.empty());//断言：背景或者前景点集合为空
                showImage();
            }
            if(lblsState == IN_PROCESS){  //标记点绘制模式
                setLblsInMask(flags, Point(x, y), false);//绘制标记点：前景或者背景 ，左键或者右键（颜色不同）
                lblsState = SET;
                showImage();//更新显示
            }
            break;
        case EVENT_RBUTTONUP: //右键弹起
            if(prLblsState == IN_PROCESS){  //右键标签点绘制模式
                setLblsInMask(flags, Point(x, y), true);  //true:右键模式
                prLblsState = SET;//右键标记点绘制完毕
                showImage();
            }
            break;
        case EVENT_MOUSEMOVE://鼠标移动
            if(rectState == IN_PROCESS){  //绘制矩形模式
                rect = Rect(Point(rect.x, rect.y), Point(x, y));//更新矩形区域
                CV_Assert(bgdPxls.empty() && fgdPxls.empty() && prBgdPxls.empty() && prFgdPxls.empty());//断言：所有标记点集合为空
                showImage();//显示图像
            }
            else if(lblsState == IN_PROCESS){  //标记点模式(左键)
                setLblsInMask(flags, Point(x, y), false);  //false：左键模式
                showImage();
            }
            else if(prLblsState == IN_PROCESS){  //标记点模式(右键)
                setLblsInMask(flags, Point(x, y), true);  //true: 右键模式
                showImage();
            }
            break;
    }
}

/** @brief 运行 GrabCut 算法。
该函数实现了【GrabCut 图像分割算法】（http://en.wikipedia.org/wiki/GrabCut）。
@param img 输入 8 位 3 通道图像。
@param mask 输入/输出 8 位单通道掩码。当模式设置为#GC_INIT_WITH_RECT 时，函数会初始化掩码。它的元素可能具有#GrabCutClasses 之一。
@param rect ROI 包含一个分段对象。 ROI 之外的像素被标记为“明显背景”。该参数仅在 mode==#GC_INIT_WITH_RECT 时使用。
@param bgdModel 背景模型的临时数组。在处理同一图像时不要修改它。
@param fgdModel 前景模型的临时数组。在处理同一图像时不要修改它。
@param iterCount 算法在返回结果之前应该进行的迭代次数。请注意，可以通过 mode==#GC_INIT_WITH_MASK 或 mode==GC_EVAL 进一步调用来优化结果。
@param mode 可能是#GrabCutModes 之一的操作模式
 */
// 第一步需选矩形或花掩码
int GCApplication::nextIter()
{
    if(!isInitialized){//初始化
        if(rectState == SET)  //矩形已选中
            gc.GrabCut(*image, mask, rect, bgdModel, fgdModel, 1, GC_INIT_WITH_RECT);
        else
            return iterCount;
        isInitialized = true;//初始化完成
    }
    else gc.GrabCut(*image, mask, rect, bgdModel, fgdModel, 1, GC_CUT);//运行 GrabCut 算法
    iterCount++;//迭代次数

    bgdPxls.clear(); fgdPxls.clear();
    prBgdPxls.clear(); prFgdPxls.clear();

    return iterCount;
}