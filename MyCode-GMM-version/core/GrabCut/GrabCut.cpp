#include "GrabCut.h"
#include<iomanip>
#include <iostream>
#include <limits>
#include <vector>
using namespace cv;
using namespace std;


//根据输入矩阵框设置 mask，框外设为MUST_BGD，框内设为MAYBE_FGD。(change: mask, rect)
void GrabCutSegmentation::initMaskWithRect(Mat& mask, Size imgSize, Rect rect)
{
	mask.create(imgSize, CV_8UC1);
	mask.setTo(MUST_BGD);  // 框外设为 MUST_BGD
	rect.x = rect.x > 0 ? rect.x : 0;
	rect.y = rect.y > 0 ? rect.y : 0;
	rect.width = rect.x + rect.width > imgSize.width ? imgSize.width - rect.x : rect.width;
	rect.height = rect.y + rect.height > imgSize.height ? imgSize.height - rect.y : rect.height;
	(mask(rect)).setTo(Scalar(MAYBE_FGD));  // 框内设为 MAYBE_FGD
}

/*  step0. Initialize GMMs by kmeans (change: bgdGMM, fgdGMM)
    1) 储存背景和前景的样本;
    2) 使用kmeans聚类初始化样本所属高斯分量的分配;
    3) 将样本加入到GMM模型中;
    4) 计算GMM中每个高斯模型的参数
*/
void GrabCutSegmentation::initGMMs(const Mat& img, const Mat& mask, GMM& bgdGMM, GMM& fgdGMM)
{
	// 分别存储背景和前景的样本
    vector<Vec3f> bgdSamples, fgdSamples;
	Point p;
	for(p.y = 0; p.y < img.rows; p.y++){
		for(p.x = 0; p.x < img.cols; p.x++){
			if(mask.at<uchar>(p) == MUST_BGD || mask.at<uchar>(p) == MAYBE_BGD)
				bgdSamples.push_back((Vec3f)img.at<Vec3b>(p));
			else
				fgdSamples.push_back((Vec3f)img.at<Vec3b>(p));
		}
	}
    // 构造行数为bgdSamples.size()，列数为3的矩阵，每一行存储一个样本，初始化为bgdSamples[0][0]
	Mat bgdSamples_3cols((int)bgdSamples.size(), 3, CV_32FC1, &bgdSamples[0][0]);  // CV_32FC1: Vec3f = vector<float, 3>
	Mat fgdSamples_3cols((int)fgdSamples.size(), 3, CV_32FC1, &fgdSamples[0][0]);

    // 开始kmeans聚类初始化GMM模型
	Mat bgdLabels, fgdLabels;  // 聚类后每个元素的标签
	const int kmeansItCount = 10;  // 聚类迭代次数
    // TermCriteria 迭代停止模式，CV_TERMCRIT_ITER表示迭代次数达到max_iter时停止迭代
	kmeans(bgdSamples_3cols, GMM::K, bgdLabels, TermCriteria(TermCriteria::MAX_ITER, kmeansItCount, 0.0), 0, KMEANS_PP_CENTERS);
	kmeans(fgdSamples_3cols, GMM::K, fgdLabels, TermCriteria(TermCriteria::MAX_ITER, kmeansItCount, 0.0), 0, KMEANS_PP_CENTERS);
    
    // 确定了每个像素所属的GMM模型,计算GMM中每个高斯模型的参数
	bgdGMM.initLearning();
	fgdGMM.initLearning();
	for(int i = 0; i < (int)bgdSamples.size(); i++)
		bgdGMM.addSample(bgdLabels.at<int>(i, 0), bgdSamples[i]);
	for(int i = 0; i < (int)fgdSamples.size(); i++)
		fgdGMM.addSample(fgdLabels.at<int>(i, 0), fgdSamples[i]);
    // 更新高斯模型参数(权重、均值、协方差)
	bgdGMM.updateParameters();
	fgdGMM.updateParameters();
}

/*  step1. Assign GMM Components to pixels: for each n in Tu (change: CompIndex)
    1) 根据mask信息, 为每个像素分配前景/背景GMM中所属的概率最大的高斯模型，保存在CompIndex中
*/
void GrabCutSegmentation::AssignGMMComponents(const Mat& img, const Mat& mask, const GMM& bgdGMM, const GMM& fgdGMM, Mat& CompIndex)
{
	Point p;
	for(p.y = 0; p.y < img.rows; p.y++){
		for(p.x = 0; p.x < img.cols; p.x++){
			Vec3d color = (Vec3d)img.at<Vec3b>(p);  // Vec3b(uchar) -> Vec3d(double)
			uchar t = mask.at<uchar>(p);
			if(t == MUST_BGD || t == MAYBE_BGD){
                CompIndex.at<int>(p) = bgdGMM.choice(color);  // 将背景GMM中概率最大的高斯模型分配给像素
            }
			else{
                CompIndex.at<int>(p) = fgdGMM.choice(color);
            }
		}
	}
    // show CompIndex
    // Mat colorCompIndex;
    // getcolorCompIndex(CompIndex, colorCompIndex);  // 展示彩色分量
    // imshow("colorCompIndex", colorCompIndex);
}
/*  step2. Learn GMM Parameters from data z (change: bgdGMM, fgdGMM)
    1) 根据CompIndex中的信息，将每个像素重新加入到对应的高斯模型中;
    2) 更新每个高斯模型分量的参数
*/
void GrabCutSegmentation::LearnGMMParameters(const Mat& img, const Mat& mask, GMM& bgdGMM, GMM& fgdGMM, const Mat& CompIndex)
{
	bgdGMM.initLearning();
	fgdGMM.initLearning();
	Point p;
	for(int i = 0; i < GMM::K; i++){
		for(p.y = 0; p.y < img.rows; p.y++){
			for(p.x = 0; p.x < img.cols; p.x++){
				int tmp = CompIndex.at<int>(p);
				if(tmp == i){  // 若像素属于当前第i个高斯模型
					if(mask.at<uchar>(p) == MUST_BGD || mask.at<uchar>(p) == MAYBE_BGD)
						bgdGMM.addSample(tmp, img.at<Vec3b>(p));
					else
						fgdGMM.addSample(tmp, img.at<Vec3b>(p));
				}
			}
		}
	}
    // 更新高斯模型参数(权重、均值、协方差)
	bgdGMM.updateParameters();
	fgdGMM.updateParameters();
}

/*  step2-1. Calculate Beta (change: beta)
    1) 根据公式(5), 使用全局像素差的平方和的期望计算 Beta 的值。
*/
double GrabCutSegmentation::CalcBeta(const Mat& img)
{
	double beta;
	double totalDiff = 0;
    // 遍历整个图像计算每个像素与其相邻像素的欧式距离的平方和
	for(int y = 0; y < img.rows; y++){
		for(int x = 0; x < img.cols; x++){
			Vec3d color = (Vec3d)img.at<Vec3b>(y, x);
			if(x > 0){
				Vec3d diff = color - (Vec3d)img.at<Vec3b>(y, x - 1);
				totalDiff += diff.dot(diff);
			}
			if(y > 0 && x > 0){
				Vec3d diff = color - (Vec3d)img.at<Vec3b>(y - 1, x - 1);
				totalDiff += diff.dot(diff);
			}
			if(y > 0){
				Vec3d diff = color - (Vec3d)img.at<Vec3b>(y - 1, x);
				totalDiff += diff.dot(diff);
			}
			if(y > 0 && x < img.cols - 1){
				Vec3d diff = color - (Vec3d)img.at<Vec3b>(y - 1, x + 1);
				totalDiff += diff.dot(diff);
			}
		}
	}
	if(totalDiff <= std::numeric_limits<double>::epsilon()) beta = 0;
	else{
        totalDiff /= (4 * img.cols*img.rows - 3 * img.cols - 3 * img.rows + 2);  // 求期望
        beta = 1.0 / (2 * totalDiff);  // beta = (2 * E(d(x, y)^2))^{-1}
    }
	return beta;
}
/*  step2-2. Calculate Smoothness Energy (change: leftW, upleftW, upW, uprightW)
    1) 根据公式(11), 计算边界项(相邻像素的)权重差V。由于对称性，八个点我们只需要计算四个方向(优化方法)
*/
void GrabCutSegmentation::CalcSmoothness(const Mat& img, const double beta, const double gamma, 
                                        Mat& leftW, Mat& upleftW, Mat& upW, Mat& uprightW)
{
	// 储存每个像素与其左、左上、上、右上边像素的权重差
    leftW.create(img.size(), CV_64FC1);
	upleftW.create(img.size(), CV_64FC1);
	upW.create(img.size(), CV_64FC1);
	uprightW.create(img.size(), CV_64FC1);
	for(int y = 0; y < img.rows; y++){
		for(int x = 0; x < img.cols; x++){
			Vec3d color = (Vec3d)img.at<Vec3b>(y, x);
			if(x - 1 >= 0){
				Vec3d diff = color - (Vec3d)img.at<Vec3b>(y, x - 1);  // 像素差
				leftW.at<double>(y, x) = gamma * exp(-beta * diff.dot(diff));  // 采用欧式距离(二范数)计算
			}
			else leftW.at<double>(y, x) = 0;
			if(x - 1 >= 0 && y - 1 >= 0){
				Vec3d diff = color - (Vec3d)img.at<Vec3b>(y - 1, x - 1);
				upleftW.at<double>(y, x) = gamma * exp(-beta*diff.dot(diff)) / sqrt(2.0f);
			}
			else upleftW.at<double>(y, x) = 0;
			if(y - 1 >= 0){
				Vec3d diff = color - (Vec3d)img.at<Vec3b>(y - 1, x);
				upW.at<double>(y, x) = gamma * exp(-beta*diff.dot(diff));
			}
			else upW.at<double>(y, x) = 0;
			if(x + 1 < img.cols && y - 1 >= 0){
				Vec3d diff = color - (Vec3d)img.at<Vec3b>(y - 1, x + 1);
				uprightW.at<double>(y, x) = gamma * exp(-beta*diff.dot(diff)) / sqrt(2.0f);
			}
			else uprightW.at<double>(y, x) = 0;
		}
	}
}
/*  step2-3. Build Graph based on edge-weights (change: graph)
    1) 根据Graph Cut, 构建t-links(terminal links)边的权重——数据项能量U (基于GMM模型的概率)
    2) 根据Graph Cut, 构建n-links(neighborhood links)边的权重——边界项能量V;
*/
void GrabCutSegmentation::getGraph(const Mat& img, const Mat& mask, 
                                const GMM& bgdGMM, const GMM& fgdGMM, 
                                const Mat& leftW, const Mat& upleftW, const Mat& upW, const Mat& uprightW, 
                                double lambda, Graph<double, double, double>& graph)
{
	Point p;
	for(p.y = 0; p.y < img.rows; p.y++){
		for(p.x = 0; p.x < img.cols; p.x++){
            int vNum = graph.add_node();  // 添加vNum节点(p.x, p.y)

            // 构建数据项能量U: t-link权重
			Vec3b color = img.at<Vec3b>(p);
			double wSource, wSink;
			if(mask.at<uchar>(p) == MAYBE_BGD || mask.at<uchar>(p) == MAYBE_FGD){  // 非确定前景背景的像素
                // 计算color像素属于背景或前景的损失(负对数)
				wSource = -log(bgdGMM.tWeight(color));
				wSink = -log(fgdGMM.tWeight(color));
			}
			else if(mask.at<uchar>(p) == MUST_BGD){  // 确定背景的像素
                wSource = 0;  // 此边权重为0，表示不与SOURCE相连
                wSink = lambda;  // 此边权重为lambda=500，割去成本最大，因此保留在背景中
            }
			else{  // 确定前景的像素
                wSource = lambda;
                wSink = 0;
            }
			graph.add_tweights(vNum, wSource, wSink);  // vNum分别与SOURCE和SINK建边
            tweight_wSource.push_back(wSource), tweight_wSink.push_back(wSink);  // 储存用于计算

            // 构建边界能量项V: n-link权重
			if(p.x > 0){  // 与左点
				double w = leftW.at<double>(p);
				graph.add_edge(vNum, vNum - 1, w, w);  // vNum与vNum-1建边
			}
			if(p.x > 0 && p.y > 0){  // 与左上点
				double w = upleftW.at<double>(p);
				graph.add_edge(vNum, vNum - img.cols - 1, w, w);  // vNum与vNum-img.cols-1建边
			}
			if(p.y > 0){  // 与上点
				double w = upW.at<double>(p);
				graph.add_edge(vNum, vNum - img.cols, w, w);  // vNum与vNum-img.cols建边
			}
			if(p.x < img.cols - 1 && p.y > 0){  // 与右上点
				double w = uprightW.at<double>(p);
				graph.add_edge(vNum, vNum - img.cols + 1, w, w);  // vNum与vNum-img.cols+1建边
			}
		}
	}
    
}
/* step3. Estimate Segmentation: use min cut to solve (change: graph, mask)
    1) 根据Graph Cut的图, 使用maxflow库进行分割;
    2) 根据分割结果更新mask;
*/
void GrabCutSegmentation::EstimateSegmentation(Graph<double, double, double>& graph, Mat& mask)
{
	graph.maxflow();  // 使用maxflow库进行分割
	Point p;
	for(p.y = 0; p.y < mask.rows; p.y++){
		for(p.x = 0; p.x < mask.cols; p.x++){
            // 只对不确定前景背景的像素, 根据分割结果进行更新
			if(mask.at<uchar>(p) == MAYBE_BGD || mask.at<uchar>(p) == MAYBE_FGD){
				// 若属于前景，标记为MAYBE_FGD，否则标记为MAYBE_BGD
                if(graph.what_segment(p.y*mask.cols + p.x) == Graph<double, double, double>::SOURCE)
					mask.at<uchar>(p) = MAYBE_FGD;
				else
                    mask.at<uchar>(p) = MAYBE_BGD;
			}
		}
	}
}
// 计算能量项，衡量分割效果
void GrabCutSegmentation::CalcEneryFunction(const Graph<double, double, double>& graph, const Mat& mask, const Mat& leftW, const Mat& upleftW, const Mat& upW, const Mat& uprightW)
{
    double Usum = 0, Vsum = 0;
    Point p;
	for(p.y = 0; p.y < mask.rows; p.y++){
		for(p.x = 0; p.x < mask.cols; p.x++){
            // 计算U项
            if(mask.at<uchar>(p) == MAYBE_BGD || mask.at<uchar>(p) == MAYBE_FGD){
                if(graph.what_segment(p.y*mask.cols + p.x) == Graph<double, double, double>::SOURCE)
                    Usum += tweight_wSource[p.y*mask.cols + p.x];
                else
                    Usum += tweight_wSink[p.y*mask.cols + p.x];
            }
            // 计算V项
            if(p.x > 0){
                double w = leftW.at<double>(p);
                if(graph.what_segment(p.y*mask.cols + p.x) != graph.what_segment(p.y*mask.cols + p.x - 1))
                    Vsum += w;
            }
            if(p.x > 0 && p.y > 0){
                double w = upleftW.at<double>(p);
                if(graph.what_segment(p.y*mask.cols + p.x) != graph.what_segment(p.y*mask.cols + p.x - mask.cols - 1))
                    Vsum += w;
            }
            if(p.y > 0){
                double w = upW.at<double>(p);
                if(graph.what_segment(p.y*mask.cols + p.x) != graph.what_segment(p.y*mask.cols + p.x - mask.cols))
                    Vsum += w;
            }
            if(p.x < mask.cols - 1 && p.y > 0){
                double w = uprightW.at<double>(p);
                if(graph.what_segment(p.y*mask.cols + p.x) != graph.what_segment(p.y*mask.cols + p.x - mask.cols + 1))
                    Vsum += w;
            }
		}
	}
    cout << "Usum = " << fixed << setprecision(0) << Usum;
    cout << ", Vsum = " << Vsum << endl;
    // cout << "(" << fixed << setprecision(0) << Usum << ", " << Vsum << ") ";
}
//GrabCut 主函数
void GrabCutSegmentation::GrabCut(InputArray arrayimg, InputOutputArray arraymask, Rect rect,	
                        InputOutputArray _bgdModel, InputOutputArray _fgdModel, 
                        int iterCount, int mode)
{
	Mat img = arrayimg.getMat();
	Mat& mask = arraymask.getMatRef();
	Mat& bgdModel = _bgdModel.getMatRef();
	Mat& fgdModel = _fgdModel.getMatRef();
	GMM backgroundGMM(bgdModel), foregroundGMM(fgdModel);  // 构建背景和前景的GMM模型
	if(mode == GC_WITH_RECT){
        // 初始化mask
        initMaskWithRect(mask, img.size(), rect);
        // 初始化GMM模型
        initGMMs(img, mask, backgroundGMM, foregroundGMM);
    }
	if(iterCount <= 0) return;
    
    // 计算Beta的值
	const double beta = CalcBeta(img);
	// 计算平滑项(边界能量项V)
	Mat leftWeight, upleftWeight, upWeight, uprightWeight;
	CalcSmoothness(img, beta, gamma, leftWeight, upleftWeight, upWeight, uprightWeight);
	// 存储每个像素属于哪个高斯模型
	Mat ComponentIndex(img.size(), CV_32SC1);
	const double lambda = 8 * gamma + 1;
	for(int i = 0; i < iterCount; i++){
        int vCount = img.cols*img.rows;
	    int eCount = 2 * (4 * vCount - 3 * img.cols - 3 * img.rows + 2);  // 无向图=双向图
        Graph<double, double, double> graph(vCount, eCount);  // 建图
		AssignGMMComponents(img, mask, backgroundGMM, foregroundGMM, ComponentIndex);
		LearnGMMParameters(img, mask, backgroundGMM, foregroundGMM, ComponentIndex);
		getGraph(img, mask, backgroundGMM, foregroundGMM, leftWeight, upleftWeight, upWeight, uprightWeight, lambda, graph);
		EstimateSegmentation(graph, mask);
        CalcEneryFunction(graph, mask, leftWeight, upleftWeight, upWeight, uprightWeight);
	}
}