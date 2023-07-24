#ifndef GMM_H_
#define GMM_H_
#include <opencv2\opencv.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\core\core.hpp>
using namespace cv;

class GMM
{
public:
	// GMM分量数量，默认为5
	static const int K = 20;
	//GMM的构造函数，从 _model 中读取参数并存储
	GMM(Mat& _model);
	//公式(9): 计算color像素属于第k个高斯分量的概率
	double Possibility(int, const Vec3d) const;
	//公式(8): 计算color像素的数据项加权和
	double tWeight(const Vec3d) const;
	//计算color像素颜色应该是属于哪个高斯分量（高斯概率最高的项）
	int choice(const Vec3d) const;

	// 学习之前对中间变量初始化
	void initLearning();
	// 第k个高斯分量添加一个样本点
	void addSample(int, const Vec3d);
	//根据添加的数据，计算新的参数结果
	void updateParameters();

private:
	//计算协方差矩阵的逆和行列式的值
	void calcuInvAndDet(int);
	//存储GMM模型
	Mat model;
	//GMM中每个高斯分布的三个参数: 权重、均值(3元向量)和协方差(3x3矩阵)
	double *Pi, *mean, *cov;
	//存储协方差的逆，便于计算
	double covInv[K][3][3];
	//存储协方差的行列式值
	double covDet[K];
	// 储存学习参数时的中间变量
	double sums[K][3];  // 每个高斯模型中样本的颜色值之和
	double prods[K][3][3];  // 每个高斯模型中样本的颜色值乘积之和
	int sampleCounts[K];  // 每个高斯模型中样本的数量
	int totalSampleCount;  // 总的样本数
};
#endif