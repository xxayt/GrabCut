#include "GMM.h"
#include <vector>
using namespace std;
using namespace cv;

//GMM的构造函数，从 _model 中读取参数并存储
GMM::GMM(Mat& _model)
{
	//GMM模型有13*K项数据，一个权重，三个均值和九个协方差possibility
	if(_model.empty()){  // 如果模型为空，则创建一个新的
		_model.create(1, 13*K, CV_64FC1);
		_model.setTo(Scalar(0));
	}
	model = _model;
	//存储顺序为权重、均值和协方差
	Pi = model.ptr<double>(0);
	mean = Pi + K;
	cov = mean + 3 * K;
	//如果某个项的权重不为0，则计算其协方差的逆和行列式
	for(int i = 0; i < K; i++)
		if(Pi[i] > 0)
			calcuInvAndDet(i);
}
//3维高斯分布: 计算color像素属于第k个高斯分量的概率
double GMM::Possibility(int k, const Vec3d color) const 
{
	double res = 0;
	if(Pi[k] > 0){
		Vec3d diff = color;
		double* m = mean + 3 * k;
        // X - \mu
		diff[0] -= m[0]; diff[1] -= m[1]; diff[2] -= m[2];
        /* (X - \mu)^T * cov^{-1} * (X - \mu)
           = [diff{0}, diff{1}, diff{2}] *  [covInv[k][0][0], covInv[k][0][1], covInv[k][0][2]] * [diff{0}]
                                            [covInv[k][1][0], covInv[k][1][1], covInv[k][1][2]]   |diff{1}|
                                            [covInv[k][2][0], covInv[k][2][1], covInv[k][2][2]]   [diff{2}]
        */
		double mult = (diff[0] * covInv[k][0][0] + diff[1] * covInv[k][1][0] + diff[2] * covInv[k][2][0]) * diff[0]
			+ (diff[0] * covInv[k][0][1] + diff[1] * covInv[k][1][1] + diff[2] * covInv[k][2][1]) * diff[1]
			+ (diff[0] * covInv[k][0][2] + diff[1] * covInv[k][1][2] + diff[2] * covInv[k][2][2]) * diff[2];
		res = 1.0f / sqrt(covDet[k]) * exp(-0.5f * mult);
	}
	return res;
}
//公式(8): 计算color像素的数据项(对各高斯分布分量的概率加权和)
double GMM::tWeight(const Vec3d color)const 
{
	double res = 0;
	for(int k = 0; k < K; k++)
		res += Pi[k] * Possibility(k, color);
	return res;
}
//计算color像素颜色应该是属于哪个高斯分量（高斯概率最高的项）
int GMM::choice(const Vec3d color) const 
{
	int choosek = 0;
	double max = 0;
	for(int k = 0; k < K; k++){
		double p = Possibility(k, color);
		if(p > max){
			choosek = k;
			max = p;
		}
	}
	return choosek;
}

// 学习之前对中间变量初始化
void GMM::initLearning()
{
	for(int k = 0; k < K; k++){
		for(int i = 0; i < 3; i++)
			sums[k][i] = 0;
		for(int i = 0; i < 3; i++){
			for(int j = 0; j < 3; j++){
				prods[k][i][j] = 0;
			}
		}
		sampleCounts[k] = 0;
	}
	totalSampleCount = 0;
}
// 第k个高斯分量添加一个样本点(更新中间变量)
void GMM::addSample(int k, const Vec3d color)
{
	for(int i = 0; i < 3; i++){
		sums[k][i] += color[i];  // 更新颜色和sums，用于计算均值
		for(int j = 0; j < 3; j++)
			prods[k][i][j] += color[i] * color[j];  // 更新颜色乘积和prods，用于计算协方差
	}
	sampleCounts[k]++;
	totalSampleCount++;
}
//计算GMM新的参数结果(分别对K个高斯模型进行学习)
void GMM::updateParameters()
{
	const double variance = 0.01;
	for(int k = 0; k < K; k++){
		int n = sampleCounts[k];
		if(n == 0)	Pi[k] = 0;
		else{
			//权重
			Pi[k] = 1.0 * n / totalSampleCount;
			//均值
			double * m = mean + 3 * k;
			for(int i = 0; i < 3; i++){
				m[i] = sums[k][i] / n;
			}
			//协方差
			double* c = cov + 9 * k;
			for(int i = 0; i < 3; i++){
				for(int j = 0; j < 3; j++){
					c[i * 3 + j] = prods[k][i][j] / n - m[i] * m[j];
				}
			}
			//如果行列式值太小，则加入一些噪音
			double det = c[0] * (c[4] * c[8] - c[5] * c[7]) - c[1] * (c[3] * c[8] - c[5] * c[6]) + c[2] * (c[3] * c[7] - c[4] * c[6]);
			if(det <= std::numeric_limits<double>::epsilon()){
				c[0] += variance;
				c[4] += variance;
				c[8] += variance;
			}
			//计算协方差的逆和行列式
			calcuInvAndDet(k);
		}
	}
    // cout << "updateParameters done: " << endl;
    // // show Pi
    // cout << "Pi: ";
    // for(int k = 0; k < K; k++){
    //     cout << Pi[k] << " ";
    // }
    // cout << endl;
    // // show mean
    // for(int k = 0; k < K; k++){
    //     cout << mean[3*k] << " " << mean[3*k+1] << " " << mean[3*k+2] << endl;
    // }
}
//计算协方差矩阵的逆和行列式的值
void GMM::calcuInvAndDet(int k)
{
	if(Pi[k] > 0){
		double *a = cov + 9 * k;
        
        /* 
        余子式: M_{ij} = 删除第i行第j列得到的2阶方阵的行列值
        M_{00} =| a[4] a[5] | = a[4]*a[8]-a[5]*a[7]
                | a[7] a[8] |
        M_{01} =| a[3] a[5] | = a[3]*a[8]-a[5]*a[6]
                | a[6] a[8] |
        M_{02} =| a[3] a[4] | = a[3]*a[7]-a[4]*a[6]
                | a[6] a[7] |'
        
        代数余子式: A_{ij} = (-1)^{i+j} * M_{ij}
        A_{00} = M_{00} = a[4]*a[8]-a[5]*a[7]
        A_{01} = -M_{01} = a[5]*a[6]-a[3]*a[8]
        A_{02} = M_{02} = a[3]*a[7]-a[4]*a[6]

        行列式
        det(A) =| a[0] a[1] a[2] | = a[0]*M_{00} - a[1]*M_{01} + a[2]*M_{02} = a[0]*A_{00} + a[1]*A_{01} + a[2]*A_{02}
                | a[3] a[4] a[5] |
                | a[6] a[7] a[8] |
        */
		double det = covDet[k] = a[0] * (a[4] * a[8] - a[5] * a[7]) - a[1] * (a[3] * a[8] - a[5] * a[6]) + a[2] * (a[3] * a[7] - a[4] * a[6]);

        /* 
        伴随矩阵: A* =  | A_{00} A_{10} A_{20} | (已转置)
                        | A_{01} A_{11} A_{21} |
                        | A_{02} A_{12} A_{22} |
        逆矩阵: A^{-1} = A* / det(A)
        */
		covInv[k][0][0] = (a[4] * a[8] - a[5] * a[7]) / det;
		covInv[k][1][0] = -(a[3] * a[8] - a[5] * a[6]) / det;
		covInv[k][2][0] = (a[3] * a[7] - a[4] * a[6]) / det;
		covInv[k][0][1] = -(a[1] * a[8] - a[2] * a[7]) / det;
		covInv[k][1][1] = (a[0] * a[8] - a[2] * a[6]) / det;
		covInv[k][2][1] = -(a[0] * a[7] - a[1] * a[6]) / det;
		covInv[k][0][2] = (a[1] * a[5] - a[2] * a[4]) / det;
		covInv[k][1][2] = -(a[0] * a[5] - a[2] * a[3]) / det;
		covInv[k][2][2] = (a[0] * a[4] - a[1] * a[3]) / det;
	}
}
