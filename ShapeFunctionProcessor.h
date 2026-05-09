#pragma once

#include <Eigen/Dense>
#include "GaussianQuadrature.h"
#include "BasisFunction.h"

class ShapeFunctionProcessor {
public:
    // 构造函数 - 接收必要的参数
    ShapeFunctionProcessor(int order_p, double a0, double a1)
        : order_p_(order_p), a0_(a0), a1_(a1) {
        gpn_ = order_p_ + 10;
    }
    
    // 主要计算函数 - 封装原main.cpp 94-173行的所有逻辑
    void compute() {
        // 计算高斯积分点和权重
        GaussianQuadrature::ComputeNodesAndWeights(gpn_, 0.0, 1.0, localGps, localGpWs);
        
        // 计算1D形状函数
        localGpValues1D = BasisFunction::FEMShapeFunction(localGps, order_p_, 0);
        localGpDerivs1D = BasisFunction::FEMShapeFunction(localGps, order_p_, 1);
        
        // 初始化2D形状函数矩阵
        localGpValues2D = MatrixXd::Zero(gpn_ * gpn_, (order_p_ + 1) * (order_p_ + 1));
        localGpDerivs2D = MatrixXd::Zero(gpn_ * gpn_, (order_p_ + 1) * (order_p_ + 1));
        localGpValues2D_s = MatrixXd::Zero(gpn_ * gpn_, (order_p_ + 1) * (order_p_ + 1));
        localGpValues2D_t = MatrixXd::Zero(gpn_ * gpn_, (order_p_ + 1) * (order_p_ + 1));
        
        // 计算PU形状函数
        localGpValues1D_PU = BasisFunction::PUShapeFunction(localGps, 0);
        localGpDerivs1D_PU = BasisFunction::PUShapeFunction(localGps, 1);
        localGpValues2D_PU = MatrixXd::Zero(gpn_ * gpn_, 4);
        localGpValues2D_PU_s = MatrixXd::Zero(gpn_ * gpn_, 4);
        localGpValues2D_PU_t = MatrixXd::Zero(gpn_ * gpn_, 4);
        
        // 创建2D积分点网格
        localGpsY = localGps.replicate(gpn_, 1);
        localGpsX = VectorXd::Zero(gpn_ * gpn_);
        for (int i = 0; i < gpn_; ++i) {
            localGpsX.segment(i * gpn_, gpn_) = VectorXd::Constant(gpn_, localGps(i));
        }
        
        localGps2D = MatrixXd(gpn_ * gpn_, 2);
        localGps2D.col(0) = localGpsX;
        localGps2D.col(1) = localGpsY;
        
        // 计算2D权重
        MatrixXd tmp = localGpWs * localGpWs.transpose();
        localGpW2D = tmp.reshaped();
        
        // 材料参数
        kappa0 = VectorXd::Constant(gpn_ * gpn_, a0_);
        kappa1 = VectorXd::Constant(gpn_ * gpn_, a1_);
        
        // 计算2D形状函数（张量积）
        for (int i = 1; i <= order_p_ + 1; ++i) {
            for (int j = 1; j <= order_p_ + 1; ++j) {
                MatrixXd temp1 = localGpValues1D.col(i - 1) * localGpValues1D.col(j - 1).transpose();
                MatrixXd temp2 = localGpDerivs1D.col(i - 1) * localGpDerivs1D.col(j - 1).transpose();
                MatrixXd temp3 = localGpValues1D.col(i - 1) * localGpDerivs1D.col(j - 1).transpose();
                MatrixXd temp4 = localGpDerivs1D.col(i - 1) * localGpValues1D.col(j - 1).transpose();
                
                int col_idx = (j - 1) * (order_p_ + 1) + i - 1;
                localGpValues2D.col(col_idx) = temp1.reshaped();
                localGpDerivs2D.col(col_idx) = temp2.reshaped();
                localGpValues2D_s.col(col_idx) = temp3.reshaped();
                localGpValues2D_t.col(col_idx) = temp4.reshaped();
            }
        }
        
        // 计算2D PU形状函数
        int PU_Dim1D = localGpValues1D_PU.cols();
        for (int i = 0; i < PU_Dim1D; ++i) {
            for (int j = 0; j < PU_Dim1D; ++j) {
                MatrixXd temp1 = localGpValues1D_PU.col(i) * localGpValues1D_PU.col(j).transpose();
                MatrixXd temp2 = localGpValues1D_PU.col(i) * localGpDerivs1D_PU.col(j).transpose();
                MatrixXd temp3 = localGpDerivs1D_PU.col(i) * localGpValues1D_PU.col(j).transpose();
                
                int col_idx = j * PU_Dim1D + i;
                localGpValues2D_PU.col(col_idx) = temp1.reshaped();
                localGpValues2D_PU_s.col(col_idx) = temp2.reshaped();
                localGpValues2D_PU_t.col(col_idx) = temp3.reshaped();
            }
        }
    }
    
    // 公共成员变量 - 保持与原代码相同的变量名
    // 这样可以在main函数中创建引用，无需修改后续代码
    
    // 高斯积分点和权重
    Eigen::VectorXd localGps;
    Eigen::VectorXd localGpWs;
    
    // 1D形状函数
    Eigen::MatrixXd localGpValues1D;
    Eigen::MatrixXd localGpDerivs1D;
    
    // 2D形状函数
    Eigen::MatrixXd localGpValues2D;
    Eigen::MatrixXd localGpDerivs2D;
    Eigen::MatrixXd localGpValues2D_s;
    Eigen::MatrixXd localGpValues2D_t;
    
    // 1D PU形状函数
    Eigen::MatrixXd localGpValues1D_PU;
    Eigen::MatrixXd localGpDerivs1D_PU;
    
    // 2D PU形状函数
    Eigen::MatrixXd localGpValues2D_PU;
    Eigen::MatrixXd localGpValues2D_PU_s;
    Eigen::MatrixXd localGpValues2D_PU_t;
    
    // 2D积分点网格
    Eigen::VectorXd localGpsY;
    Eigen::VectorXd localGpsX;
    Eigen::MatrixXd localGps2D;
    
    // 2D权重
    Eigen::VectorXd localGpW2D;
    
    // 材料参数
    Eigen::VectorXd kappa0;
    Eigen::VectorXd kappa1;
    
private:
    int order_p_;
    double a0_;
    double a1_;
    int gpn_;
};
