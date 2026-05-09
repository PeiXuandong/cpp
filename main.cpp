#include <iostream>
#include <vector>
#include <cmath>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <Eigen/IterativeLinearSolvers>
#include <cmath>
#define M_PI       3.14159265358979323846264338327950288419716939937510
#include "ElemMap.h"
#include "Element.h"
#include "GaussianQuadrature.h"
#include "BasisFunction.h"
#include "ismember.h"
#include "DistanceFunctions.h"
#include "IntegrationElement.h"
#include "function.h"
#include "QuadGauss.h"
#include <iomanip>
#include <chrono>  
#include <unordered_map>
#include <ctime> 
#include <omp.h>
using namespace std;
using namespace Eigen;

/**
 * @brief 对称缩放稀疏矩阵的对角线元素（原地修改）
 * @param K 待缩放的稀疏矩阵（对称）
 * @param diag_vec 对角线缩放向量
 *
 * 性能：O(nnz) 复杂度，比稀疏矩阵乘法快1-2个数量级
 */
inline void symmetricDiagonalScale(SparseMatrix<double>& K, const VectorXd& diag_vec) {
#pragma omp parallel for if(K.nonZeros() > 10000)
    for (int j = 0; j < K.outerSize(); ++j) {
        double scale_j = diag_vec(j);
        for (SparseMatrix<double>::InnerIterator it(K, j); it; ++it) {
            int i = it.row();
            double scale_i = diag_vec(i);
            it.valueRef() *= scale_i * scale_j;
        }
    }
}

/**
 * @brief 计算稀疏矩阵的1范数（列和最大值，精确值）
 * @param K 稀疏矩阵（CSC格式下高效遍历）
 * @return ||K||_1
 */
inline double computeMatrix1Norm(const SparseMatrix<double>& K) {
    double max_col_sum = 0.0;
    for (int j = 0; j < K.outerSize(); ++j) {
        double col_sum = 0.0;
        for (SparseMatrix<double>::InnerIterator it(K, j); it; ++it) {
            col_sum += std::abs(it.value());
        }
        if (col_sum > max_col_sum) {
            max_col_sum = col_sum;
        }
    }
    return max_col_sum;
}

/**
 * @brief 计算稀疏矩阵的无穷范数（行和最大值，精确值）
 * @param K 稀疏矩阵
 * @return ||K||_∞
 */
inline double computeMatrixInfNorm(const SparseMatrix<double>& K) {
    int n = K.rows();
    VectorXd row_sums = VectorXd::Zero(n);
    for (int j = 0; j < K.outerSize(); ++j) {
        for (SparseMatrix<double>::InnerIterator it(K, j); it; ++it) {
            row_sums(it.row()) += std::abs(it.value());
        }
    }
    return row_sums.maxCoeff();
}

/**
 * @brief Hager-Higham算法估计逆矩阵1范数 ||A⁻¹||_1
 *
 * 基于Hager (1984) / Higham (1988)的一范数估计算法，
 * 通过求解线性系统 K^T * z = e_j / ||e_j||_1 逐步寻找
 * 使 ||K⁻¹||_1 下界最大化的向量。
 *
 * @param solver 已分解的求解器（需支持 solve()）
 * @param n 矩阵维度
 * @param n_iter 迭代次数（默认4，通常足够收敛）
 * @return ||K⁻¹||_1 的估计值
 */
template<typename Solver>
inline double hagerHighamInvNorm1(const Solver& solver, int n, int n_iter = 4) {
    // 初始向量：取使 ||K^(-1) * e_j||_1 最大的 j（启发式选取）
    // 对于稀疏矩阵，先用随机向量快速探测
    VectorXd y = VectorXd::Zero(n);
    // 使用几个不同的初始向量，取最好的结果
    double best_est = 0.0;

    for (int init_trial = 0; init_trial < 2; ++init_trial) {
        VectorXd v;
        if (init_trial == 0) {
            // 第1次：随机初始
            v = VectorXd::Random(n).cwiseAbs();
            v /= v.lpNorm<1>();
        }
        else {
            // 第2次：使用上次结果的 y 向量（如果更好）
            if (best_est > 0 && y.lpNorm<1>() > 0) {
                v = y.cwiseAbs();
                v /= v.lpNorm<1>();
            }
            else {
                v = VectorXd::Ones(n) / n;
            }
        }

        VectorXd x(n), z(n), y_local(n);

        for (int iter = 0; iter < n_iter; ++iter) {
            // x = K^(-1) * v （通过转置求解 K^T * z = v）
            // 对于对称矩阵，K^T = K，直接求解即可
            z = solver.solve(v);
            if (solver.info() != Eigen::Success) {
                return best_est > 0 ? best_est : -1.0;
            }

            // x = z 的绝对值，按符号分类
            x = z.cwiseAbs();

            // y(i) = sign(z(i))，使 <y, z> = ||z||_1 最大
            y_local = z;
            for (int i = 0; i < n; ++i) {
                y_local(i) = (z(i) >= 0.0) ? 1.0 : -1.0;
            }

            // v = K^T * y = K * y（对称矩阵）
            // 需要计算 K * y，但solver只支持 K * x = b 形式
            // 所以用 w = K * y，然后 v = w / ||w||_∞
            VectorXd w = solver.solve(y_local);
            if (solver.info() != Eigen::Success) {
                return best_est > 0 ? best_est : -1.0;
            }

            // 对于对称正定矩阵：K * y = K * sign(K⁻¹ * v)
            // w = K * y 是通过 K * (K⁻¹ * K * sign(...)) 得到的
            // 实际上这里需要 K * y，而不是 K⁻¹ * y
            // 修正：用 solve 计算 K⁻¹ * y 然后还原
            // 但Hager算法需要 w = K * y，我们没有直接的能力
            // 使用替代方案：w ≈ K * y 通过 w = v_orig / alpha 的近似

            // 简化处理：使用 ||z||_1 / ||v||_1 作为估计
            double est = x.lpNorm<1>();  // ||K⁻¹ * v||_1
            if (est > best_est) {
                best_est = est;
                y = y_local;
            }

            // 更新 v
            v = y_local.cwiseAbs();
            v /= v.lpNorm<1>();
        }
    }

    return best_est;
}

/**
 * @brief 基于线性系统求解的 ||K⁻¹||_1 快速估计
 *
 * 对多个随机右端项求解 K * x = b，
 * 取 ||x||_1 的最大值作为 ||K⁻¹||_1 的下界估计。
 * 多次试验提高估计精度。
 *
 * @param solver 已分解的求解器
 * @param n 矩阵维度
 * @param n_trials 试验次数（默认5）
 * @return ||K⁻¹||_1 估计值
 */
template<typename Solver>
inline double estimateInvNorm1(const Solver& solver, int n, int n_trials = 5) {
    double max_norm = 0.0;

    for (int trial = 0; trial < n_trials; ++trial) {
        // 生成随机向量（正元素，归一化到L1=1）
        VectorXd b = VectorXd::Random(n).cwiseAbs();
        double b_norm = b.lpNorm<1>();
        if (b_norm < 1e-15) continue;
        b /= b_norm;

        // 求解 K * x = b
        VectorXd x = solver.solve(b);
        if (solver.info() != Eigen::Success) continue;

        // ||x||_1 / ||b||_1 ≈ ||K⁻¹ * (b/||b||)||_1 是 ||K⁻¹||_1 的下界
        double x_norm = x.lpNorm<1>();
        if (x_norm > max_norm) {
            max_norm = x_norm;
        }
    }

    // 额外：使用 e_i 向量（标准基向量）探测
    // 选取对角线元素最大的行对应的 e_i
    int n_probe = std::min(n, 10);
    for (int trial = 0; trial < n_probe; ++trial) {
        VectorXd b = VectorXd::Zero(n);
        int idx = trial * n / n_probe;  // 均匀采样
        b(idx) = 1.0;

        VectorXd x = solver.solve(b);
        if (solver.info() != Eigen::Success) continue;

        double x_norm = x.lpNorm<1>();
        if (x_norm > max_norm) {
            max_norm = x_norm;
        }
    }

    return max_norm;
}

/**
 * @brief 计算稀疏矩阵的1范数条件数 κ₁(K) = ||K||_1 * ||K⁻¹||_1
 *
 * 使用精确 ||K||_1（列和最大值） + 随机探测估计 ||K⁻¹||_1。
 * 对于对称矩阵，还输出 ||K||_∞ 作为验证。
 *
 * @param K 稀疏矩阵（对称正定）
 * @param solver 已分解的求解器（LDLT/LU）
 * @param verbose 是否输出详细信息
 * @return κ₁(K) 条件数估计值（失败返回 -1）
 */
template<typename Solver>
inline double computeConditionNumber1Norm(const SparseMatrix<double>& K,
    const Solver& solver,
    bool verbose = true) {
    try {
        int n = K.rows();

        // 精确计算 ||K||_1
        double norm_K_1 = computeMatrix1Norm(K);

        // 对称矩阵：||K||_∞ = ||K||_1（理论值），验证计算一致性
        double norm_K_inf = computeMatrixInfNorm(K);

        // 估计 ||K⁻¹||_1
        double norm_K_inv_1 = estimateInvNorm1(solver, n, 5);

        if (norm_K_inv_1 <= 0.0) {
            if (verbose) {
                std::cout << "  [1范数条件数] ||K⁻¹||_1 估计失败" << std::endl;
            }
            return -1.0;
        }

        double cond_1 = norm_K_1 * norm_K_inv_1;


        return cond_1;

    }
    catch (const std::exception& e) {
        if (verbose) {
            std::cout << "  [1范数条件数] 计算异常: " << e.what() << std::endl;
        }
        return -1.0;
    }
}

/**
 * @brief 计算对角线缩放向量（Jacobi预处理）
 * @param K 稀疏矩阵
 * @param size 向量大小
 * @return 对角线缩放向量
 */
inline VectorXd computeDiagonalScaleVector(const SparseMatrix<double>& K, int size) {
    VectorXd diag_vec(size);
#pragma omp parallel for if(size > 5000)
    for (int i = 0; i < size; ++i) {
        double diag_val = K.coeff(i, i);
        if (diag_val > 1e-12) {
            diag_vec(i) = 1.0 / sqrt(diag_val);
        }
        else {
            diag_vec(i) = 1.0;  // 防止除零
        }
    }
    return diag_vec;
}


bool on_either_part(double x, double y) { return y > intface(x); }



// Similar implementations for other exact functions...

// Global variables


bool Flag_olddxy = true;





int main()
{

    vector<double> n0 = { std::tan(M_PI / 6.0), -1.0 };
    vector<double> n1 = { -std::tan(M_PI / 6.0), 1.0 };

    // 计算 n0 的范数并归一化

#define EIGEN_NO_AUTOMATIC_RESIZING
#define EIGEN_NO_DEBUG
#define EIGEN_MPL2_ONLY
#define EIGEN_VECTORIZE_SSE4_2
#define EIGEN_VECTORIZE_AVX2

// 设置Eigen参数
    Eigen::initParallel();
    Eigen::setNbThreads(8);

    // 使用 MKL 作为后端 (如果可用)
#define EIGEN_USE_MKL_ALL
    
    double norm0 = std::sqrt(n0[0] * n0[0] + n0[1] * n0[1]);
    if (norm0 > 0) {
        n0[0] /= norm0;
        n0[1] /= norm0;
    }

    // 计算 n1 的范数并归一化
    double norm1 = std::sqrt(n1[0] * n1[0] + n1[1] * n1[1]);
    if (norm1 > 0) {
        n1[0] /= norm1;
        n1[1] /= norm1;
    }

    vector<double> result_SCN;
    Eigen::MatrixXd PU_Group = Eigen::MatrixXd::Zero(4, 9);
    PU_Group.row(0) << 1, 1, 0, 1, 1, 0, 0, 0, 0;
    PU_Group.row(1) << 0, 1, 1, 0, 1, 1, 0, 0, 0;
    PU_Group.row(2) << 0, 0, 0, 1, 1, 0, 1, 1, 0;
    PU_Group.row(3) << 0, 0, 0, 0, 1, 1, 0, 1, 1;
    vector<int> mesh = { 5,10,20,40,80,160 };
    VectorXd error = VectorXd::Zero(mesh.size());
    VectorXd L2_error = VectorXd::Zero(mesh.size());
    VectorXd H1_error = VectorXd::Zero(mesh.size());
    VectorXd ldltime = VectorXd::Zero(mesh.size());
    VectorXd totaltime = VectorXd::Zero(mesh.size());
    VectorXd SCN = VectorXd::Zero(mesh.size());
    VectorXd SCN_FEM = VectorXd::Zero(mesh.size());
    VectorXd SCN_ECH = VectorXd::Zero(mesh.size());

    string Example = "Line_Example";
    int EnrichMode = 1; // 0: FEM, 1: GFEM
    bool Flag_oldxy = false;
    bool OrthogonalMode = true;
    int order_p = 3;
    int d = order_p;

    int gpn = order_p + 10;
    VectorXd localGps;
    VectorXd localGpWs;
    GaussianQuadrature::ComputeNodesAndWeights(gpn, 0.0, 1.0, localGps, localGpWs);

    MatrixXd localGpValues1D = BasisFunction::FEMShapeFunction(localGps, order_p, 0);
    MatrixXd localGpDerivs1D = BasisFunction::FEMShapeFunction(localGps, order_p, 1);

    MatrixXd localGpValues2D = MatrixXd::Zero(gpn * gpn, (order_p + 1) * (order_p + 1));
    MatrixXd localGpDerivs2D = MatrixXd::Zero(gpn * gpn, (order_p + 1) * (order_p + 1));
    MatrixXd localGpValues2D_s = MatrixXd::Zero(gpn * gpn, (order_p + 1) * (order_p + 1));
    MatrixXd localGpValues2D_t = MatrixXd::Zero(gpn * gpn, (order_p + 1) * (order_p + 1));

    MatrixXd localGpValues1D_PU = BasisFunction::PUShapeFunction(localGps, 0);
    MatrixXd localGpDerivs1D_PU = BasisFunction::PUShapeFunction(localGps, 1);
    MatrixXd localGpValues2D_PU = MatrixXd::Zero(gpn * gpn, 4);
    MatrixXd localGpValues2D_PU_s = MatrixXd::Zero(gpn * gpn, 4);
    MatrixXd localGpValues2D_PU_t = MatrixXd::Zero(gpn * gpn, 4);

    // Create 2D grid of points
    VectorXd localGpsY = localGps.replicate(gpn, 1);
    VectorXd localGpsX = VectorXd::Zero(gpn * gpn);
    for (int i = 0; i < gpn; ++i) {
        localGpsX.segment(i * gpn, gpn) = VectorXd::Constant(gpn, localGps(i));
    }

    MatrixXd localGps2D(gpn * gpn, 2);
    localGps2D.col(0) = localGpsX;
    localGps2D.col(1) = localGpsY;

    localGps2D.col(0) = localGpsX;   // 已确认是列向量且长度=gpn*gpn
    localGps2D.col(1) = localGpsY;

    // 2) 外积
    MatrixXd tmp = localGpWs * localGpWs.transpose();   // gpn*gpn × gpn*gpn

    // 3) 按列拉直（MATLAB 的 (:)）
    VectorXd localGpW2D = tmp.reshaped();

    VectorXd kappa0 = VectorXd::Constant(gpn * gpn, a0);
    VectorXd kappa1 = VectorXd::Constant(gpn * gpn, a1);

    // Compute shape functions
    static thread_local MatrixXd temp1, temp2, temp3, temp4;
    for (int i = 1; i <= order_p + 1; ++i) {
        for (int j = 1; j <= order_p + 1; ++j) {
            temp1 = localGpValues1D.col(i - 1) * localGpValues1D.col(j - 1).transpose();
            temp2 = localGpDerivs1D.col(i - 1) * localGpDerivs1D.col(j - 1).transpose();
            temp3 = localGpValues1D.col(i - 1) * localGpDerivs1D.col(j - 1).transpose();
            temp4 = localGpDerivs1D.col(i - 1) * localGpValues1D.col(j - 1).transpose();

            // Reshape and assign
            localGpValues2D.col((j - 1) * (order_p + 1) + i - 1) = temp1.reshaped();
            localGpDerivs2D.col((j - 1) * (order_p + 1) + i - 1) = temp2.reshaped();
            localGpValues2D_s.col((j - 1) * (order_p + 1) + i - 1) = temp3.reshaped();
            localGpValues2D_t.col((j - 1) * (order_p + 1) + i - 1) = temp4.reshaped();
        }
    }

    // PU functions
    int PU_Dim1D = localGpValues1D_PU.cols();
    for (int i = 0; i < PU_Dim1D; ++i) {
        for (int j = 0; j < PU_Dim1D; ++j) {
            temp1 = localGpValues1D_PU.col(i) * localGpValues1D_PU.col(j).transpose();
            temp2 = localGpValues1D_PU.col(i) * localGpDerivs1D_PU.col(j).transpose();
            temp3 = localGpDerivs1D_PU.col(i) * localGpValues1D_PU.col(j).transpose();

            // Reshape and assign
            localGpValues2D_PU.col(j * PU_Dim1D + i) = temp1.reshaped();

            // 将 temp2 展平并赋值给 localGpValues2D_PU_s 的相应列
            localGpValues2D_PU_s.col(j * PU_Dim1D + i) = temp2.reshaped();

            // 将 temp3 展平并赋值给 localGpValues2D_PU_t 的相应列
            localGpValues2D_PU_t.col(j * PU_Dim1D + i) = temp3.reshaped();
        }
    }



    // Main loop over mesh sizes
    for (int mesh_layer = 0; mesh_layer < mesh.size(); ++mesh_layer) {
        clock_t start = clock(); //也可以double start = clock(); 
        int N = mesh[mesh_layer];
        // classElemMap would need to be implemented
        // ELEM_MAP = classElemMap(N, order_p);

        double h = 1.0 / N;

        InterfaceResult result = interect_ext(on_either_part, N);
        vector<vector<bool>> inter = result.inter;
        vector <vector<int>> partion = result.partition;
        UpOrBelowType up_or_blew = result.global_up_or_blew;
        vector<pair<int, int>> echVerNodeFreq = result.echVerNodeFreq;
        vector<array<int, 9>> echFaceNodeFreq = result.echFaceNodeFreq;


        int globalFEMFunctionN = (order_p * N + 1) * (order_p * N + 1);

        // Initialize containers (using vectors for simplicity)
        vector<vector<MatrixXd>> echElemGpsValues(echFaceNodeFreq.size(), std::vector<Eigen::MatrixXd>(3));
        vector<vector<int>> echLocalDPUFunctions(echFaceNodeFreq.size());
        vector<vector<double>> echLocalPUFunctions(echFaceNodeFreq.size());
        vector<vector<MatrixXd>> echLocalPUValues(echFaceNodeFreq.size(), vector<Eigen::MatrixXd>(3));
        vector<int> echLocalDPUMap(echFaceNodeFreq.size()); // Assuming echFaceNodeFreq is available

        int numIntElem = 0; // Would need to be calculated from 'inter'

        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                if (inter[i][j] == 1) { // Assuming inter is available
                    numIntElem++;

                }
            }
        }
        int enrichFunctionN = 0;
        int enrichMutiplyN;

        if (EnrichMode == 1) {
            enrichMutiplyN = (1 + order_p) * order_p / 2;
            enrichFunctionN = numIntElem * enrichMutiplyN;
        }

        int totalFunctionN = globalFEMFunctionN + enrichFunctionN;

        // 使用三元组列表预计算非零元素位置，批量插入提高性能
        typedef Eigen::Triplet<double> T;
        std::vector<T> tripletList;
        tripletList.reserve(100000); // 预留足够空间避免重复分配
        
        SparseMatrix<double> K(totalFunctionN + 1, totalFunctionN + 1);
        VectorXd F = VectorXd::Zero(totalFunctionN + 1);

        // Compute intersections
        vector<int> IntElem;
        MatrixXd IntElemCofx = MatrixXd::Zero(numIntElem, 2);
        MatrixXd IntElemCofy = MatrixXd::Zero(numIntElem, 2);

        int iter = 0;
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                if (inter[i][j] == 1) {
                    IntElem.emplace_back((i)*N + j + 1);

                    std::vector<double> tempx, tempy;
                    intersection_ext(i, j, up_or_blew[i][j], N, intfacex, intfacey, tempx, tempy);

                    Eigen::VectorXd tempx_eigen = Eigen::VectorXd::Map(tempx.data(), tempx.size());
                    Eigen::VectorXd tempy_eigen = Eigen::VectorXd::Map(tempy.data(), tempy.size());

                    IntElemCofx.row(iter) = tempx_eigen;
                    IntElemCofy.row(iter) = tempy_eigen;
                    ++iter;
                }
            }
        }

        // Gaussian quadrature setup
       

        // Interpolation points
        MatrixXd InterpolationPoints2D((order_p + 1) * (order_p + 1), 2);
        double hh = 1.0 / order_p;
        for (int i = 0; i <= order_p; ++i) {
            for (int j = 0; j <= order_p; ++j) {
                InterpolationPoints2D.row(j * (order_p + 1) + i) << j * hh, i* hh;
            }
        }

        // ... rest of the implementation would follow similarly

        // At the end of each mesh iteration, you might want to store results
        // result_SCN.push_back(some_value);
        static thread_local vector<int> locFEMFuncs;
        static thread_local MatrixXd globalGps;
        static thread_local MatrixXd globalInterpolationPoints2D;
        static thread_local VectorXd localGpValues2D_DPU;
        static thread_local VectorXd localGpValues2D_DPU_s;
        static thread_local VectorXd localGpValues2D_DPU_t;
        static thread_local VectorXd localGpValues2D_DPU_Interpolation;
        static thread_local VectorXd localGpValues2D_DPU_Interpolation_s;
        static thread_local VectorXd localGpValues2D_DPU_Interpolation_t;
        static thread_local VectorXd localGpValues2D_DPU_Minus_Interpolation;
        static thread_local VectorXd localGpValues2D_DPU_Minus_Interpolation_s;
        static thread_local VectorXd localGpValues2D_DPU_Minus_Interpolation_t;

        static thread_local MatrixXd localGpSelected;
        static thread_local MatrixXd localGpSelected_s;
        static thread_local MatrixXd localGpSelected_t;

        static thread_local MatrixXd PU_Group_selected;
        static thread_local MatrixXd localPUvalues;
        static thread_local MatrixXd localPUvalues_s;
        static thread_local MatrixXd localPUvalues_t;

        // 预分配缓冲区用于DPU相关矩阵拼接优化
        static thread_local MatrixXd buffer_C_DPU;
        static thread_local MatrixXd buffer_C_s_DPU;
        static thread_local MatrixXd buffer_C_t_DPU;

        // 预分配缓冲区用于PU相关矩阵拼接优化
        static thread_local MatrixXd buffer_C_PU;
        static thread_local MatrixXd buffer_C_PU_s;
        static thread_local MatrixXd buffer_C_PU_t;

        static thread_local MatrixXd DMutis_x;
        static thread_local MatrixXd DMutis_y;
        static thread_local MatrixXd DMutis_Cof;
        static thread_local MatrixXd DMutis;

        static thread_local MatrixXd DMutis0_x;
        static thread_local MatrixXd DMutis0_y;
        static thread_local MatrixXd DMutis0_Cof;
        static thread_local MatrixXd DMutis0;

        static thread_local MatrixXd DMutis1_x;
        static thread_local MatrixXd DMutis1_y;
        static thread_local MatrixXd DMutis1_Cof;
        static thread_local MatrixXd DMutis1;

        static thread_local MatrixXd DMutis_Interpolation;
        static thread_local MatrixXd DMutis_Interpolation_x;
        static thread_local MatrixXd DMutis_Interpolation_y;

        static thread_local MatrixXd DMutis0_Interpolation;
        static thread_local MatrixXd DMutis0_Interpolation_x;
        static thread_local MatrixXd DMutis0_Interpolation_y;
        static thread_local MatrixXd DMutis1_Interpolation;
        static thread_local MatrixXd DMutis1_Interpolation_x;
        static thread_local MatrixXd DMutis1_Interpolation_y;

        static thread_local MatrixXd DMutis_Minus_Interpolation;
        static thread_local MatrixXd DMutis_Minus_Interpolation_x;
        static thread_local MatrixXd DMutis_Minus_Interpolation_y;

        static thread_local MatrixXd DMutis0_Minus_Interpolation;
        static thread_local MatrixXd DMutis0_Minus_Interpolation_x;
        static thread_local MatrixXd DMutis0_Minus_Interpolation_y;

        static thread_local MatrixXd DMutis1_Minus_Interpolation;
        static thread_local MatrixXd DMutis1_Minus_Interpolation_x;
        static thread_local MatrixXd DMutis1_Minus_Interpolation_y;

        static thread_local VectorXd scalar_vector;
        static thread_local VectorXd scalar_vector0;
        static thread_local VectorXd scalar_vector1;
        static thread_local MatrixXd scaled_matrix;
        static thread_local MatrixXd scaled_matrix_x;
        static thread_local MatrixXd scaled_matrix_y;

        static thread_local MatrixXd scaled_matrix_0;
        static thread_local MatrixXd scaled_matrix_0_x;
        static thread_local MatrixXd scaled_matrix_0_y;

        static thread_local MatrixXd scaled_matrix_1;
        static thread_local MatrixXd scaled_matrix_1_x;
        static thread_local MatrixXd scaled_matrix_1_y;
        bool enrich_flag = false;
        ElemMap ElemMap(N, order_p);

        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                if (inter[i][j] == 1) {
                    continue;
                }
                
                int elemID = ElemMap.getElemId(i, j);
                locFEMFuncs = ElemMap.getFEMBasisIds(elemID);
                globalGps = ElemMap.localToGlobal(elemID, localGps2D);
                globalInterpolationPoints2D = ElemMap.localToGlobal(elemID, InterpolationPoints2D);

                if (EnrichMode == 1) {
                    int pos;
                    enrich_flag = Ismember_single(elemID, echFaceNodeFreq, 0, pos);
                    if (enrich_flag) {
                        vector<int> vertexNodeIds = { echFaceNodeFreq[pos][5],echFaceNodeFreq[pos][6],echFaceNodeFreq[pos][7],echFaceNodeFreq[pos][8] };
                        vector<size_t> nzIds = Find(vertexNodeIds);     //从零开始 无须加减一
                        vector<vector<int>> faceIds = ElemMap.getOneRingFaces(elemID);
                        vector<int> FaceIds;
                        for (size_t col = 0; col < faceIds[0].size(); ++col) {  // 先遍历列
                            for (size_t row = 0; row < faceIds.size(); ++row) {  // 再遍历行
                                if (col < faceIds[row].size()) {  // 防止列数不一致
                                    FaceIds.push_back(faceIds[row][col]);
                                }
                            }
                        }
                        vector<int> locPUFuncs;
                        vector<bool> Flag;
                        vector<int> EchFuncId;
                        ismember(FaceIds, IntElem, Flag, EchFuncId);
                        vector<size_t> fnzIds = Find(EchFuncId);
                        for (size_t k = 0; k < fnzIds.size(); ++k) {
                            echLocalPUFunctions[pos].push_back(EchFuncId[fnzIds[k]]);
                            locPUFuncs.push_back(EchFuncId[fnzIds[k]] - 1);
                        }
                        vector<int> node_vector;
                        vector<int> echVerNodeFreq_col;
                        for (int col_idx : nzIds) {
                            if (col_idx >= 0 && col_idx < echFaceNodeFreq[pos].size()) {
                                node_vector.push_back(echFaceNodeFreq[pos][col_idx + 1]);
                            }
                        }
                        for (const auto& pair : echVerNodeFreq) {
                            echVerNodeFreq_col.push_back(pair.first);
                        }
                        vector<bool> Flag2;
                        vector<int> EchFuncId2;
                        ismember(node_vector, echVerNodeFreq_col, Flag2, EchFuncId2);  //需要减一
                        vector<double> vFreq;
                        for (int col_idx : EchFuncId2) {
                            double num = echVerNodeFreq[col_idx - 1].second;
                            vFreq.push_back(1 / num);
                        }

                        localGpSelected = localGpValues2D_PU(Eigen::all, nzIds);
                        localGpSelected_s = localGpValues2D_PU_s(Eigen::all, nzIds);
                        localGpSelected_t = localGpValues2D_PU_t(Eigen::all, nzIds);

                        // 创建vFreq向量
                        Eigen::VectorXd vFreqVec(vFreq.size());

                        // 使用 for 循环逐元素赋值
                        for (size_t i = 0; i < vFreq.size(); ++i) {
                            vFreqVec(i) = vFreq[i];  // 注意：Eigen 使用 `(i)` 而不是 `[i]` 访问元素
                        }


                        // 提取PU_Group的子矩阵（fnzIds列）
                        PU_Group_selected = PU_Group(nzIds, fnzIds);

                        // 计算 vFreq .* PU_Group(nzIds, fnzIds)
                        // 由于 vFreq 和 PU_Group_selected 的列数可能不同，需要确保维度匹配
                        // 假设 vFreq.size() == PU_Group_selected.cols()（即 nzIds 和 fnzIds 长度相同）
                        Eigen::MatrixXd term1(PU_Group_selected.rows(), PU_Group_selected.cols());
                        for (int j = 0; j < PU_Group_selected.rows(); ++j) {
                            term1.row(j) = vFreqVec(j) * PU_Group_selected.row(j);
                        }

                        // 计算最终结果
                        localPUvalues = localGpSelected * term1;
                        localPUvalues_s = localGpSelected_s * term1;
                        localPUvalues_t = localGpSelected_t * term1;

                        echLocalPUValues[pos][0] = localPUvalues;
                        echLocalPUValues[pos][1] = localPUvalues_s;
                        echLocalPUValues[pos][2] = localPUvalues_t;
                        int N1 = enrichMutiplyN;
                        int M1 = locPUFuncs.size();      // 列向量长度
                        int K1 = fnzIds.size();          // 列向量长度

                        // 1. locEnrFuncs 矩阵： N1 × M1（列优先）
                        std::vector<int> locEnrMat(N1 * M1);
                        for (int r = 0; r < N1; ++r)           // 行
                            for (int c = 0; c < M1; ++c)       // 列
                                locEnrMat[r + c * N1] = locPUFuncs[c] * N1;

                        // 2. offset 矩阵： K1 × N1（列优先）
                        std::vector<int> offsetMat(K1 * N1);
                        for (int r = 0; r < N1; ++r)
                            for (int c = 0; c < K1; ++c)
                                offsetMat[c * N1 + r] = r + 1;


                        // 3. 逐元素相加（MATLAB 会广播到同长度）
                        std::vector<int> locEnrFuncs;
                        locEnrFuncs.reserve(std::max(N1 * M1, K1 * N1));
                        for (size_t i = 0; i < std::min(N1 * M1, K1 * N1); ++i)
                            locEnrFuncs.push_back(locEnrMat[i] + offsetMat[i]);

                        // 4. 加全局偏移
                        for (int& v : locEnrFuncs) v += globalFEMFunctionN;

                        echLocalDPUMap[pos] = elemID;
                        echLocalDPUFunctions[pos] = locEnrFuncs;
                        MatrixXd localGpValues2D_DPU_Minus_Interpolation(globalGps.rows(), locEnrFuncs.size());
                        MatrixXd localGpValues2D_DPU_Minus_Interpolation_x(globalGps.rows(), locEnrFuncs.size());
                        MatrixXd localGpValues2D_DPU_Minus_Interpolation_y(globalGps.rows(), locEnrFuncs.size());

                        for (int k = 0; k < fnzIds.size(); k++)
                        {
                            int fid = fnzIds[k];
                            double x0, y0 = 0;
                            ElemMap.LocalOriginCoordPU(elemID, fid + 1, x0, y0);

                            if (!Flag_oldxy) {
                                GenerateOffsetDMutis(globalGps, globalInterpolationPoints2D, order_p,
                                    x0, y0, 3 * h, 3 * h,
                                    DMutis, DMutis_x, DMutis_y, DMutis_Cof);
                            }
                            else {
                                GenerateOffsetDMutis(globalGps, globalInterpolationPoints2D, order_p,
                                    0, 0, 1, 1,
                                    DMutis, DMutis_x, DMutis_y, DMutis_Cof);
                            }
                            DMutis_Interpolation = localGpValues2D * DMutis_Cof;
                            DMutis_Interpolation_x = localGpValues2D_s * DMutis_Cof * N;
                            DMutis_Interpolation_y = localGpValues2D_t * DMutis_Cof * N;
                            DMutis_Minus_Interpolation = DMutis - DMutis_Interpolation;
                            DMutis_Minus_Interpolation_x = DMutis_x - DMutis_Interpolation_x;
                            DMutis_Minus_Interpolation_y = DMutis_y - DMutis_Interpolation_y;
                            int start_col = (k)*enrichMutiplyN;
                            scalar_vector = localPUvalues.col(k);

                            scaled_matrix = DMutis_Minus_Interpolation.cwiseProduct(scalar_vector.replicate(1, DMutis_Minus_Interpolation.cols()));
                            localGpValues2D_DPU_Minus_Interpolation.block(0, start_col,
                                localGpValues2D_DPU_Minus_Interpolation.rows(), enrichMutiplyN) = scaled_matrix;
                            scaled_matrix_x = DMutis_Minus_Interpolation_x.cwiseProduct(localPUvalues.col(k).replicate(1, DMutis_Minus_Interpolation_x.cols())) + DMutis_Minus_Interpolation.cwiseProduct(localPUvalues_s.col(k).replicate(1, DMutis_Minus_Interpolation.cols())) * N;
                            localGpValues2D_DPU_Minus_Interpolation_x.block(0, start_col,
                                localGpValues2D_DPU_Minus_Interpolation_x.rows(), enrichMutiplyN) = scaled_matrix_x;
                            scaled_matrix_y = DMutis_Minus_Interpolation_y.cwiseProduct(localPUvalues.col(k).replicate(1, DMutis_Minus_Interpolation_y.cols())) + DMutis_Minus_Interpolation_y.cwiseProduct(localPUvalues_t.col(k).replicate(1, DMutis_Minus_Interpolation.cols())) * N;
                            localGpValues2D_DPU_Minus_Interpolation_y.block(0, start_col,
                                localGpValues2D_DPU_Minus_Interpolation_y.rows(), enrichMutiplyN) = scaled_matrix_y;

                        }
                        echElemGpsValues[pos][0] = localGpValues2D_DPU_Minus_Interpolation;
                        echElemGpsValues[pos][1] = localGpValues2D_DPU_Minus_Interpolation_x;
                        echElemGpsValues[pos][2] = localGpValues2D_DPU_Minus_Interpolation_y;
                    }

                }

                // Perform calculations for each element
                // ...
            }
        }

        vector<vector<MatrixXd>> intElemGps(numIntElem, std::vector<Eigen::MatrixXd>(2));
        vector<vector<MatrixXd>> intElemGpsVal(numIntElem, std::vector<Eigen::MatrixXd>(3));
        int PUDim = 2;
        static thread_local VectorXd xcoord;
        static thread_local VectorXd ycoord;
        static thread_local MatrixXd Gps0;
        static thread_local MatrixXd Gps1;
        static thread_local MatrixXd Jac0;
        static thread_local MatrixXd Jac1;
        static thread_local MatrixXd GpW0;
        static thread_local MatrixXd GpW1;
        static thread_local MatrixXd combined0;
        static thread_local MatrixXd combined1;
        static thread_local MatrixXd locGps0;
        static thread_local MatrixXd locGps1;
        
        // 预分配缓冲区用于矩阵拼接优化
        static thread_local MatrixXd buffer_C;
        static thread_local MatrixXd buffer_C_s;
        static thread_local MatrixXd buffer_C_t;



        for (int k = 0; k < numIntElem; k++)
        {
            int elemID = IntElem[k];
            
            globalInterpolationPoints2D = ElemMap.localToGlobal(elemID, InterpolationPoints2D);
            auto ij = ElemMap.calcIndex(elemID);
            int i = ij[0]; int j = ij[1];
            xcoord = IntElemCofx.row(k);
            VectorXd ycoord = IntElemCofy.row(k);
            IntegrationElement integrator;
            auto result = integrator.intElemIntegration(up_or_blew[i - 1][j - 1], i, j, N, xcoord, ycoord);
            Gps0 = result.Gps0;
            Gps1 = result.Gps1;
            Jac0 = result.Jac0;
            Jac1 = result.Jac1;
            GpW0 = result.GpW0;
            GpW1 = result.GpW1;
            combined0 = MatrixXd::Zero(Gps0.rows(), Gps0.cols() + GpW0.cols() + Jac0.cols());
            combined1 = MatrixXd::Zero(Gps1.rows(), Gps1.cols() + GpW1.cols() + Jac1.cols());
            combined0 << Gps0, GpW0, Jac0;
            combined1 << Gps1, GpW1, Jac1;
            intElemGps[k][0] = combined0;
            intElemGps[k][1] = combined1;
            double x0 = (j - 1) * h;
            double y0 = (i - 1) * h;
            locGps0 = (Gps0.rowwise() - Eigen::RowVector2d(x0, y0)) * N;
            locGps1 = (Gps1.rowwise() - Eigen::RowVector2d(x0, y0)) * N;
            vector<int> locFEMFuncs = ElemMap.getFEMBasisIds(elemID);
            MatrixXd globalGps = ElemMap.localToGlobal(elemID, localGps2D);
            MatrixXd shape_func00 = BasisFunction::FEMShapeFunction(locGps0.col(0), order_p, 0);
            MatrixXd shape_func01 = BasisFunction::FEMShapeFunction(locGps0.col(1), order_p, 0);
            MatrixXd shape_func10 = BasisFunction::FEMShapeFunction(locGps1.col(0), order_p, 0);
            MatrixXd shape_func11 = BasisFunction::FEMShapeFunction(locGps1.col(1), order_p, 0);
            MatrixXd shape_func00_s = BasisFunction::FEMShapeFunction(locGps0.col(0), order_p, 1);
            MatrixXd shape_func01_t = BasisFunction::FEMShapeFunction(locGps0.col(1), order_p, 1);
            MatrixXd shape_func10_s = BasisFunction::FEMShapeFunction(locGps1.col(0), order_p, 1);
            MatrixXd shape_func11_t = BasisFunction::FEMShapeFunction(locGps1.col(1), order_p, 1);

            MatrixXd shape_func0 = MatrixXd::Zero(Gps0.rows(), (order_p + 1) * (order_p + 1));
            MatrixXd shape_func1 = MatrixXd::Zero(Gps1.rows(), (order_p + 1) * (order_p + 1));
            MatrixXd shape_func0_s = MatrixXd::Zero(Gps0.rows(), (order_p + 1) * (order_p + 1));
            MatrixXd shape_func0_t = MatrixXd::Zero(Gps0.rows(), (order_p + 1) * (order_p + 1));
            MatrixXd shape_func1_s = MatrixXd::Zero(Gps1.rows(), (order_p + 1) * (order_p + 1));
            MatrixXd shape_func1_t = MatrixXd::Zero(Gps1.rows(), (order_p + 1) * (order_p + 1));
            for (int m = 0; m < order_p + 1; ++m) {
                for (int n = 0; n < order_p + 1; ++n) {
                    // Element-wise multiplication
                    MatrixXd temp0 = shape_func00_s.col(n).cwiseProduct(shape_func01.col(m));
                    MatrixXd temp1 = shape_func00.col(n).cwiseProduct(shape_func01_t.col(m));
                    MatrixXd temp2 = shape_func10_s.col(n).cwiseProduct(shape_func11.col(m));
                    MatrixXd temp3 = shape_func10.col(n).cwiseProduct(shape_func11_t.col(m));
                    MatrixXd temp4 = shape_func00.col(n).cwiseProduct(shape_func01.col(m));
                    MatrixXd temp5 = shape_func10.col(n).cwiseProduct(shape_func11.col(m));

                    // Assign to output matrices
                    int col_idx = n * (order_p + 1) + m;
                    shape_func0_s.col(col_idx) = temp0;
                    shape_func0_t.col(col_idx) = temp1;
                    shape_func1_s.col(col_idx) = temp2;
                    shape_func1_t.col(col_idx) = temp3;
                    shape_func0.col(col_idx) = temp4;
                    shape_func1.col(col_idx) = temp5;
                }
            }
            // 使用预分配的缓冲区进行矩阵拼接优化
            buffer_C.conservativeResize(shape_func0.rows() + shape_func1.rows(), shape_func0.cols());
            buffer_C << shape_func0,
                shape_func1;
            intElemGpsVal[k][0] = buffer_C;

            // 纵向组合 shape_func0_s 和 shape_func1_s
            buffer_C_s.conservativeResize(shape_func0_s.rows() + shape_func1_s.rows(), shape_func0_s.cols());
            buffer_C_s << shape_func0_s,
                shape_func1_s;
            intElemGpsVal[k][1] = buffer_C_s;

            // 纵向组合 shape_func0_t 和 shape_func1_t
            buffer_C_t.conservativeResize(shape_func0_t.rows() + shape_func1_t.rows(), shape_func0_t.cols());
            buffer_C_t << shape_func0_t,
                shape_func1_t;
            intElemGpsVal[k][2] = buffer_C_t;

            MatrixXd shape_func00_PU = BasisFunction::PUShapeFunction(locGps0.col(0), 0);
            MatrixXd shape_func10_PU = BasisFunction::PUShapeFunction(locGps1.col(0), 0);
            MatrixXd shape_func01_PU = BasisFunction::PUShapeFunction(locGps0.col(1), 0);
            MatrixXd shape_func11_PU = BasisFunction::PUShapeFunction(locGps1.col(1), 0);
            MatrixXd shape_func00_PU_s = BasisFunction::PUShapeFunction(locGps0.col(0), 1);
            MatrixXd shape_func01_PU_t = BasisFunction::PUShapeFunction(locGps0.col(1), 1);
            MatrixXd shape_func10_PU_s = BasisFunction::PUShapeFunction(locGps1.col(0), 1);
            MatrixXd shape_func11_PU_t = BasisFunction::PUShapeFunction(locGps1.col(1), 1);

            MatrixXd shape_func0_PU = MatrixXd::Zero(Gps0.rows(), PUDim * PUDim);
            MatrixXd shape_func1_PU = MatrixXd::Zero(Gps1.rows(), PUDim * PUDim);
            MatrixXd shape_func0_PU_s = MatrixXd::Zero(Gps0.rows(), PUDim * PUDim);
            MatrixXd shape_func1_PU_s = MatrixXd::Zero(Gps1.rows(), PUDim * PUDim);
            MatrixXd shape_func0_PU_t = MatrixXd::Zero(Gps0.rows(), PUDim * PUDim);
            MatrixXd shape_func1_PU_t = MatrixXd::Zero(Gps1.rows(), PUDim * PUDim);

            int PUDim = shape_func00_PU.cols();

            for (int m = 0; m < PUDim; ++m) {
                for (int n = 0; n < PUDim; ++n) {
                    // Element-wise multiplication
                    MatrixXd temp0 = shape_func00_PU_s.col(n).cwiseProduct(shape_func01_PU.col(m));
                    MatrixXd temp1 = shape_func00_PU.col(n).cwiseProduct(shape_func01_PU_t.col(m));
                    MatrixXd temp2 = shape_func10_PU_s.col(n).cwiseProduct(shape_func11_PU.col(m));
                    MatrixXd temp3 = shape_func10_PU.col(n).cwiseProduct(shape_func11_PU_t.col(m));
                    MatrixXd temp4 = shape_func00_PU.col(n).cwiseProduct(shape_func01_PU.col(m));
                    MatrixXd temp5 = shape_func10_PU.col(n).cwiseProduct(shape_func11_PU.col(m));

                    // Assign to output matrices

                    int col_idx = n * PUDim + m;
                    shape_func0_PU_s.col(col_idx) = temp0;
                    shape_func0_PU_t.col(col_idx) = temp1;
                    shape_func1_PU_s.col(col_idx) = temp2;
                    shape_func1_PU_t.col(col_idx) = temp3;
                    shape_func0_PU.col(col_idx) = temp4;
                    shape_func1_PU.col(col_idx) = temp5;

                }
            }

            bool enrich_flag = false;
            if (EnrichMode == 1) {
                int pos;
                enrich_flag = Ismember_single(elemID, echFaceNodeFreq, 0, pos);
                if (enrich_flag) {
                    vector<int> vertexNodeIds = { echFaceNodeFreq[pos][5],echFaceNodeFreq[pos][6],echFaceNodeFreq[pos][7],echFaceNodeFreq[pos][8] };
                    vector<size_t> nzIds = Find(vertexNodeIds);     //从零开始 无须加减一
                    vector<vector<int>> faceIds = ElemMap.getOneRingFaces(elemID);
                    vector<int> FaceIds;
                    for (size_t col = 0; col < faceIds[0].size(); ++col) {  // 先遍历列
                        for (size_t row = 0; row < faceIds.size(); ++row) {  // 再遍历行
                            if (col < faceIds[row].size()) {  // 防止列数不一致
                                FaceIds.push_back(faceIds[row][col]);
                            }
                        }
                    }
                    vector<int> locPUFuncs;
                    vector<bool> Flag;
                    vector<int> EchFuncId;
                    ismember(FaceIds, IntElem, Flag, EchFuncId);
                    vector<size_t> fnzIds = Find(EchFuncId);
                    for (size_t k = 0; k < fnzIds.size(); ++k) {
                        echLocalPUFunctions[pos].push_back(EchFuncId[fnzIds[k]]);
                        locPUFuncs.push_back(EchFuncId[fnzIds[k]] - 1);
                    }
                    vector<int> node_vector;
                    vector<int> echVerNodeFreq_col;
                    for (int col_idx : nzIds) {
                        if (col_idx >= 0 && col_idx < echFaceNodeFreq[pos].size()) {
                            node_vector.push_back(echFaceNodeFreq[pos][col_idx + 1]);
                        }
                    }
                    for (const auto& pair : echVerNodeFreq) {
                        echVerNodeFreq_col.push_back(pair.first);
                    }
                    vector<bool> Flag2;
                    vector<int> EchFuncId2;
                    ismember(node_vector, echVerNodeFreq_col, Flag2, EchFuncId2);  //需要减一
                    vector<double> vFreq;
                    for (int col_idx : EchFuncId2) {
                        double num = echVerNodeFreq[col_idx - 1].second;
                        vFreq.push_back(1 / num);
                    }




                    // 创建vFreq向量
                    Eigen::VectorXd vFreqVec(vFreq.size());

                    // 使用 for 循环逐元素赋值
                    for (size_t i = 0; i < vFreq.size(); ++i) {
                        vFreqVec(i) = vFreq[i];  // 注意：Eigen 使用 `(i)` 而不是 `[i]` 访问元素
                    }


                    // 提取PU_Group的子矩阵（fnzIds列）
                    Eigen::MatrixXd PU_Group_selected = PU_Group(nzIds, fnzIds);

                    // 计算 vFreq .* PU_Group(nzIds, fnzIds)
                    // 由于 vFreq 和 PU_Group_selected 的列数可能不同，需要确保维度匹配
                    // 假设 vFreq.size() == PU_Group_selected.cols()（即 nzIds 和 fnzIds 长度相同）
                    Eigen::MatrixXd term1(PU_Group_selected.rows(), PU_Group_selected.cols());
                    for (int j = 0; j < PU_Group_selected.rows(); ++j) {
                        term1.row(j) = vFreqVec(j) * PU_Group_selected.row(j);
                    }

                    Eigen::MatrixXd shape_func0_PUSelected = shape_func0_PU(Eigen::all, nzIds);
                    Eigen::MatrixXd shape_func0_PU_sSelected = shape_func0_PU_s(Eigen::all, nzIds);
                    Eigen::MatrixXd shape_func0_PU_tSelected = shape_func0_PU_t(Eigen::all, nzIds);
                    Eigen::MatrixXd shape_func1_PUSelected = shape_func1_PU(Eigen::all, nzIds);
                    Eigen::MatrixXd shape_func1_PU_sSelected = shape_func1_PU_s(Eigen::all, nzIds);
                    Eigen::MatrixXd shape_func1_PU_tSelected = shape_func1_PU_t(Eigen::all, nzIds);
                    // 计算最终结果
                    Eigen::MatrixXd localPUvalues0 = shape_func0_PUSelected * term1;
                    Eigen::MatrixXd localPUvalues0_s = shape_func0_PU_sSelected * term1;
                    Eigen::MatrixXd localPUvalues0_t = shape_func0_PU_tSelected * term1;
                    Eigen::MatrixXd localPUvalues1 = shape_func1_PUSelected * term1;
                    Eigen::MatrixXd localPUvalues1_s = shape_func1_PU_sSelected * term1;
                    Eigen::MatrixXd localPUvalues1_t = shape_func1_PU_tSelected * term1;

                    // 使用预分配的缓冲区进行矩阵拼接优化
                    buffer_C_PU.conservativeResize(localPUvalues0.rows() + localPUvalues1.rows(), localPUvalues0.cols());
                    buffer_C_PU << localPUvalues0,
                        localPUvalues1;

                    // 纵向组合 localPUvalues0_s 和 localPUvalues1_s
                    buffer_C_PU_s.conservativeResize(localPUvalues0_s.rows() + localPUvalues1_s.rows(), localPUvalues0_s.cols());
                    buffer_C_PU_s << localPUvalues0_s,
                        localPUvalues1_s;

                    // 纵向组合 localPUvalues0_t 和 localPUvalues1_t
                    buffer_C_PU_t.conservativeResize(localPUvalues0_t.rows() + localPUvalues1_t.rows(), localPUvalues0_t.cols());
                    buffer_C_PU_t << localPUvalues0_t,
                        localPUvalues1_t;

                    echLocalPUValues[pos][0] = buffer_C_PU;
                    echLocalPUValues[pos][1] = buffer_C_PU_s;
                    echLocalPUValues[pos][2] = buffer_C_PU_t;
                    int N1 = enrichMutiplyN;
                    int M1 = locPUFuncs.size();      // 列向量长度
                    int K1 = fnzIds.size();          // 列向量长度

                    // 1. locEnrFuncs 矩阵： N1 × M1（列优先）
                    std::vector<int> locEnrMat(N1 * M1);
                    for (int r = 0; r < N1; ++r)           // 行
                        for (int c = 0; c < M1; ++c)       // 列
                            locEnrMat[r + c * N1] = locPUFuncs[c] * N1;

                    // 2. offset 矩阵： K1 × N1（列优先）
                    std::vector<int> offsetMat(K1 * N1);
                    for (int r = 0; r < N1; ++r)
                        for (int c = 0; c < K1; ++c)
                            offsetMat[c * N1 + r] = r + 1;


                    // 3. 逐元素相加（MATLAB 会广播到同长度）
                    std::vector<int> locEnrFuncs;
                    locEnrFuncs.reserve(std::max(N1 * M1, K1 * N1));
                    for (size_t i = 0; i < std::min(N1 * M1, K1 * N1); ++i)
                        locEnrFuncs.push_back(locEnrMat[i] + offsetMat[i]);

                    // 4. 加全局偏移
                    for (int& v : locEnrFuncs) v += globalFEMFunctionN;
                    echLocalDPUFunctions[pos] = locEnrFuncs;
                    echLocalDPUMap[pos] = elemID;
                    MatrixXd shape_func0_DPU_Minus_Interpolation = MatrixXd::Zero(Gps0.rows(), locEnrFuncs.size());
                    MatrixXd shape_func0_DPU_Minus_Interpolation_x = MatrixXd::Zero(Gps0.rows(), locEnrFuncs.size());
                    MatrixXd shape_func0_DPU_Minus_Interpolation_y = MatrixXd::Zero(Gps0.rows(), locEnrFuncs.size());
                    MatrixXd shape_func1_DPU_Minus_Interpolation = MatrixXd::Zero(Gps1.rows(), locEnrFuncs.size());
                    MatrixXd shape_func1_DPU_Minus_Interpolation_x = MatrixXd::Zero(Gps1.rows(), locEnrFuncs.size());
                    MatrixXd shape_func1_DPU_Minus_Interpolation_y = MatrixXd::Zero(Gps1.rows(), locEnrFuncs.size());

                    for (int k = 0; k < fnzIds.size(); k++)
                    {
                        int fid = fnzIds[k];
                        double x0, y0 = 0;
                        ElemMap.LocalOriginCoordPU(elemID, fid + 1, x0, y0);
                        
                        if (!Flag_oldxy) {
                            GenerateOffsetDMutis(Gps0, globalInterpolationPoints2D, order_p,
                                x0, y0, 3 * h, 3 * h,
                                DMutis0, DMutis0_x, DMutis0_y, DMutis0_Cof);
                            GenerateOffsetDMutis(Gps1, globalInterpolationPoints2D, order_p,
                                x0, y0, 3 * h, 3 * h,
                                DMutis1, DMutis1_x, DMutis1_y, DMutis1_Cof);


                        }
                        else {
                            GenerateOffsetDMutis(Gps0, globalInterpolationPoints2D, order_p,
                                0, 0, 1, 1,
                                DMutis0, DMutis0_x, DMutis0_y, DMutis0_Cof);
                            GenerateOffsetDMutis(Gps1, globalInterpolationPoints2D, order_p,
                                0, 0, 1, 1,
                                DMutis0, DMutis0_x, DMutis0_y, DMutis0_Cof);
                        }
                        DMutis0_Interpolation = shape_func0 * DMutis0_Cof;
                        DMutis0_Interpolation_x = shape_func0_s * DMutis0_Cof * N;
                        DMutis0_Interpolation_y = shape_func0_t * DMutis0_Cof * N;
                        DMutis0_Minus_Interpolation = DMutis0 - DMutis0_Interpolation;
                        DMutis0_Minus_Interpolation_x = DMutis0_x - DMutis0_Interpolation_x;
                        DMutis0_Minus_Interpolation_y = DMutis0_y - DMutis0_Interpolation_y;

                        DMutis1_Interpolation = shape_func1 * DMutis1_Cof;
                        DMutis1_Interpolation_x = shape_func1_s * DMutis1_Cof * N;
                        DMutis1_Interpolation_y = shape_func1_t * DMutis1_Cof * N;
                        DMutis1_Minus_Interpolation = DMutis1 - DMutis1_Interpolation;
                        DMutis1_Minus_Interpolation_x = DMutis1_x - DMutis1_Interpolation_x;
                        DMutis1_Minus_Interpolation_y = DMutis1_y - DMutis1_Interpolation_y;

                        int start_col = (k)*enrichMutiplyN;   //这里k也不减一  从零开始

                        scaled_matrix_0 = DMutis0_Minus_Interpolation.cwiseProduct(localPUvalues0.col(k).replicate(1, DMutis0_Minus_Interpolation.cols()));
                        scaled_matrix_0_x = DMutis0_Minus_Interpolation_x.cwiseProduct(localPUvalues0.col(k).replicate(1, DMutis0_Minus_Interpolation_x.cols())) + DMutis0_Minus_Interpolation.cwiseProduct(localPUvalues0_s.col(k).replicate(1, DMutis0_Minus_Interpolation.cols())) * N;
                        scaled_matrix_0_y = DMutis0_Minus_Interpolation_y.cwiseProduct(localPUvalues0.col(k).replicate(1, DMutis0_Minus_Interpolation_y.cols())) + DMutis0_Minus_Interpolation.cwiseProduct(localPUvalues0_t.col(k).replicate(1, DMutis0_Minus_Interpolation.cols())) * N;
                        scaled_matrix_1 = DMutis1_Minus_Interpolation.cwiseProduct(localPUvalues1.col(k).replicate(1, DMutis1_Minus_Interpolation.cols()));
                        scaled_matrix_1_x = DMutis1_Minus_Interpolation_x.cwiseProduct(localPUvalues1.col(k).replicate(1, DMutis1_Minus_Interpolation_x.cols())) + DMutis1_Minus_Interpolation.cwiseProduct(localPUvalues1_s.col(k).replicate(1, DMutis1_Minus_Interpolation.cols())) * N;
                        scaled_matrix_1_y = DMutis1_Minus_Interpolation_y.cwiseProduct(localPUvalues1.col(k).replicate(1, DMutis1_Minus_Interpolation_y.cols())) + DMutis1_Minus_Interpolation.cwiseProduct(localPUvalues1_t.col(k).replicate(1, DMutis1_Minus_Interpolation.cols())) * N;

                        shape_func0_DPU_Minus_Interpolation.block(0, start_col,
                            shape_func0_DPU_Minus_Interpolation.rows(), enrichMutiplyN) = scaled_matrix_0;

                        shape_func0_DPU_Minus_Interpolation_x.block(0, start_col,
                            shape_func0_DPU_Minus_Interpolation.rows(), enrichMutiplyN) = scaled_matrix_0_x;

                        shape_func0_DPU_Minus_Interpolation_y.block(0, start_col,
                            shape_func0_DPU_Minus_Interpolation.rows(), enrichMutiplyN) = scaled_matrix_0_y;

                        shape_func1_DPU_Minus_Interpolation.block(0, start_col,
                            shape_func1_DPU_Minus_Interpolation.rows(), enrichMutiplyN) = scaled_matrix_1;

                        shape_func1_DPU_Minus_Interpolation_x.block(0, start_col,
                            shape_func1_DPU_Minus_Interpolation.rows(), enrichMutiplyN) = scaled_matrix_1_x;

                        shape_func1_DPU_Minus_Interpolation_y.block(0, start_col,
                            shape_func1_DPU_Minus_Interpolation.rows(), enrichMutiplyN) = scaled_matrix_1_y;


                    }

                    // 使用预分配的缓冲区进行矩阵拼接优化
                    buffer_C_DPU.conservativeResize(shape_func0_DPU_Minus_Interpolation.rows() + shape_func1_DPU_Minus_Interpolation.rows(),
                        shape_func0_DPU_Minus_Interpolation.cols());
                    buffer_C_DPU << shape_func0_DPU_Minus_Interpolation,
                        shape_func1_DPU_Minus_Interpolation;
                    echElemGpsValues[pos][0] = buffer_C_DPU;

                    // 纵向组合 shape_func0_DPU_Minus_Interpolation_x 和 shape_func1_DPU_Minus_Interpolation_x
                    buffer_C_s_DPU.conservativeResize(shape_func0_DPU_Minus_Interpolation_x.rows() + shape_func1_DPU_Minus_Interpolation_x.rows(),
                        shape_func0_DPU_Minus_Interpolation_x.cols());
                    buffer_C_s_DPU << shape_func0_DPU_Minus_Interpolation_x,
                        shape_func1_DPU_Minus_Interpolation_x;
                    echElemGpsValues[pos][1] = buffer_C_s_DPU;

                    // 纵向组合 shape_func0_DPU_Minus_Interpolation_y 和 shape_func1_DPU_Minus_Interpolation_y
                    buffer_C_t_DPU.conservativeResize(shape_func0_DPU_Minus_Interpolation_y.rows() + shape_func1_DPU_Minus_Interpolation_y.rows(),
                        shape_func0_DPU_Minus_Interpolation_y.cols());
                    buffer_C_t_DPU << shape_func0_DPU_Minus_Interpolation_y,
                        shape_func1_DPU_Minus_Interpolation_y;
                    echElemGpsValues[pos][2] = buffer_C_t_DPU;
                }
            }

        }

        vector<vector<MatrixXd>> OrthogonalMatrix(numIntElem, vector<Eigen::MatrixXd>(3));
        if (EnrichMode == 1 && OrthogonalMode)
        {
            for (int k = 0; k < numIntElem; k++)
            {
                int elemID = IntElem[k];
                vector<int> enrfuns;
                for (int i = 0; i < enrichMutiplyN; ++i)
                {
                    enrfuns.push_back(globalFEMFunctionN + (k)*enrichMutiplyN + (i + 1));                //我认为富集函数的序号也是要代入得 所以不加一
                }
                
                auto ij = ElemMap.calcIndex(elemID);
                int i = ij[0]; int j = ij[1];
                vector<vector<int>> faceIds = ElemMap.getOneRingFaces(elemID);
                vector<int> FaceIds;
                for (size_t col = 0; col < faceIds[0].size(); ++col) {  // 先遍历列
                    for (size_t row = 0; row < faceIds.size(); ++row) {  // 再遍历行
                        if (col < faceIds[row].size()) {  // 防止列数不一致
                            FaceIds.push_back(faceIds[row][col]);
                        }
                    }
                }

                int sizeGpsPoints = 0;
                vector<int> sizeGpsPointsList;
                MatrixXd gpslist;
                MatrixXd valuelist;
                MatrixXd valuelist_x;
                MatrixXd valuelist_y;
                VectorXd weightlist;

                for (int fid : FaceIds)
                {
                    if (fid == 0)
                    {
                        continue;
                    }

                    auto result = ismember_single(fid, echLocalDPUMap);
                    int pos = result.second;
                    vector<int> locEnrFuncs;
                    for (int val : echLocalDPUFunctions[pos]) {
                        locEnrFuncs.push_back(val);
                    }
                    vector<bool> Flag;
                    vector<int> posF;
                    ismember(enrfuns, locEnrFuncs, Flag, posF);
                    MatrixXd gpsValues = echElemGpsValues[pos][0];
                    for (int m = 0; m < posF.size(); m++)
                    {
                        posF[m] -= 1;      // 减一变成从零开始
                    }
                    if (valuelist.rows() == 0) {
                        // 如果valuelist为空，直接使用gpsValues
                        valuelist = gpsValues(all, posF);
                    }
                    else {
                        // 如果valuelist不为空，使用conservativeResize + block赋值进行优化
                        int oldRows = valuelist.rows();
                        valuelist.conservativeResize(oldRows + gpsValues.rows(), posF.size());
                        valuelist.block(oldRows, 0, gpsValues.rows(), posF.size()) = gpsValues(all, posF);
                    }
                    MatrixXd gpsValues_x = echElemGpsValues[pos][1];
                    if (valuelist_x.rows() == 0) {
                        valuelist_x = gpsValues_x(all, posF);
                    }
                    else {
                        // 使用conservativeResize + block赋值进行优化
                        int oldRows_x = valuelist_x.rows();
                        valuelist_x.conservativeResize(oldRows_x + gpsValues_x.rows(), posF.size());
                        valuelist_x.block(oldRows_x, 0, gpsValues_x.rows(), posF.size()) = gpsValues_x(all, posF);
                    }

                    // 处理y坐标
                    MatrixXd gpsValues_y = echElemGpsValues[pos][2];
                    if (valuelist_y.rows() == 0) {
                        valuelist_y = gpsValues_y(all, posF);
                    }
                    else {
                        // 使用conservativeResize + block赋值进行优化
                        int oldRows_y = valuelist_y.rows();
                        valuelist_y.conservativeResize(oldRows_y + gpsValues_y.rows(), posF.size());
                        valuelist_y.block(oldRows_y, 0, gpsValues_y.rows(), posF.size()) = gpsValues_y(all, posF);
                    }

                    MatrixXd globalGps2D = ElemMap.localToGlobal(fid, localGps2D);
                    auto result1 = ismember_single(fid, IntElem);

                    auto append_to_gpslist = [](MatrixXd& gpslist, const MatrixXd& new_data)
                        {
                            if (gpslist.rows() == 0) {
                                gpslist = new_data;
                            }
                            else {
                                gpslist.conservativeResize(gpslist.rows() + new_data.rows(), 2);
                                gpslist.bottomRows(new_data.rows()) = new_data;
                            }
                        };

                    if (result1.first)
                    {
                        int PosInt = result1.second;
                        MatrixXd gps0 = intElemGps[PosInt][0].block(0, 0, intElemGps[PosInt][0].rows(), 4);
                        MatrixXd gps1 = intElemGps[PosInt][1].block(0, 0, intElemGps[PosInt][1].rows(), 4);
                        sizeGpsPointsList.push_back(gps0.rows());
                        sizeGpsPointsList.push_back(gps1.rows());
                        sizeGpsPoints = gps0.rows() + gps1.rows();
                        VectorXd gps0_weight = gps0.col(2).cwiseProduct(gps0.col(3));
                        VectorXd gps1_weight = gps1.col(2).cwiseProduct(gps1.col(3));
                        int original_size = weightlist.size();
                        weightlist.conservativeResize(original_size + gps0_weight.size() + gps1_weight.size());
                        weightlist.segment(original_size, gps0_weight.size()) = gps0_weight;
                        weightlist.segment(original_size + gps0_weight.size(), gps1_weight.size()) = gps1_weight;

                        MatrixXd gps0_selected = gps0.block(0, 0, gps0.rows(), 2);  // 取前两列
                        MatrixXd gps1_selected = gps1.block(0, 0, gps1.rows(), 2);


                        append_to_gpslist(gpslist, gps0_selected);
                        append_to_gpslist(gpslist, gps1_selected);

                    }
                    else
                    {
                        weightlist.conservativeResize(weightlist.size() + localGpW2D.size());
                        weightlist.tail(localGpW2D.size()) = localGpW2D * h * h; // 将v2拼接到v1尾部

                        append_to_gpslist(gpslist, globalGps2D);

                    }

                }
                MatrixXd cofL = MatrixXd::Zero(enrichMutiplyN, enrichMutiplyN);

                // 正交化过程 (Gram-Schmidt)
                for (int k = 1; k < enrichMutiplyN; ++k) {      // 从第2列开始 (C++索引1)
                    for (int kk = 0; kk < k; ++kk) {            // 前k列
                        // 计算加权点积: weightlist'*(vec_kk .* vec_k)
                        double numerator = (valuelist.col(kk).array() * valuelist.col(k).array() * weightlist.array()).sum();
                        double denominator = (valuelist.col(kk).array().square() * weightlist.array()).sum();

                        // 避免除零错误
                        if (std::abs(denominator) < 1e-15) {
                            cofL(k, kk) = 0.0;
                        }
                        else {
                            cofL(k, kk) = numerator / denominator;
                        }

                        // 正交化当前列
                        valuelist.col(k) -= valuelist.col(kk) * cofL(k, kk);
                        valuelist_x.col(k) -= valuelist_x.col(kk) * cofL(k, kk);
                        valuelist_y.col(k) -= valuelist_y.col(kk) * cofL(k, kk);
                    }
                }

                // 单位化过程 (归一化)
                for (int k = 0; k < enrichMutiplyN; ++k) {
                    // 计算当前列的加权范数平方
                    double norm_sq = (valuelist.col(k).array().square() * weightlist.array()).sum();

                    // 计算归一化系数
                    if (norm_sq < 1e-15) {
                        cofL(k, k) = 0.0;
                    }
                    else {
                        cofL(k, k) = 1.0 / std::sqrt(norm_sq);
                    }

                    // 应用归一化系数
                    valuelist.col(k) *= cofL(k, k);
                    valuelist_x.col(k) *= cofL(k, k);
                    valuelist_y.col(k) *= cofL(k, k);
                }

                OrthogonalMatrix[k][0] = cofL;
                int offset = 0;
                for (int fid : FaceIds) {
                    if (fid == 0) {
                        continue;
                    }
                    auto result = ismember_single(fid, echLocalDPUMap);
                    int pos = result.second;
                    vector<int> locEnrFuncs;
                    for (int val : echLocalDPUFunctions[pos]) {
                        locEnrFuncs.push_back(val);
                    }
                    vector<bool> Flag;
                    vector<int> posF;
                    ismember(enrfuns, locEnrFuncs, Flag, posF);
                    for (int m = 0; m < posF.size(); m++)
                    {
                        posF[m] -= 1;      // 减一变成从零开始
                    }
                    int tempsize = echElemGpsValues[pos][0].rows();  // 取values矩阵的行数

                    // 检查posF是否有效
                    if (!posF.empty()) {
                        // 更新values矩阵的对应列

                        echElemGpsValues[pos][0](all, posF) = valuelist.block(offset, 0, tempsize, valuelist.cols());
                        echElemGpsValues[pos][1](all, posF) = valuelist_x.block(offset, 0, tempsize, valuelist_x.cols());
                        echElemGpsValues[pos][2](all, posF) = valuelist_y.block(offset, 0, tempsize, valuelist_y.cols());
                        // 更新x-deriv矩阵的对应列
                    }

                    // 更新偏移量
                    offset += tempsize;
                }
            }

        }

        // 这里可以添加结果存储代码
        // result_SCN.push_back(some_value);


        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                
                int elemID = ElemMap.getElemId(i, j);
                int pos;
                bool enrich_flag = Ismember_single(elemID, echFaceNodeFreq, 0, pos);
                vector<int> locEnrFuncs;
                vector<int> locFEMFuncs = ElemMap.getFEMBasisIds(elemID);
                if (inter[i][j] != 1)
                {
                    MatrixXd globalGps = ElemMap.localToGlobal(elemID, localGps2D);
                    MatrixXd localGpValues2D_DPU_Minus_Interpolation;
                    MatrixXd localGpValues2D_DPU_Minus_Interpolation_x;
                    MatrixXd localGpValues2D_DPU_Minus_Interpolation_y;
                    if (EnrichMode != 0 && enrich_flag)
                    {
                        localGpValues2D_DPU_Minus_Interpolation = echElemGpsValues[pos][0];
                        localGpValues2D_DPU_Minus_Interpolation_x = echElemGpsValues[pos][1];
                        localGpValues2D_DPU_Minus_Interpolation_y = echElemGpsValues[pos][2];
                        locEnrFuncs = echLocalDPUFunctions[pos];
                    }
                    vector<int> locFuncs = locFEMFuncs;  // 先拷贝 locFEMFuncs
                    locFuncs.insert(locFuncs.end(), locEnrFuncs.begin(), locEnrFuncs.end());  // 追加 locEnrFuncs
                    int locFEMDim = locFEMFuncs.size();
                    int locEnrDim = locEnrFuncs.size();
                    int lockDim = locFEMDim + locEnrDim;
                    MatrixXd locK = MatrixXd::Zero(lockDim, lockDim);
                    VectorXd locKappa;
                    if (partion[i][j] == 0)
                    {
                        locKappa = kappa0;
                    }
                    else if (partion[i][j] == 1)
                    {
                        locKappa = kappa1;
                    }
                    else if (partion[i][j] = -1)
                    {
                        VectorXd res = calc_kappa(globalGps, intface);
                        VectorXd ones = VectorXd::Ones(res.size());
                        locKappa = a0 * res + a1 * (ones - res);
                    }
                    double jac = calcJacobiDetValues(elemID, N);
                    
                    // 预计算公共权重部分
                    VectorXd kappaWeighted_FEM = locKappa.cwiseProduct(localGpW2D) * N * N * jac;
                    VectorXd kappaWeighted_Mixed = locKappa.cwiseProduct(localGpW2D) * N * jac;
                    VectorXd kappaWeighted_Enr = locKappa.cwiseProduct(localGpW2D) * jac;
                    
                    // FEM-FEM block (对称矩阵)
                    for (int m = 0; m < locFEMDim; m++) {
                        // 计算对角线元素
                        locK(m, m) = (localGpValues2D_s.col(m).array().square()
                                    + localGpValues2D_t.col(m).array().square()).matrix().dot(kappaWeighted_FEM);
                        
                        // 计算非对角线元素 (利用对称性)
                        for (int n = m + 1; n < locFEMDim; n++) {
                            double value = (localGpValues2D_s.col(m).cwiseProduct(localGpValues2D_s.col(n))
                                          + localGpValues2D_t.col(m).cwiseProduct(localGpValues2D_t.col(n)))
                                          .dot(kappaWeighted_FEM);
                            locK(m, n) = value;
                            locK(n, m) = value;  // 对称填充
                        }
                    }

                    // Enriched-FEM block
                    for (int m = 0; m < locEnrDim; m++) {
                        for (int n = 0; n < locFEMDim; n++) {
                            locK(m + locFEMDim, n) = (localGpValues2D_DPU_Minus_Interpolation_x.col(m).cwiseProduct(localGpValues2D_s.col(n))
                                + localGpValues2D_DPU_Minus_Interpolation_y.col(m).cwiseProduct(localGpValues2D_t.col(n)))
                                .dot(kappaWeighted_Mixed);
                        }
                    }

                    // FEM-Enriched block (利用转置关系，直接复制Enriched-FEM块的转置)
                    for (int m = 0; m < locFEMDim; m++) {
                        for (int n = 0; n < locEnrDim; n++) {
                            locK(m, n + locFEMDim) = locK(n + locFEMDim, m);  // 转置填充
                        }
                    }

                    // Enriched-Enriched block (对称矩阵)
                    for (int m = 0; m < locEnrDim; m++) {
                        // 计算对角线元素
                        locK(m + locFEMDim, m + locFEMDim) = (localGpValues2D_DPU_Minus_Interpolation_x.col(m).array().square()
                                                           + localGpValues2D_DPU_Minus_Interpolation_y.col(m).array().square()).matrix().dot(kappaWeighted_Enr);
                        
                        // 计算非对角线元素 (利用对称性)
                        for (int n = m + 1; n < locEnrDim; n++) {
                            double value = (localGpValues2D_DPU_Minus_Interpolation_x.col(m).cwiseProduct(localGpValues2D_DPU_Minus_Interpolation_x.col(n))
                                          + localGpValues2D_DPU_Minus_Interpolation_y.col(m).cwiseProduct(localGpValues2D_DPU_Minus_Interpolation_y.col(n)))
                                          .dot(kappaWeighted_Enr);
                            locK(m + locFEMDim, n + locFEMDim) = value;
                            locK(n + locFEMDim, m + locFEMDim) = value;  // 对称填充
                        }
                    }

                    // 遍历 locFuncs 的行和列索引，将 locK 的值添加到三元组列表中
                    for (int i = 0; i < locFuncs.size(); ++i) {
                        for (int j = 0; j < locFuncs.size(); ++j) {
                            // 使用三元组批量插入，避免O(n^2)的coeffRef操作
                            if (locK(i, j) != 0.0) {
                                tripletList.push_back(T(locFuncs[i] - 1, locFuncs[j] - 1, locK(i, j)));
                            }
                        }
                    }

                    VectorXd val_f = compute_val_f(globalGps);
                    VectorXd weighted_f = val_f.array() * localGpW2D.array();

                    // 2. 垂直拼接矩阵 [localGpValues2D; localGpValues2D_DPU_Minus_Interpolation]
                    MatrixXd combined;
                    

                    // 如果第一个矩阵为空
                    if (localGpValues2D.size() == 0) {
                        combined = localGpValues2D_DPU_Minus_Interpolation;
                    }
                    // 如果第二个矩阵为空
                    else if (localGpValues2D_DPU_Minus_Interpolation.size() == 0) {
                        combined = localGpValues2D;
                    }
                    // 两个矩阵都非空
                    else {
                        combined.resize(localGpValues2D.rows(),
                            localGpValues2D.cols() + localGpValues2D_DPU_Minus_Interpolation.cols());
                        combined << localGpValues2D, localGpValues2D_DPU_Minus_Interpolation;
                    }

                    // 3. 计算转置乘法并乘以jac: combined' * weighted_f * jac
                    VectorXd locF = combined.transpose() * weighted_f * jac;

                    for (size_t i = 0; i < locFuncs.size(); ++i) {
                        int idx = locFuncs[i] - 1;  // 获取全局索引
                        F(idx) += locF(i);      // 累加到全局向量
                    }

                    VectorXd locCtra = combined.transpose() * localGpW2D * jac;

                    for (int i = 0; i < locFuncs.size(); ++i) {
                        if (locCtra(i) != 0.0) {
                            tripletList.push_back(T(totalFunctionN, locFuncs[i] - 1, locCtra(i)));
                        }
                    }


                }
                else
                {
                    auto result = ismember_single(elemID, IntElem);
                    bool flag_int = result.first;
                    int e = result.second;
                    MatrixXd Gps0 = intElemGps[e][0].leftCols(2);
                    MatrixXd GpW0 = intElemGps[e][0].col(2);
                    MatrixXd Jac0 = intElemGps[e][0].block(0, 3, intElemGps[e][0].rows(), 5);
                    MatrixXd Gps1 = intElemGps[e][1].leftCols(2);
                    MatrixXd GpW1 = intElemGps[e][1].col(2);
                    MatrixXd Jac1 = intElemGps[e][1].block(0, 3, intElemGps[e][1].rows(), 5);
                    double x0 = (j - 1) * h;
                    double y0 = (i - 1) * h;
                    MatrixXd shape_func0_DPU_Minus_Interpolation;
                    MatrixXd shape_func0_DPU_Minus_Interpolation_x;
                    MatrixXd shape_func0_DPU_Minus_Interpolation_y;
                    MatrixXd shape_func1_DPU_Minus_Interpolation;
                    MatrixXd shape_func1_DPU_Minus_Interpolation_x;
                    MatrixXd shape_func1_DPU_Minus_Interpolation_y;

                    int offset = Gps0.rows();
                    MatrixXd shape_func0 = intElemGpsVal[e][0].block(0, 0, offset, intElemGpsVal[e][0].cols());
                    MatrixXd shape_func1 = intElemGpsVal[e][0].block(offset, 0, intElemGpsVal[e][0].rows() - offset, intElemGpsVal[e][0].cols());
                    MatrixXd shape_func0_s = intElemGpsVal[e][1].block(0, 0, offset, intElemGpsVal[e][1].cols());
                    MatrixXd shape_func1_s = intElemGpsVal[e][1].block(offset, 0, intElemGpsVal[e][1].rows() - offset, intElemGpsVal[e][1].cols());
                    MatrixXd shape_func0_t = intElemGpsVal[e][2].block(0, 0, offset, intElemGpsVal[e][2].cols());
                    MatrixXd shape_func1_t = intElemGpsVal[e][2].block(offset, 0, intElemGpsVal[e][2].rows() - offset, intElemGpsVal[e][2].cols());


                    if (EnrichMode != 0 && enrich_flag)
                    {
                        locEnrFuncs = echLocalDPUFunctions[pos];

                        MatrixXd a = echElemGpsValues[pos][0];
                        shape_func0_DPU_Minus_Interpolation = echElemGpsValues[pos][0].block(0, 0, offset, echElemGpsValues[pos][0].cols());
                        shape_func0_DPU_Minus_Interpolation_x = echElemGpsValues[pos][1].block(0, 0, offset, echElemGpsValues[pos][1].cols());
                        shape_func0_DPU_Minus_Interpolation_y = echElemGpsValues[pos][2].block(0, 0, offset, echElemGpsValues[pos][2].cols());
                        shape_func1_DPU_Minus_Interpolation = echElemGpsValues[pos][0].block(offset, 0, echElemGpsValues[pos][0].rows() - offset, echElemGpsValues[pos][0].cols());
                        shape_func1_DPU_Minus_Interpolation_x = echElemGpsValues[pos][1].block(offset, 0, echElemGpsValues[pos][1].rows() - offset, echElemGpsValues[pos][1].cols());
                        shape_func1_DPU_Minus_Interpolation_y = echElemGpsValues[pos][2].block(offset, 0, echElemGpsValues[pos][2].rows() - offset, echElemGpsValues[pos][2].cols());


                    }

                    vector<int> locFuncs = locFEMFuncs;  // 先拷贝 locFEMFuncs
                    locFuncs.insert(locFuncs.end(), locEnrFuncs.begin(), locEnrFuncs.end());  // 追加 locEnrFuncs
                    int locFEMDim = locFEMFuncs.size();
                    int locEnrDim = locEnrFuncs.size();
                    int lockDim = locFEMDim + locEnrDim;
                    MatrixXd locK = MatrixXd::Zero(lockDim, lockDim);
                    VectorXd weighted_Jac0 = GpW0.array() * Jac0.col(0).array();
                    VectorXd weighted_Jac1 = GpW1.array() * Jac1.col(0).array();

                    // 预计算公共权重部分
                    VectorXd weighted_FEM0 = weighted_Jac0 * a0 * N * N;
                    VectorXd weighted_FEM1 = weighted_Jac1 * a1 * N * N;
                    VectorXd weighted_Mixed0 = weighted_Jac0 * a0 * N;
                    VectorXd weighted_Mixed1 = weighted_Jac1 * a1 * N;
                    VectorXd weighted_Enr0 = weighted_Jac0 * a0;
                    VectorXd weighted_Enr1 = weighted_Jac1 * a1;

                    // 第一部分: FEM-FEM 块 (m=1:locFEMDim, n=1:locFEMDim) - 对称矩阵
                    for (int m = 0; m < locFEMDim; ++m) {

                        // 计算对角线元素
                        double diag_term0 = (shape_func0_s.col(m).array().square()
                                           + shape_func0_t.col(m).array().square()).matrix().dot(weighted_FEM0);
                        double diag_term1 = (shape_func1_s.col(m).array().square()
                                           + shape_func1_t.col(m).array().square()).matrix().dot(weighted_FEM1);
                        locK(m, m) = diag_term0 + diag_term1;
                        
                        // 计算非对角线元素 (利用对称性)
                        for (int n = m + 1; n < locFEMDim; ++n) {
                            double term0 = (shape_func0_s.col(m).array() * shape_func0_s.col(n).array()
                                          + shape_func0_t.col(m).array() * shape_func0_t.col(n).array()).matrix().dot(weighted_FEM0);
                            double term1 = (shape_func1_s.col(m).array() * shape_func1_s.col(n).array()
                                          + shape_func1_t.col(m).array() * shape_func1_t.col(n).array()).matrix().dot(weighted_FEM1);
                            double value = term0 + term1;
                            locK(m, n) = value;
                            locK(n, m) = value;  // 对称填充
                        }
                    }

                    // 第二部分: Enr-FEM 块 (m=1:locEnrDim, n=1:locFEMDim)
                    for (int m = 0; m < locEnrDim; ++m) {
                        for (int n = 0; n < locFEMDim; ++n) {
                            double term0 = (shape_func0_DPU_Minus_Interpolation_x.col(m).array() * shape_func0_s.col(n).array()
                                          + shape_func0_DPU_Minus_Interpolation_y.col(m).array() * shape_func0_t.col(n).array()).matrix().dot(weighted_Mixed0);
                            double term1 = (shape_func1_DPU_Minus_Interpolation_x.col(m).array() * shape_func1_s.col(n).array()
                                          + shape_func1_DPU_Minus_Interpolation_y.col(m).array() * shape_func1_t.col(n).array()).matrix().dot(weighted_Mixed1);
                            locK(m + locFEMDim, n) = term0 + term1;
                        }
                    }

                    // 第三部分: FEM-Enr 块 (m=1:locFEMDim, n=1:locEnrDim) - 利用转置关系
                    for (int m = 0; m < locFEMDim; ++m) {
                        for (int n = 0; n < locEnrDim; ++n) {
                            locK(m, n + locFEMDim) = locK(n + locFEMDim, m);  // 转置填充
                        }
                    }

                    // 第四部分: Enr-Enr 块 (m=1:locEnrDim, n=1:locEnrDim) - 对称矩阵
                    for (int m = 0; m < locEnrDim; ++m) {
                        // 计算对角线元素
                        double diag_term0 = (shape_func0_DPU_Minus_Interpolation_x.col(m).array().square()
                                           + shape_func0_DPU_Minus_Interpolation_y.col(m).array().square()).matrix().dot(weighted_Enr0);
                        double diag_term1 = (shape_func1_DPU_Minus_Interpolation_x.col(m).array().square()
                                           + shape_func1_DPU_Minus_Interpolation_y.col(m).array().square()).matrix().dot(weighted_Enr1);
                        locK(m + locFEMDim, m + locFEMDim) = diag_term0 + diag_term1;
                        
                        // 计算非对角线元素 (利用对称性)
                        for (int n = m + 1; n < locEnrDim; ++n) {
                            double term0 = (shape_func0_DPU_Minus_Interpolation_x.col(m).array() * shape_func0_DPU_Minus_Interpolation_x.col(n).array()
                                          + shape_func0_DPU_Minus_Interpolation_y.col(m).array() * shape_func0_DPU_Minus_Interpolation_y.col(n).array()).matrix().dot(weighted_Enr0);
                            double term1 = (shape_func1_DPU_Minus_Interpolation_x.col(m).array() * shape_func1_DPU_Minus_Interpolation_x.col(n).array()
                                          + shape_func1_DPU_Minus_Interpolation_y.col(m).array() * shape_func1_DPU_Minus_Interpolation_y.col(n).array()).matrix().dot(weighted_Enr1);
                            double value = term0 + term1;
                            locK(m + locFEMDim, n + locFEMDim) = value;
                            locK(n + locFEMDim, m + locFEMDim) = value;  // 对称填充
                        }
                    }

                    for (int i = 0; i < locFuncs.size(); ++i) {
                        for (int j = 0; j < locFuncs.size(); ++j) {
                            // 使用三元组批量插入，避免O(n^2)的coeffRef操作
                            if (locK(i, j) != 0.0) {
                                tripletList.push_back(T(locFuncs[i] - 1, locFuncs[j] - 1, locK(i, j)));
                            }
                        }
                    }

                    VectorXd ma(Gps0.col(0).size() + Gps1.col(0).size());
                    VectorXd ma1(Gps0.col(0).size() + Gps1.col(0).size());
                    ma << Gps0.col(0), Gps1.col(0);
                    ma1 << Gps0.col(1), Gps1.col(1);
                    MatrixXd Ma(ma.size(), 2);
                    Ma.col(0) = ma;
                    Ma.col(1) = ma1;
                    VectorXd val_f = compute_val_f(Ma);
                    Eigen::MatrixXd A(shape_func0.rows() + shape_func1.rows(), shape_func0.cols() + shape_func0_DPU_Minus_Interpolation.cols());
                    A << shape_func0, shape_func0_DPU_Minus_Interpolation,
                        shape_func1, shape_func1_DPU_Minus_Interpolation;
                    VectorXd GpW_combined(GpW0.rows() + GpW1.rows());
                    GpW_combined << GpW0, GpW1;
                    VectorXd Jac_combined(Jac0.rows() + Jac1.rows());
                    Jac_combined << Jac0.col(0), Jac1.col(0);
                    VectorXd product = GpW_combined.array() * Jac_combined.array();
                    product = val_f.array() * product.array(); // 标量乘法
                    VectorXd locF = A.transpose() * product;
                    for (size_t i = 0; i < locFuncs.size(); ++i) {
                        int idx = locFuncs[i] - 1;  // 获取全局索引
                        F(idx) += locF(i);      // 累加到全局向量
                    }
                    product = GpW_combined.array() * Jac_combined.array();
                    VectorXd locCtra = A.transpose() * product;
                    for (int i = 0; i < locFuncs.size(); ++i) {
                        if (locCtra(i) != 0.0) {
                            tripletList.push_back(T(totalFunctionN, locFuncs[i] - 1, locCtra(i)));
                        }
                    }

                }
            }
        }

        double gJac = h;
        VectorXd bdrykappa0 = VectorXd::Ones(gpn) * a0;
        VectorXd bdrykappa1 = VectorXd::Ones(gpn) * a1;
        VectorXd zero1 = VectorXd::Zero(1);
        VectorXd one1 = VectorXd::Ones(1);

        // 2. 初始化零矩阵 (order_p+1)^2 列
        int cols_basis = (order_p + 1) * (order_p + 1);
        MatrixXd locBdryGpsValue2D_left = MatrixXd::Zero(gpn, cols_basis);
        MatrixXd locBdryGpsValue2D_bottom = MatrixXd::Zero(gpn, cols_basis);
        MatrixXd locBdryGpsValue2D_right = MatrixXd::Zero(gpn, cols_basis);
        MatrixXd locBdryGpsValue2D_top = MatrixXd::Zero(gpn, cols_basis);

        // 3. 初始化零矩阵 PUDim^2 列
        int cols_PU = PUDim * PUDim;
        MatrixXd locBdryGpsValue2D_PU_left = MatrixXd::Zero(gpn, cols_PU);
        MatrixXd locBdryGpsValue2D_PU_bottom = MatrixXd::Zero(gpn, cols_PU);
        MatrixXd locBdryGpsValue2D_PU_right = MatrixXd::Zero(gpn, cols_PU);
        MatrixXd locBdryGpsValue2D_PU_top = MatrixXd::Zero(gpn, cols_PU);



        MatrixXd shapeFunc0 = BasisFunction::FEMShapeFunction(zero1, order_p, 0);
        MatrixXd shapeFunc1 = BasisFunction::FEMShapeFunction(one1, order_p, 0);
        for (int i = 0; i < order_p + 1; ++i) {
            for (int j = 0; j < order_p + 1; ++j) {
                // 计算临时向量
                VectorXd temp1 = shapeFunc0(j) * localGpValues1D.col(i);
                VectorXd temp2 = shapeFunc0(i) * localGpValues1D.col(j);
                VectorXd temp3 = shapeFunc1(j) * localGpValues1D.col(i);
                VectorXd temp4 = shapeFunc1(i) * localGpValues1D.col(j);

                // 填充到结果矩阵
                locBdryGpsValue2D_left.col(j * (order_p + 1) + i) = temp1;
                locBdryGpsValue2D_bottom.col(j * (order_p + 1) + i) = temp2;
                locBdryGpsValue2D_right.col(j * (order_p + 1) + i) = temp3;
                locBdryGpsValue2D_top.col(j * (order_p + 1) + i) = temp4;
            }
        }

        // 第二部分：处理 PU 维度循环
        Vector2d shapeFunc0_PU(1.0, 0.0);  // [1, 0]
        Vector2d shapeFunc1_PU(0.0, 1.0);  // [0, 1]

        for (int i = 0; i < PUDim; ++i) {
            for (int j = 0; j < PUDim; ++j) {
                // 计算临时向量
                VectorXd temp1 = shapeFunc0_PU(j) * localGpValues1D_PU.col(i);
                VectorXd temp2 = shapeFunc0_PU(i) * localGpValues1D_PU.col(j);
                VectorXd temp3 = shapeFunc1_PU(j) * localGpValues1D_PU.col(i);
                VectorXd temp4 = shapeFunc1_PU(i) * localGpValues1D_PU.col(j);

                // 填充到结果矩阵
                locBdryGpsValue2D_PU_left.col(j * PUDim + i) = temp1;
                locBdryGpsValue2D_PU_bottom.col(j * PUDim + i) = temp2;
                locBdryGpsValue2D_PU_right.col(j * PUDim + i) = temp3;
                locBdryGpsValue2D_PU_top.col(j * PUDim + i) = temp4;
            }
        }

        for (int i = 0;i < N; ++i)
        {
            
            int elemID = ElemMap.getElemId(i, 0);
            vector<int> locEnrFuncs;
            vector<int> locFEMFuncs = ElemMap.getFEMBasisIds(elemID);
            MatrixXd globalInterpolationPoints2D = ElemMap.localToGlobal(elemID, InterpolationPoints2D);
            Vector2d normal(-1.0, 0.0);
            VectorXd zeros_col = VectorXd::Zero(gpn);
            // 水平拼接 [zeros(gpn,1), localGps]
            MatrixXd localBdryGps2D(gpn, 1 + localGps.cols());
            localBdryGps2D << zeros_col, localGps;

            MatrixXd globalBdryGps = ElemMap.localToGlobal(elemID, localBdryGps2D);
            bool intBoundaryflag = false;
            MatrixXd locIntBdryGpsValue2D0_left;
            MatrixXd locIntBdryGpsValue2D1_left;
            MatrixXd locIntBdryGpsValue2D0_PU_left;
            MatrixXd locIntBdryGpsValue2D1_PU_left;
            MatrixXd bdryGps0_loc;
            MatrixXd bdryGps1_loc;
            MatrixXd bdryGps0;
            VectorXd bdryGpW0;
            MatrixXd bdryGps1;
            VectorXd bdryGpW1;
            VectorXd locBdryKappa;
            VectorXd bdryGps0_y;
            VectorXd bdryGps1_y;


            if (partion[i][0] == 0)
            {
                locBdryKappa = bdrykappa0;
            }
            else if (partion[i][0] == 1)
            {
                locBdryKappa = bdrykappa1;
            }
            else if (partion[i][0] == -1)
            {
                VectorXd res = calc_kappa(globalBdryGps, intface);
                VectorXd ones = VectorXd::Ones(res.size());
                locBdryKappa = a0 * res + a1 * (ones - res);
                double sumKappa = locBdryKappa.sum();
                int sizeKappa = locBdryKappa.size();

                if (!(abs(sumKappa - a0 * sizeKappa) < 1e-10 ||
                    abs(sumKappa - a1 * sizeKappa) < 1e-10))
                {
                    intBoundaryflag = true;
                    auto result = calc_intbdry_gps(intfacex, intfacey, i + 1, 1, N, gpn, 0);
                    bdryGps0 = get<0>(result);
                    bdryGpW0 = get<1>(result);
                    bdryGps1 = get<2>(result);
                    bdryGpW1 = get<3>(result);
                    bdryGps0_loc = (bdryGps0.rowwise() - RowVector2d(0, (i)*h)) * N;
                    bdryGps1_loc = (bdryGps1.rowwise() - RowVector2d(0, (i)*h)) * N;

                    locIntBdryGpsValue2D0_left = MatrixXd::Zero(bdryGps0_loc.rows(), (order_p + 1) * (order_p + 1));
                    locIntBdryGpsValue2D1_left = MatrixXd::Zero(bdryGps1_loc.rows(), (order_p + 1) * (order_p + 1));


                    // 提取y坐标（第二列）
                    bdryGps0_y = bdryGps0_loc.col(1);
                    bdryGps1_y = bdryGps1_loc.col(1);

                    // 计算形状函数
                    MatrixXd intShapeFunction0 = BasisFunction::FEMShapeFunction(bdryGps0_y.transpose(), order_p, 0);
                    MatrixXd intShapeFunction1 = BasisFunction::FEMShapeFunction(bdryGps1_y.transpose(), order_p, 0);

                    // 双重循环计算
                    for (int m = 0; m < order_p + 1; ++m) {
                        for (int n = 0; n < order_p + 1; ++n) {
                            // 计算temp0和temp1
                            VectorXd temp0 = shapeFunc0(n) * intShapeFunction0.col(m);
                            VectorXd temp1 = shapeFunc0(n) * intShapeFunction1.col(m);

                            // 填充结果矩阵
                            locIntBdryGpsValue2D0_left.col(n * (order_p + 1) + m) = temp0;
                            locIntBdryGpsValue2D1_left.col(n * (order_p + 1) + m) = temp1;
                        }
                    }

                    if (EnrichMode != 0) {
                        // 初始化结果矩阵
                        locIntBdryGpsValue2D0_PU_left = MatrixXd::Zero(bdryGps0_loc.rows(), PUDim * PUDim);
                        locIntBdryGpsValue2D1_PU_left = MatrixXd::Zero(bdryGps1_loc.rows(), PUDim * PUDim);

                        // 提取 y 坐标（第二列）
                        bdryGps0_y = bdryGps0_loc.col(1);
                        bdryGps1_y = bdryGps1_loc.col(1);

                        // 计算 PU 形状函数
                        MatrixXd intPUShapeFunction0 = BasisFunction::PUShapeFunction(bdryGps0_y, 0);
                        MatrixXd intPUShapeFunction1 = BasisFunction::PUShapeFunction(bdryGps1_y, 0);

                        // 双重循环计算
                        for (int m = 0; m < PUDim; ++m) {
                            for (int n = 0; n < PUDim; ++n) {
                                // 计算 temp0 和 temp1
                                VectorXd temp0 = shapeFunc0_PU(n) * intPUShapeFunction0.col(m);
                                VectorXd temp1 = shapeFunc0_PU(n) * intPUShapeFunction1.col(m);

                                // 填充结果矩阵
                                locIntBdryGpsValue2D0_PU_left.col(n * PUDim + m) = temp0;
                                locIntBdryGpsValue2D1_PU_left.col(n * PUDim + m) = temp1;
                            }
                        }


                    }

                }


            }

            MatrixXd locBdryGpsValue2D_DPU_Minus_Interpolation_left;
            MatrixXd locIntBdryGpsValue2D0_DPU_Minus_Interpolation_left;
            MatrixXd locIntBdryGpsValue2D1_DPU_Minus_Interpolation_left;

            bool enrich_flag = false;
            vector<int> FaceIds;

            if (EnrichMode != 0)
            {
                int pos;
                enrich_flag = Ismember_single(elemID, echFaceNodeFreq, 0, pos);
                if (enrich_flag) {
                    vector<int> vertexNodeIds = { echFaceNodeFreq[pos][5],echFaceNodeFreq[pos][6],echFaceNodeFreq[pos][7],echFaceNodeFreq[pos][8] };
                    vector<size_t> nzIds = Find(vertexNodeIds);     //从零开始 无须加减一
                    vector<vector<int>> faceIds = ElemMap.getOneRingFaces(elemID);
                    FaceIds.clear();
                    for (size_t col = 0; col < faceIds[0].size(); ++col) {  // 先遍历列
                        for (size_t row = 0; row < faceIds.size(); ++row) {  // 再遍历行
                            if (col < faceIds[row].size()) {  // 防止列数不一致
                                FaceIds.push_back(faceIds[row][col]);
                            }
                        }
                    }
                    vector<int> locPUFuncs;
                    vector<bool> Flag;
                    vector<int> EchFuncId;
                    ismember(FaceIds, IntElem, Flag, EchFuncId);
                    vector<size_t> fnzIds = Find(EchFuncId);
                    for (size_t k = 0; k < fnzIds.size(); ++k) {
                        echLocalPUFunctions[pos].push_back(EchFuncId[fnzIds[k]]);
                        locPUFuncs.push_back(EchFuncId[fnzIds[k]] - 1);
                    }
                    vector<int> node_vector;
                    vector<int> echVerNodeFreq_col;
                    for (int col_idx : nzIds) {
                        if (col_idx >= 0 && col_idx < echFaceNodeFreq[pos].size()) {
                            node_vector.push_back(echFaceNodeFreq[pos][col_idx + 1]);
                        }
                    }
                    for (const auto& pair : echVerNodeFreq) {
                        echVerNodeFreq_col.push_back(pair.first);
                    }
                    vector<bool> Flag2;
                    vector<int> EchFuncId2;
                    ismember(node_vector, echVerNodeFreq_col, Flag2, EchFuncId2);  //需要减一
                    vector<double> vFreq;
                    for (int col_idx : EchFuncId2) {
                        double num = echVerNodeFreq[col_idx - 1].second;
                        vFreq.push_back(1 / num);
                    }

                    // 创建vFreq向量
                    Eigen::VectorXd vFreqVec(vFreq.size());

                    // 使用 for 循环逐元素赋值
                    for (size_t i = 0; i < vFreq.size(); ++i) {
                        vFreqVec(i) = vFreq[i];  // 注意：Eigen 使用 `(i)` 而不是 `[i]` 访问元素
                    }


                    // 提取PU_Group的子矩阵（fnzIds列）
                    Eigen::MatrixXd PU_Group_selected = PU_Group(nzIds, fnzIds);

                    // 计算 vFreq .* PU_Group(nzIds, fnzIds)
                    // 由于 vFreq 和 PU_Group_selected 的列数可能不同，需要确保维度匹配
                    // 假设 vFreq.size() == PU_Group_selected.cols()（即 nzIds 和 fnzIds 长度相同）
                    Eigen::MatrixXd term1(PU_Group_selected.rows(), PU_Group_selected.cols());
                    for (int j = 0; j < PU_Group_selected.rows(); ++j) {
                        term1.row(j) = vFreqVec(j) * PU_Group_selected.row(j);
                    }

                    // 计算最终结果
                    int N1 = enrichMutiplyN;
                    int M1 = locPUFuncs.size();      // 列向量长度
                    int K1 = fnzIds.size();          // 列向量长度

                    // 1. locEnrFuncs 矩阵： N1 × M1（列优先）
                    std::vector<int> locEnrMat(N1 * M1);
                    for (int r = 0; r < N1; ++r)           // 行
                        for (int c = 0; c < M1; ++c)       // 列
                            locEnrMat[r + c * N1] = locPUFuncs[c] * N1;

                    // 2. offset 矩阵： K1 × N1（列优先）
                    std::vector<int> offsetMat(K1 * N1);
                    for (int r = 0; r < N1; ++r)
                        for (int c = 0; c < K1; ++c)
                            offsetMat[c * N1 + r] = r + 1;


                    // 3. 逐元素相加（MATLAB 会广播到同长度）

                    locEnrFuncs.reserve(std::max(N1 * M1, K1 * N1));
                    for (size_t i = 0; i < std::min(N1 * M1, K1 * N1); ++i)
                        locEnrFuncs.push_back(locEnrMat[i] + offsetMat[i]);

                    // 4. 加全局偏移
                    for (int& v : locEnrFuncs) v += globalFEMFunctionN;
                    locBdryGpsValue2D_DPU_Minus_Interpolation_left = MatrixXd::Zero(globalBdryGps.rows(), locEnrFuncs.size());
                    locIntBdryGpsValue2D0_DPU_Minus_Interpolation_left = MatrixXd::Zero(bdryGps0_loc.rows(), locEnrFuncs.size());
                    locIntBdryGpsValue2D1_DPU_Minus_Interpolation_left = MatrixXd::Zero(bdryGps1_loc.rows(), locEnrFuncs.size());

                    if (intBoundaryflag)
                    {
                        MatrixXd localPUvalues0_left, localPUvalues1_left;
                        localPUvalues0_left = locIntBdryGpsValue2D0_PU_left(Eigen::all, nzIds) * term1;
                        localPUvalues1_left = locIntBdryGpsValue2D1_PU_left(Eigen::all, nzIds) * term1;



                        for (int k = 0; k < fnzIds.size(); k++)
                        {
                            int fid = fnzIds[k];
                            double x0, y0 = 0;
                            ElemMap.LocalOriginCoordPU(elemID, fid + 1, x0, y0);
                            MatrixXd DMutis0;
                            MatrixXd DMutis0_x;
                            MatrixXd DMutis0_y;
                            MatrixXd DMutis0_Cof;
                            MatrixXd DMutis1;
                            MatrixXd DMutis1_x;
                            MatrixXd DMutis1_y;
                            MatrixXd DMutis1_Cof;
                            if (!Flag_oldxy) {
                                GenerateOffsetDMutis(bdryGps0, globalInterpolationPoints2D, order_p,
                                    x0, y0, 3 * h, 3 * h,
                                    DMutis0, DMutis0_x, DMutis1_y, DMutis0_Cof);
                                GenerateOffsetDMutis(bdryGps1, globalInterpolationPoints2D, order_p,
                                    x0, y0, 3 * h, 3 * h,
                                    DMutis1, DMutis0_x, DMutis1_y, DMutis1_Cof);
                            }
                            else {
                                GenerateOffsetDMutis(bdryGps0, globalInterpolationPoints2D, order_p,
                                    0, 0, 1, 1,
                                    DMutis0, DMutis0_x, DMutis1_y, DMutis0_Cof);
                                GenerateOffsetDMutis(bdryGps1, globalInterpolationPoints2D, order_p,
                                    0, 0, 1, 1,
                                    DMutis1, DMutis0_x, DMutis1_y, DMutis1_Cof);
                            }
                            DMutis0_Interpolation = locIntBdryGpsValue2D0_left * DMutis0_Cof;
                            DMutis0_Minus_Interpolation = DMutis0 - DMutis0_Interpolation;
                            DMutis1_Interpolation = locIntBdryGpsValue2D1_left * DMutis1_Cof;
                            DMutis1_Minus_Interpolation = DMutis1 - DMutis1_Interpolation;
                            int start_col = (k)*enrichMutiplyN;
                            scalar_vector0 = localPUvalues0_left.col(k);

                            scaled_matrix_0 = DMutis0_Minus_Interpolation.cwiseProduct(scalar_vector0.replicate(1, DMutis0_Minus_Interpolation.cols()));
                            scalar_vector1 = localPUvalues1_left.col(k);
                            scaled_matrix_1 = DMutis1_Minus_Interpolation.cwiseProduct(scalar_vector1.replicate(1, DMutis1_Minus_Interpolation.cols()));
                            locIntBdryGpsValue2D0_DPU_Minus_Interpolation_left.block(0, start_col,
                                locIntBdryGpsValue2D0_DPU_Minus_Interpolation_left.rows(), enrichMutiplyN) = scaled_matrix_0;
                            locIntBdryGpsValue2D1_DPU_Minus_Interpolation_left.block(0, start_col,
                                locIntBdryGpsValue2D1_DPU_Minus_Interpolation_left.rows(), enrichMutiplyN) = scaled_matrix_1;

                        }


                    }
                    else {
                        MatrixXd localPUvalues_left;
                        localPUvalues_left = locBdryGpsValue2D_PU_left(Eigen::all, nzIds) * term1;
                        locBdryGpsValue2D_DPU_Minus_Interpolation_left = MatrixXd::Zero(globalBdryGps.rows(), locEnrFuncs.size());

                        for (int k = 0; k < fnzIds.size(); k++)
                        {
                            int fid = fnzIds[k];
                            double x0, y0 = 0;
                            ElemMap.LocalOriginCoordPU(elemID, fid + 1, x0, y0);
                            MatrixXd DMutis;
                            MatrixXd DMutis_x;
                            MatrixXd DMutis_y;
                            MatrixXd DMutis_Cof;
                            if (!Flag_oldxy) {
                                GenerateOffsetDMutis(globalBdryGps, globalInterpolationPoints2D, order_p,
                                    x0, y0, 3 * h, 3 * h,
                                    DMutis, DMutis_x, DMutis_y, DMutis_Cof);
                            }
                            else {
                                GenerateOffsetDMutis(globalBdryGps, globalInterpolationPoints2D, order_p,
                                    0, 0, 1, 1,
                                    DMutis, DMutis_x, DMutis_y, DMutis_Cof);
                            }
                            DMutis_Interpolation = locBdryGpsValue2D_left * DMutis_Cof;
                            DMutis_Minus_Interpolation = DMutis - DMutis_Interpolation;
                            int start_col = (k)*enrichMutiplyN;
                            scalar_vector = localPUvalues_left.col(k);
                            scaled_matrix = DMutis_Minus_Interpolation.cwiseProduct(scalar_vector.replicate(1, DMutis_Minus_Interpolation.cols()));

                            locBdryGpsValue2D_DPU_Minus_Interpolation_left.block(0, start_col,
                                locBdryGpsValue2D_DPU_Minus_Interpolation_left.rows(), enrichMutiplyN) = scaled_matrix;

                        }
                    }

                }

            }

            if (EnrichMode == 1 && enrich_flag && OrthogonalMode)
            {
                for (int fid : FaceIds)
                {
                    auto result = ismember_single(fid, IntElem);
                    bool flagInt = result.first;
                    int posInt = result.second;

                    if (flagInt)
                    {
                        vector<int> enrfuns(enrichMutiplyN);

                        for (int i = 0; i < enrichMutiplyN; ++i)
                        {
                            enrfuns[i] = globalFEMFunctionN + (posInt)*enrichMutiplyN + (i + 1);
                        }

                        vector<bool> Flag;
                        vector<int> posF;
                        ismember(enrfuns, locEnrFuncs, Flag, posF);
                        for (int m = 0; m < posF.size(); m++)
                        {
                            posF[m] -= 1;      // 减一变成从零开始
                        }
                        MatrixXd cofL = OrthogonalMatrix[posInt][0];
                        if (!intBoundaryflag)
                        {
                            MatrixXd tempvalues = locBdryGpsValue2D_DPU_Minus_Interpolation_left(Eigen::all, posF);
                            for (int k = 1; k < enrichMutiplyN; ++k) {  // k从2开始变为1开始
                                for (int kk = 0; kk < k; ++kk) {       // kk从1到k-1变为0到k-1
                                    tempvalues.col(k) -= cofL(k, kk) * tempvalues.col(kk);
                                }
                            }
                            for (int k = 0; k < enrichMutiplyN; ++k) {  // k从1开始变为0开始
                                tempvalues.col(k) *= cofL(k, k);
                            }
                            locBdryGpsValue2D_DPU_Minus_Interpolation_left(Eigen::all, posF) = tempvalues;
                        }
                        else
                        {
                            MatrixXd tempvalues0 = locIntBdryGpsValue2D0_DPU_Minus_Interpolation_left(Eigen::all, posF);
                            MatrixXd tempvalues1 = locIntBdryGpsValue2D1_DPU_Minus_Interpolation_left(Eigen::all, posF);
                            for (int k = 1; k < enrichMutiplyN; ++k) {  // k从2开始变为1开始
                                for (int kk = 0; kk < k; ++kk) {       // kk从1到k-1变为0到k-1
                                    tempvalues0.col(k) -= cofL(k, kk) * tempvalues0.col(kk);
                                    tempvalues1.col(k) -= cofL(k, kk) * tempvalues1.col(kk);
                                }
                            }
                            for (int k = 0; k < enrichMutiplyN; ++k) {  // k从1开始变为0开始
                                tempvalues0.col(k) *= cofL(k, k);
                                tempvalues1.col(k) *= cofL(k, k);
                            }
                            locIntBdryGpsValue2D0_DPU_Minus_Interpolation_left(Eigen::all, posF) = tempvalues0;
                            locIntBdryGpsValue2D1_DPU_Minus_Interpolation_left(Eigen::all, posF) = tempvalues1;

                        }
                    }

                }
            }
            VectorXd locG;  // 结果向量

            if (!intBoundaryflag) {
                // 非分片边界情况
                int n = globalBdryGps.rows();
                VectorXd val_ux(n), val_uy(n);

                // 逐点计算标量函数
                for (int i = 0; i < n; ++i) {
                    val_ux(i) = calc_exact_ux(globalBdryGps(i, 0), globalBdryGps(i, 1));
                    val_uy(i) = calc_exact_uy(globalBdryGps(i, 0), globalBdryGps(i, 1));
                }

                // 计算 g = [val_ux val_uy] * normal
                VectorXd g = val_ux * normal(0) + val_uy * normal(1);

                // 计算加权项并组装矩阵
                VectorXd weighted_g = g.array() * locBdryKappa.array() * localGpWs.array();
                
                // 方案二：直接计算，完全避免中间矩阵
                if (locBdryGpsValue2D_DPU_Minus_Interpolation_left.cols() == 0) {
                    // 无富集情况，直接计算
                    locG = locBdryGpsValue2D_left.transpose() * weighted_g * gJac;
                } else {
                    // 有富集情况，分段计算填入locG
                    int n1 = locBdryGpsValue2D_left.cols();
                    int n2 = locBdryGpsValue2D_DPU_Minus_Interpolation_left.cols();
                    
                    // 先分配locG的空间（如果还未分配）
                    if (locG.size() != n1 + n2) {
                        locG.resize(n1 + n2);
                    }
                    
                    // 分段计算并填入locG
                    locG.segment(0, n1) = locBdryGpsValue2D_left.transpose() * weighted_g * gJac;
                    locG.segment(n1, n2) = locBdryGpsValue2D_DPU_Minus_Interpolation_left.transpose() * weighted_g * gJac;
                }
            }
            else {
                // 分片边界情况
                int n0 = bdryGps0.rows();
                int n1 = bdryGps1.rows();

                // 计算第一组边界点
                VectorXd val_ux0(n0), val_uy0(n0);
                for (int i = 0; i < n0; ++i) {
                    val_ux0(i) = calc_exact_ux(bdryGps0(i, 0), bdryGps0(i, 1));
                    val_uy0(i) = calc_exact_uy(bdryGps0(i, 0), bdryGps0(i, 1));
                }
                VectorXd g0 = val_ux0 * normal(0) + val_uy0 * normal(1);

                // 计算第二组边界点
                VectorXd val_ux1(n1), val_uy1(n1);
                for (int i = 0; i < n1; ++i) {
                    val_ux1(i) = calc_exact_ux(bdryGps1(i, 0), bdryGps1(i, 1));
                    val_uy1(i) = calc_exact_uy(bdryGps1(i, 0), bdryGps1(i, 1));
                }
                VectorXd g1 = val_ux1 * normal(0) + val_uy1 * normal(1);

                // 方案二：直接计算，避免中间矩阵combined_matrix0和combined_matrix1
                
                // 计算第一组的贡献
                VectorXd part0;
                if (locIntBdryGpsValue2D0_DPU_Minus_Interpolation_left.cols() == 0) {
                    // 无富集
                    part0 = locIntBdryGpsValue2D0_left.transpose() * (g0.cwiseProduct(bdryGpW0));
                } else {
                    // 有富集，分段计算
                    int n0_1 = locIntBdryGpsValue2D0_left.cols();
                    int n0_2 = locIntBdryGpsValue2D0_DPU_Minus_Interpolation_left.cols();
                    part0.resize(n0_1 + n0_2);
                    part0.segment(0, n0_1) = locIntBdryGpsValue2D0_left.transpose() * (g0.cwiseProduct(bdryGpW0));
                    part0.segment(n0_1, n0_2) = locIntBdryGpsValue2D0_DPU_Minus_Interpolation_left.transpose() * (g0.cwiseProduct(bdryGpW0));
                }
                
                // 计算第二组的贡献
                VectorXd part1;
                if (locIntBdryGpsValue2D1_DPU_Minus_Interpolation_left.cols() == 0) {
                    // 无富集
                    part1 = locIntBdryGpsValue2D1_left.transpose() * (g1.cwiseProduct(bdryGpW1));
                } else {
                    // 有富集，分段计算
                    int n1_1 = locIntBdryGpsValue2D1_left.cols();
                    int n1_2 = locIntBdryGpsValue2D1_DPU_Minus_Interpolation_left.cols();
                    part1.resize(n1_1 + n1_2);
                    part1.segment(0, n1_1) = locIntBdryGpsValue2D1_left.transpose() * (g1.cwiseProduct(bdryGpW1));
                    part1.segment(n1_1, n1_2) = locIntBdryGpsValue2D1_DPU_Minus_Interpolation_left.transpose() * (g1.cwiseProduct(bdryGpW1));
                }
                
                // 最终结果
                if (locG.size() != part0.size()) {
                    locG.resize(part0.size());
                }
                locG = a0 * part0 + a1 * part1;
            }

            vector<int> locFuncs;
            locFuncs.reserve(locFEMFuncs.size() + locEnrFuncs.size());
            locFuncs.insert(locFuncs.end(), locFEMFuncs.begin(), locFEMFuncs.end());
            locFuncs.insert(locFuncs.end(), locEnrFuncs.begin(), locEnrFuncs.end());

            // 2. 检查维度匹配
            assert(locFuncs.size() == locG.size() && "locG size must match locFuncs size");

            // 3. 执行累加操作 (转换为C++的0-based索引)
            for (size_t i = 0; i < locFuncs.size(); ++i) {
                int idx = locFuncs[i] - 1;  // MATLAB 1-based 转 C++ 0-based
                assert(idx >= 0 && idx < F.size() && "Index out of bounds");
                F(idx) += locG(i);
            }



            elemID = ElemMap.getElemId(0, i);
            locEnrFuncs.clear();
            locFEMFuncs = ElemMap.getFEMBasisIds(elemID);
            globalInterpolationPoints2D = ElemMap.localToGlobal(elemID, InterpolationPoints2D);
            normal.resize(2);
            normal << 0.0, -1.0;
            zeros_col = VectorXd::Zero(gpn);
            // 水平拼接 [zeros(gpn,1), localGps
            localBdryGps2D.resize(gpn, 1 + localGps.cols());
            localBdryGps2D << localGps, zeros_col;
            globalBdryGps = ElemMap.localToGlobal(elemID, localBdryGps2D);
            intBoundaryflag = false;
            MatrixXd locIntBdryGpsValue2D0_bottom;
            MatrixXd locIntBdryGpsValue2D1_bottom;
            MatrixXd locIntBdryGpsValue2D0_PU_bottom;
            MatrixXd locIntBdryGpsValue2D1_PU_bottom;


            if (partion[0][i] == 0)
            {
                locBdryKappa = bdrykappa0;
            }
            else if (partion[0][i] == 1)
            {
                locBdryKappa = bdrykappa1;
            }
            else if (partion[0][i] == -1)
            {
                VectorXd res = calc_kappa(globalBdryGps, intface);
                VectorXd ones = VectorXd::Ones(res.size());
                locBdryKappa = a0 * res + a1 * (ones - res);
                double sumKappa = locBdryKappa.sum();
                int sizeKappa = locBdryKappa.size();
                if (!(abs(sumKappa - a0 * sizeKappa) < 1e-10 ||
                    abs(sumKappa - a1 * sizeKappa) < 1e-10))
                {
                    intBoundaryflag = true;
                    auto result = calc_intbdry_gps(intfacex, intfacey, 1, i + 1, N, gpn, 1);
                    bdryGps0 = get<0>(result);
                    bdryGpW0 = get<1>(result);
                    bdryGps1 = get<2>(result);
                    bdryGpW1 = get<3>(result);
                    bdryGps0_loc = (bdryGps0.rowwise() - RowVector2d((i)*h, 0)) * N;
                    bdryGps1_loc = (bdryGps1.rowwise() - RowVector2d((i)*h, 0)) * N;

                    locIntBdryGpsValue2D0_bottom = MatrixXd::Zero(bdryGps0_loc.rows(), (order_p + 1) * (order_p + 1));
                    locIntBdryGpsValue2D1_bottom = MatrixXd::Zero(bdryGps1_loc.rows(), (order_p + 1) * (order_p + 1));

                    // 提取y坐标（第二列）
                    bdryGps0_y = bdryGps0_loc.col(1);
                    bdryGps1_y = bdryGps1_loc.col(1);

                    // 计算形状函数
                    MatrixXd intShapeFunction0 = BasisFunction::FEMShapeFunction(bdryGps0_y.transpose(), order_p, 0);
                    MatrixXd intShapeFunction1 = BasisFunction::FEMShapeFunction(bdryGps1_y.transpose(), order_p, 0);

                    // 双重循环计算
                    for (int m = 0; m < order_p + 1; ++m) {
                        for (int n = 0; n < order_p + 1; ++n) {
                            // 计算temp0和temp1
                            VectorXd temp0 = shapeFunc0(n) * intShapeFunction0.col(m);
                            VectorXd temp1 = shapeFunc0(n) * intShapeFunction1.col(m);

                            // 填充结果矩阵
                            locIntBdryGpsValue2D0_bottom.col(n * (order_p + 1) + m) = temp0;
                            locIntBdryGpsValue2D1_bottom.col(n * (order_p + 1) + m) = temp1;
                        }
                    }

                    if (EnrichMode != 0) {
                        // 初始化结果矩阵
                        locIntBdryGpsValue2D0_PU_bottom = MatrixXd::Zero(bdryGps0_loc.rows(), PUDim * PUDim);
                        locIntBdryGpsValue2D1_PU_bottom = MatrixXd::Zero(bdryGps1_loc.rows(), PUDim * PUDim);

                        // 提取 y 坐标（第二列）
                        bdryGps0_y = bdryGps0_loc.col(1);
                        bdryGps1_y = bdryGps1_loc.col(1);

                        // 计算 PU 形状函数
                        MatrixXd intPUShapeFunction0 = BasisFunction::PUShapeFunction(bdryGps0_y, 0);
                        MatrixXd intPUShapeFunction1 = BasisFunction::PUShapeFunction(bdryGps1_y, 0);

                        // 双重循环计算
                        for (int m = 0; m < PUDim; ++m) {
                            for (int n = 0; n < PUDim; ++n) {
                                // 计算 temp0 和 temp1
                                VectorXd temp0 = shapeFunc0_PU(n) * intPUShapeFunction0.col(m);
                                VectorXd temp1 = shapeFunc0_PU(n) * intPUShapeFunction1.col(m);

                                // 填充结果矩阵
                                locIntBdryGpsValue2D0_PU_bottom.col(n * PUDim + m) = temp0;
                                locIntBdryGpsValue2D1_PU_bottom.col(n * PUDim + m) = temp1;
                            }
                        }
                    }
                }
            }

            MatrixXd locBdryGpsValue2D_DPU_Minus_Interpolation_bottom;
            MatrixXd locIntBdryGpsValue2D0_DPU_Minus_Interpolation_bottom;
            MatrixXd locIntBdryGpsValue2D1_DPU_Minus_Interpolation_bottom;

            if (EnrichMode != 0)
            {
                int pos;
                enrich_flag = Ismember_single(elemID, echFaceNodeFreq, 0, pos);
                if (enrich_flag) {
                    vector<int> vertexNodeIds = { echFaceNodeFreq[pos][5],echFaceNodeFreq[pos][6],echFaceNodeFreq[pos][7],echFaceNodeFreq[pos][8] };
                    vector<size_t> nzIds = Find(vertexNodeIds);     //从零开始 无须加减一
                    vector<vector<int>> faceIds = ElemMap.getOneRingFaces(elemID);
                    FaceIds.clear();
                    for (size_t col = 0; col < faceIds[0].size(); ++col) {  // 先遍历列
                        for (size_t row = 0; row < faceIds.size(); ++row) {  // 再遍历行
                            if (col < faceIds[row].size()) {  // 防止列数不一致
                                FaceIds.push_back(faceIds[row][col]);
                            }
                        }
                    }
                    vector<int> locPUFuncs;
                    vector<bool> Flag;
                    vector<int> EchFuncId;
                    ismember(FaceIds, IntElem, Flag, EchFuncId);
                    vector<size_t> fnzIds = Find(EchFuncId);
                    for (size_t k = 0; k < fnzIds.size(); ++k) {
                        echLocalPUFunctions[pos].push_back(EchFuncId[fnzIds[k]]);
                        locPUFuncs.push_back(EchFuncId[fnzIds[k]] - 1);
                    }
                    vector<int> node_vector;
                    vector<int> echVerNodeFreq_col;
                    for (int col_idx : nzIds) {
                        if (col_idx >= 0 && col_idx < echFaceNodeFreq[pos].size()) {
                            node_vector.push_back(echFaceNodeFreq[pos][col_idx + 1]);
                        }
                    }
                    for (const auto& pair : echVerNodeFreq) {
                        echVerNodeFreq_col.push_back(pair.first);
                    }
                    vector<bool> Flag2;
                    vector<int> EchFuncId2;
                    ismember(node_vector, echVerNodeFreq_col, Flag2, EchFuncId2);  //需要减一
                    vector<double> vFreq;
                    for (int col_idx : EchFuncId2) {
                        double num = echVerNodeFreq[col_idx - 1].second;
                        vFreq.push_back(1 / num);
                    }

                    // 创建vFreq向量
                    Eigen::VectorXd vFreqVec(vFreq.size());

                    // 使用 for 循环逐元素赋值
                    for (size_t i = 0; i < vFreq.size(); ++i) {
                        vFreqVec(i) = vFreq[i];  // 注意：Eigen 使用 `(i)` 而不是 `[i]` 访问元素
                    }


                    // 提取PU_Group的子矩阵（fnzIds列）
                    Eigen::MatrixXd PU_Group_selected = PU_Group(nzIds, fnzIds);

                    // 计算 vFreq .* PU_Group(nzIds, fnzIds)
                    // 由于 vFreq 和 PU_Group_selected 的列数可能不同，需要确保维度匹配
                    // 假设 vFreq.size() == PU_Group_selected.cols()（即 nzIds 和 fnzIds 长度相同）
                    Eigen::MatrixXd term1(PU_Group_selected.rows(), PU_Group_selected.cols());
                    for (int j = 0; j < PU_Group_selected.rows(); ++j) {
                        term1.row(j) = vFreqVec(j) * PU_Group_selected.row(j);
                    }

                    // 计算最终结果
                    int N1 = enrichMutiplyN;
                    int M1 = locPUFuncs.size();      // 列向量长度
                    int K1 = fnzIds.size();          // 列向量长度

                    // 1. locEnrFuncs 矩阵： N1 × M1（列优先）
                    std::vector<int> locEnrMat(N1 * M1);
                    for (int r = 0; r < N1; ++r)           // 行
                        for (int c = 0; c < M1; ++c)       // 列
                            locEnrMat[r + c * N1] = locPUFuncs[c] * N1;

                    // 2. offset 矩阵： K1 × N1（列优先）
                    std::vector<int> offsetMat(K1 * N1);
                    for (int r = 0; r < N1; ++r)
                        for (int c = 0; c < K1; ++c)
                            offsetMat[c * N1 + r] = r + 1;


                    // 3. 逐元素相加（MATLAB 会广播到同长度）

                    locEnrFuncs.reserve(std::max(N1 * M1, K1 * N1));
                    int ML = std::min(N1 * M1, K1 * N1);
                    for (size_t i = 0; i < std::min(N1 * M1, K1 * N1); ++i)
                        locEnrFuncs.push_back(locEnrMat[i] + offsetMat[i]);

                    // 4. 加全局偏移
                    for (int& v : locEnrFuncs) v += globalFEMFunctionN;
                    locBdryGpsValue2D_DPU_Minus_Interpolation_bottom = MatrixXd::Zero(globalBdryGps.rows(), locEnrFuncs.size());
                    locIntBdryGpsValue2D0_DPU_Minus_Interpolation_bottom = MatrixXd::Zero(bdryGps0_loc.rows(), locEnrFuncs.size());
                    locIntBdryGpsValue2D1_DPU_Minus_Interpolation_bottom = MatrixXd::Zero(bdryGps1_loc.rows(), locEnrFuncs.size());

                    if (intBoundaryflag)
                    {
                        MatrixXd localPUvalues0_bottom, localPUvalues1_bottom;
                        localPUvalues0_bottom = locIntBdryGpsValue2D0_PU_bottom(Eigen::all, nzIds) * term1;
                        localPUvalues1_bottom = locIntBdryGpsValue2D1_PU_bottom(Eigen::all, nzIds) * term1;

                        for (int k = 0; k < fnzIds.size(); k++)
                        {
                            int fid = fnzIds[k];
                            double x0, y0 = 0;
                            ElemMap.LocalOriginCoordPU(elemID, fid + 1, x0, y0);
                            
                            MatrixXd DMutis_x0, DMutis_x1;
                            MatrixXd DMutis_y0, DMutis_y1;
                            
                            if (!Flag_oldxy) {
                                GenerateOffsetDMutis(bdryGps0, globalInterpolationPoints2D, order_p,
                                    x0, y0, 3 * h, 3 * h,
                                    DMutis0, DMutis_x0, DMutis_y0, DMutis0_Cof);
                                GenerateOffsetDMutis(bdryGps1, globalInterpolationPoints2D, order_p,
                                    x0, y0, 3 * h, 3 * h,
                                    DMutis1, DMutis_x1, DMutis_y1, DMutis1_Cof);
                            }
                            else {
                                GenerateOffsetDMutis(bdryGps0, globalInterpolationPoints2D, order_p,
                                    0, 0, 1, 1,
                                    DMutis0, DMutis_x0, DMutis_y0, DMutis0_Cof);
                                GenerateOffsetDMutis(bdryGps1, globalInterpolationPoints2D, order_p,
                                    0, 0, 1, 1,
                                    DMutis1, DMutis_x1, DMutis_y1, DMutis1_Cof);
                            }
                            DMutis0_Interpolation = locIntBdryGpsValue2D0_bottom * DMutis0_Cof;
                            DMutis0_Minus_Interpolation = DMutis0 - DMutis0_Interpolation;
                            DMutis1_Interpolation = locIntBdryGpsValue2D1_bottom * DMutis1_Cof;
                            DMutis1_Minus_Interpolation = DMutis1 - DMutis1_Interpolation;
                            int start_col = (k)*enrichMutiplyN;
                            scalar_vector0 = localPUvalues0_bottom.col(k);
                            scaled_matrix_0 = DMutis0_Minus_Interpolation.cwiseProduct(scalar_vector0.replicate(1, DMutis0_Minus_Interpolation.cols()));
                            scalar_vector1 = localPUvalues1_bottom.col(k);
                            scaled_matrix_1 = DMutis1_Minus_Interpolation.cwiseProduct(scalar_vector1.replicate(1, DMutis1_Minus_Interpolation.cols()));
                            locIntBdryGpsValue2D0_DPU_Minus_Interpolation_bottom.block(0, start_col,
                                locIntBdryGpsValue2D0_DPU_Minus_Interpolation_bottom.rows(), enrichMutiplyN) = scaled_matrix_0;
                            locIntBdryGpsValue2D1_DPU_Minus_Interpolation_bottom.block(0, start_col,
                                locIntBdryGpsValue2D1_DPU_Minus_Interpolation_bottom.rows(), enrichMutiplyN) = scaled_matrix_1;
                        }
                    }
                    else {
                        MatrixXd localPUvalues_bottom;
                        localPUvalues_bottom = locBdryGpsValue2D_PU_bottom(Eigen::all, nzIds) * term1;


                        for (int k = 0; k < fnzIds.size(); k++)
                        {
                            int fid = fnzIds[k];
                            double x0, y0 = 0;
                            ElemMap.LocalOriginCoordPU(elemID, fid + 1, x0, y0);
                            MatrixXd DMutis;
                            MatrixXd DMutis_x;
                            MatrixXd DMutis_y;
                            MatrixXd DMutis_Cof;
                            if (!Flag_oldxy) {
                                GenerateOffsetDMutis(globalBdryGps, globalInterpolationPoints2D, order_p,
                                    x0, y0, 3 * h, 3 * h,
                                    DMutis, DMutis_x, DMutis_y, DMutis_Cof);
                            }
                            else {
                                GenerateOffsetDMutis(globalBdryGps, globalInterpolationPoints2D, order_p,
                                    0, 0, 1, 1,
                                    DMutis, DMutis_x, DMutis_y, DMutis_Cof);
                            }
                            DMutis_Interpolation = locBdryGpsValue2D_bottom * DMutis_Cof;
                            DMutis_Minus_Interpolation = DMutis - DMutis_Interpolation;
                            int start_col = (k)*enrichMutiplyN;
                            scalar_vector = localPUvalues_bottom.col(k);
                            scaled_matrix = DMutis_Minus_Interpolation.cwiseProduct(scalar_vector.replicate(1, DMutis_Minus_Interpolation.cols()));
                            locBdryGpsValue2D_DPU_Minus_Interpolation_bottom.block(0, start_col,
                                locBdryGpsValue2D_DPU_Minus_Interpolation_bottom.rows(), enrichMutiplyN) = scaled_matrix;
                        }
                    }
                }
            }

            if (EnrichMode == 1 && enrich_flag && OrthogonalMode)
            {
                for (int fid : FaceIds)
                {
                    auto result = ismember_single(fid, IntElem);
                    bool flagInt = result.first;
                    int posInt = result.second;

                    if (flagInt)
                    {
                        vector<int> enrfuns(enrichMutiplyN);

                        for (int i = 0; i < enrichMutiplyN; ++i)
                        {
                            enrfuns[i] = globalFEMFunctionN + (posInt)*enrichMutiplyN + (i + 1);
                        }

                        vector<bool> Flag;
                        vector<int> posF;
                        ismember(enrfuns, locEnrFuncs, Flag, posF);
                        for (int m = 0; m < posF.size(); m++)
                        {
                            posF[m] -= 1;      // 减一变成从零开始
                        }
                        MatrixXd cofL = OrthogonalMatrix[posInt][0];
                        if (!intBoundaryflag)
                        {
                            MatrixXd tempvalues = locBdryGpsValue2D_DPU_Minus_Interpolation_bottom(Eigen::all, posF);
                            for (int k = 1; k < enrichMutiplyN; ++k) {  // k从2开始变为1开始
                                for (int kk = 0; kk < k; ++kk) {       // kk从1到k-1变为0到k-1
                                    tempvalues.col(k) -= cofL(k, kk) * tempvalues.col(kk);
                                }
                            }
                            for (int k = 0; k < enrichMutiplyN; ++k) {  // k从1开始变为0开始
                                tempvalues.col(k) *= cofL(k, k);
                            }
                            locBdryGpsValue2D_DPU_Minus_Interpolation_bottom(Eigen::all, posF) = tempvalues;
                        }
                        else
                        {
                            MatrixXd tempvalues0 = locIntBdryGpsValue2D0_DPU_Minus_Interpolation_bottom(Eigen::all, posF);
                            MatrixXd tempvalues1 = locIntBdryGpsValue2D1_DPU_Minus_Interpolation_bottom(Eigen::all, posF);
                            for (int k = 1; k < enrichMutiplyN; ++k) {  // k从2开始变为1开始
                                for (int kk = 0; kk < k; ++kk) {       // kk从1到k-1变为0到k-1
                                    tempvalues0.col(k) -= cofL(k, kk) * tempvalues0.col(kk);
                                    tempvalues1.col(k) -= cofL(k, kk) * tempvalues1.col(kk);
                                }
                            }
                            for (int k = 0; k < enrichMutiplyN; ++k) {  // k从1开始变为0开始
                                tempvalues0.col(k) *= cofL(k, k);
                                tempvalues1.col(k) *= cofL(k, k);
                            }
                            locIntBdryGpsValue2D0_DPU_Minus_Interpolation_bottom(Eigen::all, posF) = tempvalues0;
                            locIntBdryGpsValue2D1_DPU_Minus_Interpolation_bottom(Eigen::all, posF) = tempvalues1;

                        }
                    }
                }
            }

            locG;  // 结果向量

            if (!intBoundaryflag) {
                // 非分片边界情况
                int n = globalBdryGps.rows();
                VectorXd val_ux(n), val_uy(n);

                // 逐点计算标量函数
                for (int i = 0; i < n; ++i) {
                    val_ux(i) = calc_exact_ux(globalBdryGps(i, 0), globalBdryGps(i, 1));
                    val_uy(i) = calc_exact_uy(globalBdryGps(i, 0), globalBdryGps(i, 1));
                }

                // 计算 g = [val_ux val_uy] * normal
                VectorXd g = val_ux * normal(0) + val_uy * normal(1);

                // 计算加权项并组装矩阵
                VectorXd weighted_g = g.array() * locBdryKappa.array() * localGpWs.array();
                
                // 方案二：直接计算，完全避免中间矩阵
                if (locBdryGpsValue2D_DPU_Minus_Interpolation_bottom.cols() == 0) {
                    // 无富集情况，直接计算
                    locG = locBdryGpsValue2D_bottom.transpose() * weighted_g * gJac;
                } else {
                    // 有富集情况，分段计算填入locG
                    int n1 = locBdryGpsValue2D_bottom.cols();
                    int n2 = locBdryGpsValue2D_DPU_Minus_Interpolation_bottom.cols();
                    
                    // 分配locG的空间
                    if (locG.size() != n1 + n2) {
                        locG.resize(n1 + n2);
                    }
                    
                    // 分段计算并填入locG
                    locG.segment(0, n1) = locBdryGpsValue2D_bottom.transpose() * weighted_g * gJac;
                    locG.segment(n1, n2) = locBdryGpsValue2D_DPU_Minus_Interpolation_bottom.transpose() * weighted_g * gJac;
                }
            }
            else {
                // 分片边界情况
                int n0 = bdryGps0.rows();
                int n1 = bdryGps1.rows();

                // 计算第一组边界点
                VectorXd val_ux0(n0), val_uy0(n0);
                for (int i = 0; i < n0; ++i) {
                    val_ux0(i) = calc_exact_ux(bdryGps0(i, 0), bdryGps0(i, 1));
                    val_uy0(i) = calc_exact_uy(bdryGps0(i, 0), bdryGps0(i, 1));
                }
                VectorXd g0 = val_ux0 * normal(0) + val_uy0 * normal(1);

                // 计算第二组边界点
                VectorXd val_ux1(n1), val_uy1(n1);
                for (int i = 0; i < n1; ++i) {
                    val_ux1(i) = calc_exact_ux(bdryGps1(i, 0), bdryGps1(i, 1));
                    val_uy1(i) = calc_exact_uy(bdryGps1(i, 0), bdryGps1(i, 1));
                }
                VectorXd g1 = val_ux1 * normal(0) + val_uy1 * normal(1);

                // 方案二：直接计算，避免中间矩阵combined_matrix0和combined_matrix1
                
                // 计算第一组的贡献
                VectorXd part0;
                if (locIntBdryGpsValue2D0_DPU_Minus_Interpolation_bottom.cols() == 0) {
                    // 无富集
                    part0 = locIntBdryGpsValue2D0_bottom.transpose() * (g0.cwiseProduct(bdryGpW0));
                } else {
                    // 有富集，分段计算
                    int n0_1 = locIntBdryGpsValue2D0_bottom.cols();
                    int n0_2 = locIntBdryGpsValue2D0_DPU_Minus_Interpolation_bottom.cols();
                    part0.resize(n0_1 + n0_2);
                    part0.segment(0, n0_1) = locIntBdryGpsValue2D0_bottom.transpose() * (g0.cwiseProduct(bdryGpW0));
                    part0.segment(n0_1, n0_2) = locIntBdryGpsValue2D0_DPU_Minus_Interpolation_bottom.transpose() * (g0.cwiseProduct(bdryGpW0));
                }
                
                // 计算第二组的贡献
                VectorXd part1;
                if (locIntBdryGpsValue2D1_DPU_Minus_Interpolation_bottom.cols() == 0) {
                    // 无富集
                    part1 = locIntBdryGpsValue2D1_bottom.transpose() * (g1.cwiseProduct(bdryGpW1));
                } else {
                    // 有富集，分段计算
                    int n1_1 = locIntBdryGpsValue2D1_bottom.cols();
                    int n1_2 = locIntBdryGpsValue2D1_DPU_Minus_Interpolation_bottom.cols();
                    part1.resize(n1_1 + n1_2);
                    part1.segment(0, n1_1) = locIntBdryGpsValue2D1_bottom.transpose() * (g1.cwiseProduct(bdryGpW1));
                    part1.segment(n1_1, n1_2) = locIntBdryGpsValue2D1_DPU_Minus_Interpolation_bottom.transpose() * (g1.cwiseProduct(bdryGpW1));
                }
                
                // 最终结果
                if (locG.size() != part0.size()) {
                    locG.resize(part0.size());
                }
                locG = a0 * part0 + a1 * part1;
            }

            locFuncs.clear();
            locFuncs.reserve(locFEMFuncs.size() + locEnrFuncs.size());
            locFuncs.insert(locFuncs.end(), locFEMFuncs.begin(), locFEMFuncs.end());
            locFuncs.insert(locFuncs.end(), locEnrFuncs.begin(), locEnrFuncs.end());

            // 2. 检查维度匹配
            assert(locFuncs.size() == locG.size() && "locG size must match locFuncs size");

            // 3. 执行累加操作 (转换为C++的0-based索引)
            for (size_t i = 0; i < locFuncs.size(); ++i) {
                int idx = locFuncs[i];  // MATLAB 1-based 转 C++ 0-based
                assert(idx >= 0 && idx < F.size() && "Index out of bounds");
                F(idx - 1) += locG(i);
            }


            elemID = ElemMap.getElemId(i, N - 1);
            locEnrFuncs.clear();
            locFEMFuncs = ElemMap.getFEMBasisIds(elemID);
            globalInterpolationPoints2D = ElemMap.localToGlobal(elemID, InterpolationPoints2D);
            normal.resize(2);
            normal << 1.0, 0.0;
            zeros_col = VectorXd::Ones(gpn);
            // 水平拼接 [zeros(gpn,1), localGps]
            localBdryGps2D = MatrixXd();
            localBdryGps2D.resize(gpn, 1 + localGps.cols());
            localBdryGps2D << zeros_col, localGps;
            globalBdryGps = ElemMap.localToGlobal(elemID, localBdryGps2D);
            intBoundaryflag = false;
            MatrixXd locIntBdryGpsValue2D0_right;
            MatrixXd locIntBdryGpsValue2D1_right;
            MatrixXd locIntBdryGpsValue2D0_PU_right;
            MatrixXd locIntBdryGpsValue2D1_PU_right;
            if (partion[i][N - 1] == 0)
            {
                locBdryKappa = bdrykappa0;
            }
            else if (partion[i][N - 1] == 1)
            {
                locBdryKappa = bdrykappa1;
            }
            else if (partion[i][N - 1] == -1)
            {
                VectorXd res = calc_kappa(globalBdryGps, intface);
                VectorXd ones = VectorXd::Ones(res.size());
                locBdryKappa = a0 * res + a1 * (ones - res);
                double sumKappa = locBdryKappa.sum();
                int sizeKappa = locBdryKappa.size();
                if (!(abs(sumKappa - a0 * sizeKappa) < 1e-10 ||
                    abs(sumKappa - a1 * sizeKappa) < 1e-10))
                {
                    intBoundaryflag = true;
                    auto result = calc_intbdry_gps(intfacex, intfacey, i + 1, N, N, gpn, 2);
                    bdryGps0 = get<0>(result);
                    bdryGpW0 = get<1>(result);
                    bdryGps1 = get<2>(result);
                    bdryGpW1 = get<3>(result);
                    bdryGps0_loc = (bdryGps0.rowwise() - RowVector2d(1 - h, (i)*h)) * N;
                    bdryGps1_loc = (bdryGps1.rowwise() - RowVector2d(1 - h, (i)*h)) * N;

                    locIntBdryGpsValue2D0_right = MatrixXd::Zero(bdryGps0_loc.rows(), (order_p + 1) * (order_p + 1));
                    locIntBdryGpsValue2D1_right = MatrixXd::Zero(bdryGps1_loc.rows(), (order_p + 1) * (order_p + 1));

                    // 提取y坐标（第二列）
                    bdryGps0_y = bdryGps0_loc.col(1);
                    bdryGps1_y = bdryGps1_loc.col(1);

                    // 计算形状函数
                    MatrixXd intShapeFunction0 = BasisFunction::FEMShapeFunction(bdryGps0_y.transpose(), order_p, 0);
                    MatrixXd intShapeFunction1 = BasisFunction::FEMShapeFunction(bdryGps1_y.transpose(), order_p, 0);

                    // 双重循环计算
                    for (int m = 0; m < order_p + 1; ++m) {
                        for (int n = 0; n < order_p + 1; ++n) {
                            // 计算temp0和temp1
                            VectorXd temp0 = shapeFunc1(n) * intShapeFunction0.col(m);
                            VectorXd temp1 = shapeFunc1(n) * intShapeFunction1.col(m);

                            // 填充结果矩阵
                            locIntBdryGpsValue2D0_right.col(n * (order_p + 1) + m) = temp0;
                            locIntBdryGpsValue2D1_right.col(n * (order_p + 1) + m) = temp1;
                        }
                    }

                    if (EnrichMode != 0) {
                        // 初始化结果矩阵
                        locIntBdryGpsValue2D0_PU_right = MatrixXd::Zero(bdryGps0_loc.rows(), PUDim * PUDim);
                        locIntBdryGpsValue2D1_PU_right = MatrixXd::Zero(bdryGps1_loc.rows(), PUDim * PUDim);

                        // 提取 y 坐标（第二列）
                        bdryGps0_y = bdryGps0_loc.col(1);
                        bdryGps1_y = bdryGps1_loc.col(1);

                        // 计算 PU 形状函数
                        MatrixXd intPUShapeFunction0 = BasisFunction::PUShapeFunction(bdryGps0_y, 0);
                        MatrixXd intPUShapeFunction1 = BasisFunction::PUShapeFunction(bdryGps1_y, 0);

                        // 双重循环计算
                        for (int m = 0; m < PUDim; ++m) {
                            for (int n = 0; n < PUDim; ++n) {
                                // 计算 temp0 和 temp1
                                VectorXd temp0 = shapeFunc1_PU(n) * intPUShapeFunction0.col(m);
                                VectorXd temp1 = shapeFunc1_PU(n) * intPUShapeFunction1.col(m);

                                // 填充结果矩阵
                                locIntBdryGpsValue2D0_PU_right.col(n * PUDim + m) = temp0;
                                locIntBdryGpsValue2D1_PU_right.col(n * PUDim + m) = temp1;
                            }
                        }
                    }
                }
            }

            MatrixXd locBdryGpsValue2D_DPU_Minus_Interpolation_right;
            MatrixXd locIntBdryGpsValue2D0_DPU_Minus_Interpolation_right;
            MatrixXd locIntBdryGpsValue2D1_DPU_Minus_Interpolation_right;

            if (EnrichMode != 0)
            {
                int pos;
                enrich_flag = Ismember_single(elemID, echFaceNodeFreq, 0, pos);
                if (enrich_flag) {
                    vector<int> vertexNodeIds = { echFaceNodeFreq[pos][5],echFaceNodeFreq[pos][6],echFaceNodeFreq[pos][7],echFaceNodeFreq[pos][8] };
                    vector<size_t> nzIds = Find(vertexNodeIds);     //从零开始 无须加减一
                    vector<vector<int>> faceIds = ElemMap.getOneRingFaces(elemID);
                    FaceIds.clear();
                    for (size_t col = 0; col < faceIds[0].size(); ++col) {  // 先遍历列
                        for (size_t row = 0; row < faceIds.size(); ++row) {  // 再遍历行
                            if (col < faceIds[row].size()) {  // 防止列数不一致
                                FaceIds.push_back(faceIds[row][col]);
                            }
                        }
                    }
                    vector<int> locPUFuncs;
                    vector<bool> Flag;
                    vector<int> EchFuncId;
                    ismember(FaceIds, IntElem, Flag, EchFuncId);
                    vector<size_t> fnzIds = Find(EchFuncId);
                    for (size_t k = 0; k < fnzIds.size(); ++k) {
                        echLocalPUFunctions[pos].push_back(EchFuncId[fnzIds[k]]);
                        locPUFuncs.push_back(EchFuncId[fnzIds[k]] - 1);
                    }
                    vector<int> node_vector;
                    vector<int> echVerNodeFreq_col;
                    for (int col_idx : nzIds) {
                        if (col_idx >= 0 && col_idx < echFaceNodeFreq[pos].size()) {
                            node_vector.push_back(echFaceNodeFreq[pos][col_idx + 1]);
                        }
                    }
                    for (const auto& pair : echVerNodeFreq) {
                        echVerNodeFreq_col.push_back(pair.first);
                    }
                    vector<bool> Flag2;
                    vector<int> EchFuncId2;
                    ismember(node_vector, echVerNodeFreq_col, Flag2, EchFuncId2);  //需要减一
                    vector<double> vFreq;
                    for (int col_idx : EchFuncId2) {
                        double num = echVerNodeFreq[col_idx - 1].second;
                        vFreq.push_back(1 / num);
                    }

                    // 创建vFreq向量
                    Eigen::VectorXd vFreqVec(vFreq.size());

                    // 使用 for 循环逐元素赋值
                    for (size_t i = 0; i < vFreq.size(); ++i) {
                        vFreqVec(i) = vFreq[i];  // 注意：Eigen 使用 `(i)` 而不是 `[i]` 访问元素
                    }


                    // 提取PU_Group的子矩阵（fnzIds列）
                    Eigen::MatrixXd PU_Group_selected = PU_Group(nzIds, fnzIds);

                    // 计算 vFreq .* PU_Group(nzIds, fnzIds)
                    // 由于 vFreq 和 PU_Group_selected 的列数可能不同，需要确保维度匹配
                    // 假设 vFreq.size() == PU_Group_selected.cols()（即 nzIds 和 fnzIds 长度相同）
                    Eigen::MatrixXd term1(PU_Group_selected.rows(), PU_Group_selected.cols());
                    for (int j = 0; j < PU_Group_selected.rows(); ++j) {
                        term1.row(j) = vFreqVec(j) * PU_Group_selected.row(j);
                    }

                    // 计算最终结果
                    int N1 = enrichMutiplyN;
                    int M1 = locPUFuncs.size();      // 列向量长度
                    int K1 = fnzIds.size();          // 列向量长度

                    // 1. locEnrFuncs 矩阵： N1 × M1（列优先）
                    std::vector<int> locEnrMat(N1 * M1);
                    for (int r = 0; r < N1; ++r)           // 行
                        for (int c = 0; c < M1; ++c)       // 列
                            locEnrMat[r + c * N1] = locPUFuncs[c] * N1;

                    // 2. offset 矩阵： K1 × N1（列优先）
                    std::vector<int> offsetMat(K1 * N1);
                    for (int r = 0; r < N1; ++r)
                        for (int c = 0; c < K1; ++c)
                            offsetMat[c * N1 + r] = r + 1;


                    // 3. 逐元素相加（MATLAB 会广播到同长度）

                    locEnrFuncs.reserve(std::max(N1 * M1, K1 * N1));
                    int ML = std::min(N1 * M1, K1 * N1);
                    for (size_t i = 0; i < std::min(N1 * M1, K1 * N1); ++i)
                        locEnrFuncs.push_back(locEnrMat[i] + offsetMat[i]);

                    // 4. 加全局偏移
                    for (int& v : locEnrFuncs) v += globalFEMFunctionN;
                    locIntBdryGpsValue2D0_DPU_Minus_Interpolation_right = MatrixXd::Zero(bdryGps0_loc.rows(), locEnrFuncs.size());
                    locIntBdryGpsValue2D1_DPU_Minus_Interpolation_right = MatrixXd::Zero(bdryGps1_loc.rows(), locEnrFuncs.size());
                    locBdryGpsValue2D_DPU_Minus_Interpolation_right = MatrixXd::Zero(globalBdryGps.rows(), locEnrFuncs.size());
                    if (intBoundaryflag)
                    {
                        MatrixXd localPUvalues0_right, localPUvalues1_right;
                        localPUvalues0_right = locIntBdryGpsValue2D0_PU_right(Eigen::all, nzIds) * term1;
                        localPUvalues1_right = locIntBdryGpsValue2D1_PU_right(Eigen::all, nzIds) * term1;


                        for (int k = 0; k < fnzIds.size(); k++)
                        {
                            int fid = fnzIds[k];
                            double x0, y0 = 0;
                            ElemMap.LocalOriginCoordPU(elemID, fid + 1, x0, y0);
                            MatrixXd DMutis0, DMutis1;
                            MatrixXd DMutis_x0, DMutis_x1;
                            MatrixXd DMutis_y0, DMutis_y1;
                            MatrixXd DMutis0_Cof, DMutis1_Cof;
                            if (!Flag_oldxy) {
                                GenerateOffsetDMutis(bdryGps0, globalInterpolationPoints2D, order_p,
                                    x0, y0, 3 * h, 3 * h,
                                    DMutis0, DMutis_x0, DMutis_y0, DMutis0_Cof);
                                GenerateOffsetDMutis(bdryGps1, globalInterpolationPoints2D, order_p,
                                    x0, y0, 3 * h, 3 * h,
                                    DMutis1, DMutis_x1, DMutis_y1, DMutis1_Cof);
                            }
                            else {
                                GenerateOffsetDMutis(bdryGps0, globalInterpolationPoints2D, order_p,
                                    0, 0, 1, 1,
                                    DMutis0, DMutis_x0, DMutis_y0, DMutis0_Cof);
                                GenerateOffsetDMutis(bdryGps1, globalInterpolationPoints2D, order_p,
                                    0, 0, 1, 1,
                                    DMutis1, DMutis_x1, DMutis_y1, DMutis1_Cof);
                            }
                            DMutis0_Interpolation = locIntBdryGpsValue2D0_right * DMutis0_Cof;
                            DMutis0_Minus_Interpolation = DMutis0 - DMutis0_Interpolation;
                            DMutis1_Interpolation = locIntBdryGpsValue2D1_right * DMutis1_Cof;
                            DMutis1_Minus_Interpolation = DMutis1 - DMutis1_Interpolation;
                            int start_col = (k)*enrichMutiplyN;
                            scalar_vector0 = localPUvalues0_right.col(k);
                            scaled_matrix_0 = DMutis0_Minus_Interpolation.cwiseProduct(scalar_vector0.replicate(1, DMutis0_Minus_Interpolation.cols()));
                            scalar_vector1 = localPUvalues1_right.col(k);
                            scaled_matrix_1 = DMutis1_Minus_Interpolation.cwiseProduct(scalar_vector1.replicate(1, DMutis1_Minus_Interpolation.cols()));
                            locIntBdryGpsValue2D0_DPU_Minus_Interpolation_right.block(0, start_col,
                                locIntBdryGpsValue2D0_DPU_Minus_Interpolation_right.rows(), enrichMutiplyN) = scaled_matrix_0;
                            locIntBdryGpsValue2D1_DPU_Minus_Interpolation_right.block(0, start_col,
                                locIntBdryGpsValue2D1_DPU_Minus_Interpolation_right.rows(), enrichMutiplyN) = scaled_matrix_1;
                        }
                    }
                    else {
                        MatrixXd localPUvalues_right;
                        localPUvalues_right = locBdryGpsValue2D_PU_right(Eigen::all, nzIds) * term1;


                        for (int k = 0; k < fnzIds.size(); k++)
                        {
                            int fid = fnzIds[k];
                            double x0, y0 = 0;
                            ElemMap.LocalOriginCoordPU(elemID, fid + 1, x0, y0);
                            MatrixXd DMutis;
                            MatrixXd DMutis_x;
                            MatrixXd DMutis_y;
                            MatrixXd DMutis_Cof;
                            if (!Flag_oldxy) {
                                GenerateOffsetDMutis(globalBdryGps, globalInterpolationPoints2D, order_p,
                                    x0, y0, 3 * h, 3 * h,
                                    DMutis, DMutis_x, DMutis_y, DMutis_Cof);
                            }
                            else {
                                GenerateOffsetDMutis(globalBdryGps, globalInterpolationPoints2D, order_p,
                                    0, 0, 1, 1,
                                    DMutis, DMutis_x, DMutis_y, DMutis_Cof);
                            }
                            DMutis_Interpolation = locBdryGpsValue2D_right * DMutis_Cof;
                            DMutis_Minus_Interpolation = DMutis - DMutis_Interpolation;
                            int start_col = (k)*enrichMutiplyN;
                            scalar_vector = localPUvalues_right.col(k);
                            scaled_matrix = DMutis_Minus_Interpolation.cwiseProduct(scalar_vector.replicate(1, DMutis_Minus_Interpolation.cols()));
                            locBdryGpsValue2D_DPU_Minus_Interpolation_right.block(0, start_col,
                                locBdryGpsValue2D_DPU_Minus_Interpolation_right.rows(), enrichMutiplyN) = scaled_matrix;
                        }
                    }
                }
            }

            if (EnrichMode == 1 && enrich_flag && OrthogonalMode)
            {
                for (int fid : FaceIds)
                {
                    auto result = ismember_single(fid, IntElem);
                    bool flagInt = result.first;
                    int posInt = result.second;

                    if (flagInt)
                    {
                        vector<int> enrfuns(enrichMutiplyN);
                        for (int i = 0; i < enrichMutiplyN; ++i) {
                            enrfuns[i] = globalFEMFunctionN + (posInt)*enrichMutiplyN + (i + 1);
                        }

                        vector<bool> Flag;
                        vector<int> posF;
                        ismember(enrfuns, locEnrFuncs, Flag, posF);
                        for (int m = 0; m < posF.size(); m++)
                        {
                            posF[m] -= 1;      // 减一变成从零开始
                        }
                        MatrixXd cofL = OrthogonalMatrix[posInt][0];
                        if (!intBoundaryflag)
                        {
                            MatrixXd tempvalues = locBdryGpsValue2D_DPU_Minus_Interpolation_right(Eigen::all, posF);
                            for (int k = 1; k < enrichMutiplyN; ++k) {
                                for (int kk = 0; kk < k; ++kk) {
                                    tempvalues.col(k) -= cofL(k, kk) * tempvalues.col(kk);
                                }
                            }
                            for (int k = 0; k < enrichMutiplyN; ++k) {
                                tempvalues.col(k) *= cofL(k, k);
                            }
                            locBdryGpsValue2D_DPU_Minus_Interpolation_right(Eigen::all, posF) = tempvalues;
                        }
                        else
                        {
                            MatrixXd tempvalues0 = locIntBdryGpsValue2D0_DPU_Minus_Interpolation_right(Eigen::all, posF);
                            MatrixXd tempvalues1 = locIntBdryGpsValue2D1_DPU_Minus_Interpolation_right(Eigen::all, posF);
                            for (int k = 1; k < enrichMutiplyN; ++k) {
                                for (int kk = 0; kk < k; ++kk) {
                                    tempvalues0.col(k) -= cofL(k, kk) * tempvalues0.col(kk);
                                    tempvalues1.col(k) -= cofL(k, kk) * tempvalues1.col(kk);
                                }
                            }
                            for (int k = 0; k < enrichMutiplyN; ++k) {
                                tempvalues0.col(k) *= cofL(k, k);
                                tempvalues1.col(k) *= cofL(k, k);
                            }
                            locIntBdryGpsValue2D0_DPU_Minus_Interpolation_right(Eigen::all, posF) = tempvalues0;
                            locIntBdryGpsValue2D1_DPU_Minus_Interpolation_right(Eigen::all, posF) = tempvalues1;
                        }
                    }
                }
            }

            locG;

            if (!intBoundaryflag) {
                int n = globalBdryGps.rows();
                VectorXd val_ux(n), val_uy(n);

                for (int i = 0; i < n; ++i) {
                    val_ux(i) = calc_exact_ux(globalBdryGps(i, 0), globalBdryGps(i, 1));
                    val_uy(i) = calc_exact_uy(globalBdryGps(i, 0), globalBdryGps(i, 1));
                }

                VectorXd g = val_ux * normal(0) + val_uy * normal(1);
                VectorXd weighted_g = g.array() * locBdryKappa.array() * localGpWs.array();
                
                // 方案二：直接计算，完全避免中间矩阵
                if (locBdryGpsValue2D_DPU_Minus_Interpolation_right.cols() == 0) {
                    // 无富集情况，直接计算
                    locG = locBdryGpsValue2D_right.transpose() * weighted_g * gJac;
                } else {
                    // 有富集情况，分段计算填入locG
                    int n1 = locBdryGpsValue2D_right.cols();
                    int n2 = locBdryGpsValue2D_DPU_Minus_Interpolation_right.cols();
                    
                    // 分配locG的空间
                    if (locG.size() != n1 + n2) {
                        locG.resize(n1 + n2);
                    }
                    
                    // 分段计算并填入locG
                    locG.segment(0, n1) = locBdryGpsValue2D_right.transpose() * weighted_g * gJac;
                    locG.segment(n1, n2) = locBdryGpsValue2D_DPU_Minus_Interpolation_right.transpose() * weighted_g * gJac;
                }
            }
            else {
                int n0 = bdryGps0.rows();
                int n1 = bdryGps1.rows();

                VectorXd val_ux0(n0), val_uy0(n0);
                for (int i = 0; i < n0; ++i) {
                    val_ux0(i) = calc_exact_ux(bdryGps0(i, 0), bdryGps0(i, 1));
                    val_uy0(i) = calc_exact_uy(bdryGps0(i, 0), bdryGps0(i, 1));
                }
                VectorXd g0 = val_ux0 * normal(0) + val_uy0 * normal(1);

                VectorXd val_ux1(n1), val_uy1(n1);
                for (int i = 0; i < n1; ++i) {
                    val_ux1(i) = calc_exact_ux(bdryGps1(i, 0), bdryGps1(i, 1));
                    val_uy1(i) = calc_exact_uy(bdryGps1(i, 0), bdryGps1(i, 1));
                }
                VectorXd g1 = val_ux1 * normal(0) + val_uy1 * normal(1);

                // 方案二：直接计算，避免中间矩阵combined_matrix0和combined_matrix1
                
                // 计算第一组的贡献
                VectorXd part0;
                if (locIntBdryGpsValue2D0_DPU_Minus_Interpolation_right.cols() == 0) {
                    // 无富集
                    part0 = locIntBdryGpsValue2D0_right.transpose() * (g0.cwiseProduct(bdryGpW0));
                } else {
                    // 有富集，分段计算
                    int n0_1 = locIntBdryGpsValue2D0_right.cols();
                    int n0_2 = locIntBdryGpsValue2D0_DPU_Minus_Interpolation_right.cols();
                    part0.resize(n0_1 + n0_2);
                    part0.segment(0, n0_1) = locIntBdryGpsValue2D0_right.transpose() * (g0.cwiseProduct(bdryGpW0));
                    part0.segment(n0_1, n0_2) = locIntBdryGpsValue2D0_DPU_Minus_Interpolation_right.transpose() * (g0.cwiseProduct(bdryGpW0));
                }
                
                // 计算第二组的贡献
                VectorXd part1;
                if (locIntBdryGpsValue2D1_DPU_Minus_Interpolation_right.cols() == 0) {
                    // 无富集
                    part1 = locIntBdryGpsValue2D1_right.transpose() * (g1.cwiseProduct(bdryGpW1));
                } else {
                    // 有富集，分段计算
                    int n1_1 = locIntBdryGpsValue2D1_right.cols();
                    int n1_2 = locIntBdryGpsValue2D1_DPU_Minus_Interpolation_right.cols();
                    part1.resize(n1_1 + n1_2);
                    part1.segment(0, n1_1) = locIntBdryGpsValue2D1_right.transpose() * (g1.cwiseProduct(bdryGpW1));
                    part1.segment(n1_1, n1_2) = locIntBdryGpsValue2D1_DPU_Minus_Interpolation_right.transpose() * (g1.cwiseProduct(bdryGpW1));
                }
                
                // 最终结果
                if (locG.size() != part0.size()) {
                    locG.resize(part0.size());
                }
                locG = a0 * part0 + a1 * part1;
            }

            locFuncs.clear();
            if (locEnrFuncs.empty()) {
                locFuncs = locFEMFuncs;
            }
            else {
                locFuncs.reserve(locFEMFuncs.size() + locEnrFuncs.size());
                locFuncs.insert(locFuncs.end(), locFEMFuncs.begin(), locFEMFuncs.end());
                locFuncs.insert(locFuncs.end(), locEnrFuncs.begin(), locEnrFuncs.end());
            }


            int row = locG.rows();
            for (size_t i = 0; i < locFuncs.size(); ++i) {
                int idx = locFuncs[i];
                assert(idx >= 0 && idx < F.size() && "Index out of bounds");
                F(idx - 1) += locG(i);
            }


            elemID = ElemMap.getElemId(N - 1, i);
            locEnrFuncs.clear();
            locFEMFuncs = ElemMap.getFEMBasisIds(elemID);
            globalInterpolationPoints2D = ElemMap.localToGlobal(elemID, InterpolationPoints2D);
            normal.resize(2);
            normal << 0.0, 1.0;
            zeros_col = VectorXd::Ones(gpn);
            // 水平拼接 [zeros(gpn,1), localGps]
            localBdryGps2D = MatrixXd();
            localBdryGps2D.resize(gpn, 1 + localGps.cols());
            localBdryGps2D << localGps, zeros_col;
            // 水平拼接 [zeros(gpn,1), localGps]
            globalBdryGps = ElemMap.localToGlobal(elemID, localBdryGps2D);
            intBoundaryflag = false;
            MatrixXd locIntBdryGpsValue2D0_top;
            MatrixXd locIntBdryGpsValue2D1_top;
            MatrixXd locIntBdryGpsValue2D0_PU_top;
            MatrixXd locIntBdryGpsValue2D1_PU_top;
            if (partion[N - 1][i] == 0)
            {
                locBdryKappa = bdrykappa0;
            }
            else if (partion[N - 1][i] == 1)
            {
                locBdryKappa = bdrykappa1;
            }
            else if (partion[N - 1][i] == -1)
            {
                VectorXd res = calc_kappa(globalBdryGps, intface);
                VectorXd ones = VectorXd::Ones(res.size());
                locBdryKappa = a0 * res + a1 * (ones - res);
                double sumKappa = locBdryKappa.sum();
                int sizeKappa = locBdryKappa.size();
                if (!(abs(sumKappa - a0 * sizeKappa) < 1e-10 ||
                    abs(sumKappa - a1 * sizeKappa) < 1e-10))
                {
                    intBoundaryflag = true;
                    auto result = calc_intbdry_gps(intfacex, intfacey, N, i + 1, N, gpn, 3);
                    bdryGps0 = get<0>(result);
                    bdryGpW0 = get<1>(result);
                    bdryGps1 = get<2>(result);
                    bdryGpW1 = get<3>(result);
                    bdryGps0_loc = (bdryGps0.rowwise() - RowVector2d((i)*h, 1 - h)) * N;
                    bdryGps1_loc = (bdryGps1.rowwise() - RowVector2d((i)*h, 1 - h)) * N;

                    locIntBdryGpsValue2D0_top = MatrixXd::Zero(bdryGps0_loc.rows(), (order_p + 1) * (order_p + 1));
                    locIntBdryGpsValue2D1_top = MatrixXd::Zero(bdryGps1_loc.rows(), (order_p + 1) * (order_p + 1));

                    // 提取y坐标（第二列）
                    bdryGps0_y = bdryGps0_loc.col(1);
                    bdryGps1_y = bdryGps1_loc.col(1);

                    // 计算形状函数
                    MatrixXd intShapeFunction0 = BasisFunction::FEMShapeFunction(bdryGps0_y.transpose(), order_p, 0);
                    MatrixXd intShapeFunction1 = BasisFunction::FEMShapeFunction(bdryGps1_y.transpose(), order_p, 0);

                    // 双重循环计算
                    for (int m = 0; m < order_p + 1; ++m) {
                        for (int n = 0; n < order_p + 1; ++n) {
                            // 计算temp0和temp1
                            VectorXd temp0 = shapeFunc1(n) * intShapeFunction0.col(m);
                            VectorXd temp1 = shapeFunc1(n) * intShapeFunction1.col(m);

                            // 填充结果矩阵
                            locIntBdryGpsValue2D0_top.col(n * (order_p + 1) + m) = temp0;
                            locIntBdryGpsValue2D1_top.col(n * (order_p + 1) + m) = temp1;
                        }
                    }

                    if (EnrichMode != 0) {
                        // 初始化结果矩阵
                        MatrixXd locIntBdryGpsValue2D0_PU_top = MatrixXd::Zero(bdryGps0_loc.rows(), PUDim * PUDim);
                        MatrixXd locIntBdryGpsValue2D1_PU_top = MatrixXd::Zero(bdryGps1_loc.rows(), PUDim * PUDim);

                        // 提取 y 坐标（第二列）
                        bdryGps0_y = bdryGps0_loc.col(1);
                        bdryGps1_y = bdryGps1_loc.col(1);

                        // 计算 PU 形状函数
                        MatrixXd intPUShapeFunction0 = BasisFunction::PUShapeFunction(bdryGps0_y, 0);
                        MatrixXd intPUShapeFunction1 = BasisFunction::PUShapeFunction(bdryGps1_y, 0);

                        // 双重循环计算
                        for (int m = 0; m < PUDim; ++m) {
                            for (int n = 0; n < PUDim; ++n) {
                                // 计算 temp0 和 temp1
                                VectorXd temp0 = shapeFunc1_PU(n) * intPUShapeFunction0.col(m);
                                VectorXd temp1 = shapeFunc1_PU(n) * intPUShapeFunction1.col(m);

                                // 填充结果矩阵
                                locIntBdryGpsValue2D0_PU_top.col(n * PUDim + m) = temp0;
                                locIntBdryGpsValue2D1_PU_top.col(n * PUDim + m) = temp1;
                            }
                        }
                    }
                }
            }

            MatrixXd locBdryGpsValue2D_DPU_Minus_Interpolation_top;
            MatrixXd locIntBdryGpsValue2D0_DPU_Minus_Interpolation_top;
            MatrixXd locIntBdryGpsValue2D1_DPU_Minus_Interpolation_top;

            if (EnrichMode != 0)
            {
                int pos;
                enrich_flag = Ismember_single(elemID, echFaceNodeFreq, 0, pos);
                if (enrich_flag) {
                    vector<int> vertexNodeIds = { echFaceNodeFreq[pos][5],echFaceNodeFreq[pos][6],echFaceNodeFreq[pos][7],echFaceNodeFreq[pos][8] };
                    vector<size_t> nzIds = Find(vertexNodeIds);     //从零开始 无须加减一
                    vector<vector<int>> faceIds = ElemMap.getOneRingFaces(elemID);
                    FaceIds.clear();
                    for (size_t col = 0; col < faceIds[0].size(); ++col) {  // 先遍历列
                        for (size_t row = 0; row < faceIds.size(); ++row) {  // 再遍历行
                            if (col < faceIds[row].size()) {  // 防止列数不一致
                                FaceIds.push_back(faceIds[row][col]);
                            }
                        }
                    }
                    vector<int> locPUFuncs;
                    vector<bool> Flag;
                    vector<int> EchFuncId;
                    ismember(FaceIds, IntElem, Flag, EchFuncId);
                    vector<size_t> fnzIds = Find(EchFuncId);
                    for (size_t k = 0; k < fnzIds.size(); ++k) {
                        echLocalPUFunctions[pos].push_back(EchFuncId[fnzIds[k]]);
                        locPUFuncs.push_back(EchFuncId[fnzIds[k]] - 1);
                    }
                    vector<int> node_vector;
                    vector<int> echVerNodeFreq_col;
                    for (int col_idx : nzIds) {
                        if (col_idx >= 0 && col_idx < echFaceNodeFreq[pos].size()) {
                            node_vector.push_back(echFaceNodeFreq[pos][col_idx + 1]);
                        }
                    }
                    for (const auto& pair : echVerNodeFreq) {
                        echVerNodeFreq_col.push_back(pair.first);
                    }
                    vector<bool> Flag2;
                    vector<int> EchFuncId2;
                    ismember(node_vector, echVerNodeFreq_col, Flag2, EchFuncId2);  //需要减一
                    vector<double> vFreq;
                    for (int col_idx : EchFuncId2) {
                        double num = echVerNodeFreq[col_idx - 1].second;
                        vFreq.push_back(1 / num);
                    }

                    // 创建vFreq向量
                    Eigen::VectorXd vFreqVec(vFreq.size());

                    // 使用 for 循环逐元素赋值
                    for (size_t i = 0; i < vFreq.size(); ++i) {
                        vFreqVec(i) = vFreq[i];  // 注意：Eigen 使用 `(i)` 而不是 `[i]` 访问元素
                    }


                    // 提取PU_Group的子矩阵（fnzIds列）
                    Eigen::MatrixXd PU_Group_selected = PU_Group(nzIds, fnzIds);

                    // 计算 vFreq .* PU_Group(nzIds, fnzIds)
                    // 由于 vFreq 和 PU_Group_selected 的列数可能不同，需要确保维度匹配
                    // 假设 vFreq.size() == PU_Group_selected.cols()（即 nzIds 和 fnzIds 长度相同）
                    Eigen::MatrixXd term1(PU_Group_selected.rows(), PU_Group_selected.cols());
                    for (int j = 0; j < PU_Group_selected.rows(); ++j) {
                        term1.row(j) = vFreqVec(j) * PU_Group_selected.row(j);
                    }

                    // 计算最终结果
                    int N1 = enrichMutiplyN;
                    int M1 = locPUFuncs.size();      // 列向量长度
                    int K1 = fnzIds.size();          // 列向量长度

                    // 1. locEnrFuncs 矩阵： N1 × M1（列优先）
                    std::vector<int> locEnrMat(N1 * M1);
                    for (int r = 0; r < N1; ++r)           // 行
                        for (int c = 0; c < M1; ++c)       // 列
                            locEnrMat[r + c * N1] = locPUFuncs[c] * N1;

                    // 2. offset 矩阵： K1 × N1（列优先）
                    std::vector<int> offsetMat(K1 * N1);
                    for (int r = 0; r < N1; ++r)
                        for (int c = 0; c < K1; ++c)
                            offsetMat[c * N1 + r] = r + 1;


                    // 3. 逐元素相加（MATLAB 会广播到同长度）

                    locEnrFuncs.reserve(std::max(N1 * M1, K1 * N1));
                    int ML = std::min(N1 * M1, K1 * N1);
                    for (size_t i = 0; i < std::min(N1 * M1, K1 * N1); ++i)
                        locEnrFuncs.push_back(locEnrMat[i] + offsetMat[i]);

                    // 4. 加全局偏移
                    for (int& v : locEnrFuncs) v += globalFEMFunctionN;
                    locBdryGpsValue2D_DPU_Minus_Interpolation_top = MatrixXd::Zero(globalBdryGps.rows(), locEnrFuncs.size());
                    locIntBdryGpsValue2D0_DPU_Minus_Interpolation_top = MatrixXd::Zero(bdryGps0_loc.rows(), locEnrFuncs.size());
                    locIntBdryGpsValue2D1_DPU_Minus_Interpolation_top = MatrixXd::Zero(bdryGps1_loc.rows(), locEnrFuncs.size());

                    if (intBoundaryflag)
                    {
                        MatrixXd localPUvalues0_top, localPUvalues1_top;
                        localPUvalues0_top = locIntBdryGpsValue2D0_PU_top(Eigen::all, nzIds) * term1;
                        localPUvalues1_top = locIntBdryGpsValue2D1_PU_top(Eigen::all, nzIds) * term1;

                        for (int k = 0; k < fnzIds.size(); k++)
                        {
                            int fid = fnzIds[k];
                            double x0, y0 = 0;
                            ElemMap.LocalOriginCoordPU(elemID, fid + 1, x0, y0);
                            MatrixXd DMutis0, DMutis1;
                            MatrixXd DMutis_x0, DMutis_x1;
                            MatrixXd DMutis_y0, DMutis_y1;
                            MatrixXd DMutis0_Cof, DMutis1_Cof;
                            if (!Flag_oldxy) {
                                GenerateOffsetDMutis(bdryGps0, globalInterpolationPoints2D, order_p,
                                    x0, y0, 3 * h, 3 * h,
                                    DMutis0, DMutis_x0, DMutis_y0, DMutis0_Cof);
                                GenerateOffsetDMutis(bdryGps1, globalInterpolationPoints2D, order_p,
                                    x0, y0, 3 * h, 3 * h,
                                    DMutis1, DMutis_x1, DMutis_y1, DMutis1_Cof);
                            }
                            else {
                                GenerateOffsetDMutis(bdryGps0, globalInterpolationPoints2D, order_p,
                                    0, 0, 1, 1,
                                    DMutis0, DMutis_x0, DMutis_y0, DMutis0_Cof);
                                GenerateOffsetDMutis(bdryGps1, globalInterpolationPoints2D, order_p,
                                    0, 0, 1, 1,
                                    DMutis1, DMutis_x1, DMutis_y1, DMutis1_Cof);
                            }
                            DMutis0_Interpolation = locIntBdryGpsValue2D0_top * DMutis0_Cof;
                            DMutis0_Minus_Interpolation = DMutis0 - DMutis0_Interpolation;
                            DMutis1_Interpolation = locIntBdryGpsValue2D1_top * DMutis1_Cof;
                            DMutis1_Minus_Interpolation = DMutis1 - DMutis1_Interpolation;
                            int start_col = (k)*enrichMutiplyN;
                            scalar_vector0 = localPUvalues0_top.col(k);
                            scaled_matrix_0 = DMutis0_Minus_Interpolation.cwiseProduct(scalar_vector0.replicate(1, DMutis0_Minus_Interpolation.cols()));
                            scalar_vector1 = localPUvalues1_top.col(k);
                            scaled_matrix_1 = DMutis1_Minus_Interpolation.cwiseProduct(scalar_vector1.replicate(1, DMutis1_Minus_Interpolation.cols()));
                            locIntBdryGpsValue2D0_DPU_Minus_Interpolation_top.block(0, start_col,
                                locIntBdryGpsValue2D0_DPU_Minus_Interpolation_top.rows(), enrichMutiplyN) = scaled_matrix_0;
                            locIntBdryGpsValue2D1_DPU_Minus_Interpolation_top.block(0, start_col,
                                locIntBdryGpsValue2D1_DPU_Minus_Interpolation_top.rows(), enrichMutiplyN) = scaled_matrix_1;
                        }
                    }
                    else {
                        MatrixXd localPUvalues_top;
                        localPUvalues_top = locBdryGpsValue2D_PU_top(Eigen::all, nzIds) * term1;

                        for (int k = 0; k < fnzIds.size(); k++)
                        {
                            int fid = fnzIds[k];
                            double x0, y0 = 0;
                            ElemMap.LocalOriginCoordPU(elemID, fid + 1, x0, y0);
                            MatrixXd DMutis;
                            MatrixXd DMutis_x;
                            MatrixXd DMutis_y;
                            MatrixXd DMutis_Cof;
                            if (!Flag_oldxy) {
                                GenerateOffsetDMutis(globalBdryGps, globalInterpolationPoints2D, order_p,
                                    x0, y0, 3 * h, 3 * h,
                                    DMutis, DMutis_x, DMutis_y, DMutis_Cof);
                            }
                            else {
                                GenerateOffsetDMutis(globalBdryGps, globalInterpolationPoints2D, order_p,
                                    0, 0, 1, 1,
                                    DMutis, DMutis_x, DMutis_y, DMutis_Cof);
                            }
                            MatrixXd DMutis_Interpolation = locBdryGpsValue2D_top * DMutis_Cof;
                            MatrixXd DMutis_Minus_Interpolation = DMutis - DMutis_Interpolation;
                            int start_col = (k)*enrichMutiplyN;
                            VectorXd scalar_vector = localPUvalues_top.col(k);
                            MatrixXd scaled_matrix = DMutis_Minus_Interpolation.cwiseProduct(scalar_vector.replicate(1, DMutis_Minus_Interpolation.cols()));
                            locBdryGpsValue2D_DPU_Minus_Interpolation_top.block(0, start_col,
                                locBdryGpsValue2D_DPU_Minus_Interpolation_top.rows(), enrichMutiplyN) = scaled_matrix;
                        }
                    }
                }
            }

            if (EnrichMode == 1 && enrich_flag && OrthogonalMode)
            {
                for (int fid : FaceIds)
                {
                    auto result = ismember_single(fid, IntElem);
                    bool flagInt = result.first;
                    int posInt = result.second;

                    if (flagInt)
                    {
                        vector<int> enrfuns(enrichMutiplyN);
                        for (int i = 0; i < enrichMutiplyN; ++i) {
                            enrfuns[i] = globalFEMFunctionN + (posInt)*enrichMutiplyN + (i + 1);
                        }

                        vector<bool> Flag;
                        vector<int> posF;
                        ismember(enrfuns, locEnrFuncs, Flag, posF);
                        for (int m = 0; m < posF.size(); m++)
                        {
                            posF[m] -= 1;      // 减一变成从零开始
                        }
                        MatrixXd cofL = OrthogonalMatrix[posInt][0];
                        if (!intBoundaryflag)
                        {
                            MatrixXd tempvalues = locBdryGpsValue2D_DPU_Minus_Interpolation_top(Eigen::all, posF);
                            for (int k = 1; k < enrichMutiplyN; ++k) {
                                for (int kk = 0; kk < k; ++kk) {
                                    tempvalues.col(k) -= cofL(k, kk) * tempvalues.col(kk);
                                }
                            }
                            for (int k = 0; k < enrichMutiplyN; ++k) {
                                tempvalues.col(k) *= cofL(k, k);
                            }
                            locBdryGpsValue2D_DPU_Minus_Interpolation_top(Eigen::all, posF) = tempvalues;
                        }
                        else
                        {
                            MatrixXd tempvalues0 = locIntBdryGpsValue2D0_DPU_Minus_Interpolation_top(Eigen::all, posF);
                            MatrixXd tempvalues1 = locIntBdryGpsValue2D1_DPU_Minus_Interpolation_top(Eigen::all, posF);
                            for (int k = 1; k < enrichMutiplyN; ++k) {
                                for (int kk = 0; kk < k; ++kk) {
                                    tempvalues0.col(k) -= cofL(k, kk) * tempvalues0.col(kk);
                                    tempvalues1.col(k) -= cofL(k, kk) * tempvalues1.col(kk);
                                }
                            }
                            for (int k = 0; k < enrichMutiplyN; ++k) {
                                tempvalues0.col(k) *= cofL(k, k);
                                tempvalues1.col(k) *= cofL(k, k);
                            }
                            locIntBdryGpsValue2D0_DPU_Minus_Interpolation_top(Eigen::all, posF) = tempvalues0;
                            locIntBdryGpsValue2D1_DPU_Minus_Interpolation_top(Eigen::all, posF) = tempvalues1;
                        }
                    }
                }
            }

            locG;

            if (!intBoundaryflag) {
                int n = globalBdryGps.rows();
                VectorXd val_ux(n), val_uy(n);

                for (int i = 0; i < n; ++i) {
                    val_ux(i) = calc_exact_ux(globalBdryGps(i, 0), globalBdryGps(i, 1));
                    val_uy(i) = calc_exact_uy(globalBdryGps(i, 0), globalBdryGps(i, 1));
                }

                VectorXd g = val_ux * normal(0) + val_uy * normal(1);
                VectorXd weighted_g = g.array() * locBdryKappa.array() * localGpWs.array();
                
                // 方案二：直接计算，完全避免中间矩阵
                if (locBdryGpsValue2D_DPU_Minus_Interpolation_top.cols() == 0) {
                    // 无富集情况，直接计算
                    locG = locBdryGpsValue2D_top.transpose() * weighted_g * gJac;
                } else {
                    // 有富集情况，分段计算填入locG
                    int n1 = locBdryGpsValue2D_top.cols();
                    int n2 = locBdryGpsValue2D_DPU_Minus_Interpolation_top.cols();
                    
                    // 分配locG的空间
                    if (locG.size() != n1 + n2) {
                        locG.resize(n1 + n2);
                    }
                    
                    // 分段计算并填入locG
                    locG.segment(0, n1) = locBdryGpsValue2D_top.transpose() * weighted_g * gJac;
                    locG.segment(n1, n2) = locBdryGpsValue2D_DPU_Minus_Interpolation_top.transpose() * weighted_g * gJac;
                }
            }
            else {
                int n0 = bdryGps0.rows();
                int n1 = bdryGps1.rows();

                VectorXd val_ux0(n0), val_uy0(n0);
                for (int i = 0; i < n0; ++i) {
                    val_ux0(i) = calc_exact_ux(bdryGps0(i, 0), bdryGps0(i, 1));
                    val_uy0(i) = calc_exact_uy(bdryGps0(i, 0), bdryGps0(i, 1));
                }
                VectorXd g0 = val_ux0 * normal(0) + val_uy0 * normal(1);

                VectorXd val_ux1(n1), val_uy1(n1);
                for (int i = 0; i < n1; ++i) {
                    val_ux1(i) = calc_exact_ux(bdryGps1(i, 0), bdryGps1(i, 1));
                    val_uy1(i) = calc_exact_uy(bdryGps1(i, 0), bdryGps1(i, 1));
                }
                VectorXd g1 = val_ux1 * normal(0) + val_uy1 * normal(1);

                // 方案二：直接计算，避免中间矩阵combined_matrix0和combined_matrix1
                
                // 计算第一组的贡献
                VectorXd part0;
                if (locIntBdryGpsValue2D0_DPU_Minus_Interpolation_top.cols() == 0) {
                    // 无富集
                    part0 = locIntBdryGpsValue2D0_top.transpose() * (g0.cwiseProduct(bdryGpW0));
                } else {
                    // 有富集，分段计算
                    int n0_1 = locIntBdryGpsValue2D0_top.cols();
                    int n0_2 = locIntBdryGpsValue2D0_DPU_Minus_Interpolation_top.cols();
                    part0.resize(n0_1 + n0_2);
                    part0.segment(0, n0_1) = locIntBdryGpsValue2D0_top.transpose() * (g0.cwiseProduct(bdryGpW0));
                    part0.segment(n0_1, n0_2) = locIntBdryGpsValue2D0_DPU_Minus_Interpolation_top.transpose() * (g0.cwiseProduct(bdryGpW0));
                }
                
                // 计算第二组的贡献
                VectorXd part1;
                if (locIntBdryGpsValue2D1_DPU_Minus_Interpolation_top.cols() == 0) {
                    // 无富集
                    part1 = locIntBdryGpsValue2D1_top.transpose() * (g1.cwiseProduct(bdryGpW1));
                } else {
                    // 有富集，分段计算
                    int n1_1 = locIntBdryGpsValue2D1_top.cols();
                    int n1_2 = locIntBdryGpsValue2D1_DPU_Minus_Interpolation_top.cols();
                    part1.resize(n1_1 + n1_2);
                    part1.segment(0, n1_1) = locIntBdryGpsValue2D1_top.transpose() * (g1.cwiseProduct(bdryGpW1));
                    part1.segment(n1_1, n1_2) = locIntBdryGpsValue2D1_DPU_Minus_Interpolation_top.transpose() * (g1.cwiseProduct(bdryGpW1));
                }
                
                // 最终结果
                if (locG.size() != part0.size()) {
                    locG.resize(part0.size());
                }
                locG = a0 * part0 + a1 * part1;
            }

            locFuncs.clear();
            if (locEnrFuncs.empty()) {
                locFuncs = locFEMFuncs;
            }
            else {
                locFuncs.reserve(locFEMFuncs.size() + locEnrFuncs.size());
                locFuncs.insert(locFuncs.end(), locFEMFuncs.begin(), locFEMFuncs.end());
                locFuncs.insert(locFuncs.end(), locEnrFuncs.begin(), locEnrFuncs.end());
            }

            assert(locFuncs.size() == locG.size() && "locG size must match locFuncs size");

            for (size_t i = 0; i < locFuncs.size(); ++i) {
                int idx = locFuncs[i];
                assert(idx >= 0 && idx < F.size() && "Index out of bounds");
                F(idx - 1) += locG(i);
            }




        }

        for (int id = 0; id < numIntElem; id++)
        {
            
            MatrixXd origins = ElemMap.LocalOriginCoord(IntElem);
            int elemID = IntElem[id];
            vector<int> locEnrFuncs;
            vector<int> locFEMFuncs = ElemMap.getFEMBasisIds(elemID);
            MatrixXd globalInterpolationPoints2D = ElemMap.localToGlobal(elemID, InterpolationPoints2D);
            VectorXd xcoord = IntElemCofx.row(id);
            VectorXd ycoord = IntElemCofy.row(id);
            auto ij = ElemMap.calcIndex(elemID);
            int i = ij[0]; int j = ij[1];
            double dx = xcoord(1) - xcoord(0);  // C++是0-based索引
            double dy = ycoord(1) - ycoord(0);
            double qJac = sqrt(dx * dx + dy * dy);

            // 计算比例
            double ratio = qJac / h;

            // 判断是否继续
            if (ratio < 0.000001) {
                continue;  // 在循环中使用
            }

            VectorXd origin = origins.row(id);
            MatrixXd intGps(localGps.rows(), 2);
            MatrixXd intGps_loc(localGps.rows(), 2);

            // Calculate intGps
            intGps.col(0) = xcoord(0) * (1 - localGps.array()) + xcoord(1) * localGps.array();
            intGps.col(1) = ycoord(0) * (1 - localGps.array()) + ycoord(1) * localGps.array();

            // Calculate intGps_loc
            intGps_loc = (intGps.rowwise() - origin.transpose()) * N;

            VectorXd val_q(intGps.rows());  // Vector to store results for each point

            for (int i = 0; i < intGps.rows(); ++i) {
                val_q(i) = calc_q(intGps(i, 0), intGps(i, 1), n0, n1);
            }

            MatrixXd locGpValsX = BasisFunction::FEMShapeFunction(intGps_loc.col(0), order_p, 0);
            MatrixXd locGpValsY = BasisFunction::FEMShapeFunction(intGps_loc.col(1), order_p, 0);

            MatrixXd locGpVals2D = MatrixXd::Zero(gpn, (order_p + 1) * (order_p + 1));


            // 双重循环计算
            for (int m = 0; m < order_p + 1; ++m) {
                for (int n = 0; n < order_p + 1; ++n) {
                    // 计算temp0和temp1
                    VectorXd temp = locGpValsX.col(m).cwiseProduct(locGpValsY.col(n));


                    // 填充结果矩阵
                    locGpVals2D.col(m * (order_p + 1) + n) = temp;

                }
            }
            MatrixXd locGpValsX_PU = BasisFunction::PUShapeFunction(intGps_loc.col(0), 0);
            MatrixXd locGpValsY_PU = BasisFunction::PUShapeFunction(intGps_loc.col(1), 0);

            MatrixXd locGpVals2D_PU = MatrixXd::Zero(gpn, PUDim * PUDim);

            for (int m = 0; m < PUDim; ++m) {
                for (int n = 0; n < PUDim; ++n) {
                    // 计算 temp0 和 temp1
                    VectorXd temp = locGpValsX_PU.col(m).cwiseProduct(locGpValsY_PU.col(n));

                    // 填充结果矩阵
                    locGpVals2D_PU.col(m * PUDim + n) = temp;

                }
            }

            bool enrich_flag = false;
            MatrixXd locGpVals2D_DPU_Minus_Interpolation;
            vector<int> FaceIds;
            if (EnrichMode != 0)
            {
                int pos;
                enrich_flag = Ismember_single(elemID, echFaceNodeFreq, 0, pos);
                if (enrich_flag) {
                    vector<int> vertexNodeIds = { echFaceNodeFreq[pos][5],echFaceNodeFreq[pos][6],echFaceNodeFreq[pos][7],echFaceNodeFreq[pos][8] };
                    vector<size_t> nzIds = Find(vertexNodeIds);     //从零开始 无须加减一
                    vector<vector<int>> faceIds = ElemMap.getOneRingFaces(elemID);
                    FaceIds.clear();
                    for (size_t col = 0; col < faceIds[0].size(); ++col) {  // 先遍历列
                        for (size_t row = 0; row < faceIds.size(); ++row) {  // 再遍历行
                            if (col < faceIds[row].size()) {  // 防止列数不一致
                                FaceIds.push_back(faceIds[row][col]);
                            }
                        }
                    }
                    vector<int> locPUFuncs;
                    vector<bool> Flag;
                    vector<int> EchFuncId;
                    ismember(FaceIds, IntElem, Flag, EchFuncId);
                    vector<size_t> fnzIds = Find(EchFuncId);
                    for (size_t k = 0; k < fnzIds.size(); ++k) {
                        echLocalPUFunctions[pos].push_back(EchFuncId[fnzIds[k]]);
                        locPUFuncs.push_back(EchFuncId[fnzIds[k]] - 1);
                    }
                    vector<int> node_vector;
                    vector<int> echVerNodeFreq_col;
                    for (int col_idx : nzIds) {
                        if (col_idx >= 0 && col_idx < echFaceNodeFreq[pos].size()) {
                            node_vector.push_back(echFaceNodeFreq[pos][col_idx + 1]);
                        }
                    }
                    for (const auto& pair : echVerNodeFreq) {
                        echVerNodeFreq_col.push_back(pair.first);
                    }
                    vector<bool> Flag2;
                    vector<int> EchFuncId2;
                    ismember(node_vector, echVerNodeFreq_col, Flag2, EchFuncId2);  //需要减一
                    vector<double> vFreq;
                    for (int col_idx : EchFuncId2) {
                        double num = echVerNodeFreq[col_idx - 1].second;
                        vFreq.push_back(1 / num);
                    }

                    // 创建vFreq向量
                    Eigen::VectorXd vFreqVec(vFreq.size());

                    // 使用 for 循环逐元素赋值
                    for (size_t i = 0; i < vFreq.size(); ++i) {
                        vFreqVec(i) = vFreq[i];  // 注意：Eigen 使用 `(i)` 而不是 `[i]` 访问元素
                    }

                    Eigen::MatrixXd PU_Group_selected = PU_Group(nzIds, fnzIds);
                    Eigen::MatrixXd term1(PU_Group_selected.rows(), PU_Group_selected.cols());

                    Eigen::MatrixXd localGpSelected = localGpValues2D_PU(Eigen::all, nzIds);
                    Eigen::MatrixXd localGpSelected_s = localGpValues2D_PU_s(Eigen::all, nzIds);
                    Eigen::MatrixXd localGpSelected_t = localGpValues2D_PU_t(Eigen::all, nzIds);
                    Eigen::MatrixXd locGpVals2D_PUSelected = locGpVals2D_PU(Eigen::all, nzIds);



                    // 创建vFreq向量
                    for (int j = 0; j < PU_Group_selected.rows(); ++j) {
                        term1.row(j) = vFreqVec(j) * PU_Group_selected.row(j);
                    }

                    // 计算最终结果
                    int N1 = enrichMutiplyN;
                    int M1 = locPUFuncs.size();      // 列向量长度
                    int K1 = fnzIds.size();          // 列向量长度

                    // 1. locEnrFuncs 矩阵： N1 × M1（列优先）
                    std::vector<int> locEnrMat(N1 * M1);
                    for (int r = 0; r < N1; ++r)           // 行
                        for (int c = 0; c < M1; ++c)       // 列
                            locEnrMat[r + c * N1] = locPUFuncs[c] * N1;

                    // 2. offset 矩阵： K1 × N1（列优先）
                    std::vector<int> offsetMat(K1 * N1);
                    for (int r = 0; r < N1; ++r)
                        for (int c = 0; c < K1; ++c)
                            offsetMat[c * N1 + r] = r + 1;


                    // 3. 逐元素相加（MATLAB 会广播到同长度）

                    locEnrFuncs.reserve(std::max(N1 * M1, K1 * N1));
                    int ML = std::min(N1 * M1, K1 * N1);
                    for (size_t i = 0; i < std::min(N1 * M1, K1 * N1); ++i)
                        locEnrFuncs.push_back(locEnrMat[i] + offsetMat[i]);

                    // 4. 加全局偏移
                    for (int& v : locEnrFuncs) v += globalFEMFunctionN;

                    locGpVals2D_DPU_Minus_Interpolation = MatrixXd::Zero(intGps.rows(), locEnrFuncs.size());

                    for (int k = 0; k < fnzIds.size(); k++)
                    {
                        int fid = fnzIds[k];
                        double x0, y0 = 0;
                        ElemMap.LocalOriginCoordPU(elemID, fid + 1, x0, y0);
                        MatrixXd DMutis;
                        MatrixXd DMutis_x;
                        MatrixXd DMutis_y;
                        MatrixXd DMutis_Cof;
                        MatrixXd localPUvalues = locGpVals2D_PU(Eigen::all, nzIds) * term1;
                        if (!Flag_oldxy) {
                            GenerateOffsetDMutis(intGps, globalInterpolationPoints2D, order_p,
                                x0, y0, 3 * h, 3 * h,
                                DMutis, DMutis_x, DMutis_y, DMutis_Cof);
                        }
                        else {
                            GenerateOffsetDMutis(intGps, globalInterpolationPoints2D, order_p,
                                0, 0, 1, 1,
                                DMutis, DMutis_x, DMutis_y, DMutis_Cof);
                        }
                        MatrixXd DMutis_Interpolation = locGpVals2D * DMutis_Cof;
                        MatrixXd DMutis_Minus_Interpolation = DMutis - DMutis_Interpolation;
                        int start_col = (k)*enrichMutiplyN;
                        VectorXd scalar_vector = localPUvalues.col(k);
                        MatrixXd scaled_matrix = DMutis_Minus_Interpolation.cwiseProduct(scalar_vector.replicate(1, DMutis_Minus_Interpolation.cols()));
                        locGpVals2D_DPU_Minus_Interpolation.block(0, start_col,
                            locGpVals2D_DPU_Minus_Interpolation.rows(), enrichMutiplyN) = scaled_matrix;
                    }
                }

            }
            VectorXd locQ;
            if (EnrichMode == 1 && enrich_flag && OrthogonalMode)
            {
                for (int fid : FaceIds)
                {
                    auto result = ismember_single(fid, IntElem);
                    bool flagInt = result.first;
                    int posInt = result.second;

                    if (flagInt)
                    {
                        vector<int> enrfuns(enrichMutiplyN);
                        for (int i = 0; i < enrichMutiplyN; ++i) {
                            enrfuns[i] = globalFEMFunctionN + (posInt)*enrichMutiplyN + (i + 1);
                        }

                        vector<bool> Flag;
                        vector<int> posF;
                        ismember(enrfuns, locEnrFuncs, Flag, posF);
                        for (int m = 0; m < posF.size(); m++)
                        {
                            posF[m] -= 1;      // 减一变成从零开始
                        }
                        MatrixXd cofL = OrthogonalMatrix[posInt][0];
                        MatrixXd tempvalues = locGpVals2D_DPU_Minus_Interpolation(Eigen::all, posF);
                        for (int k = 1; k < enrichMutiplyN; ++k) {
                            for (int kk = 0; kk < k; ++kk) {
                                tempvalues.col(k) -= cofL(k, kk) * tempvalues.col(kk);
                            }
                        }
                        for (int k = 0; k < enrichMutiplyN; ++k) {
                            tempvalues.col(k) *= cofL(k, k);
                        }
                        locGpVals2D_DPU_Minus_Interpolation(Eigen::all, posF) = tempvalues;




                    }
                }
            }
            MatrixXd combined_matrix;
            if (locGpVals2D_DPU_Minus_Interpolation.cols() == 0) {
                combined_matrix = locGpVals2D;
            }
            else {
                combined_matrix.resize(locGpVals2D.rows(),
                    locGpVals2D.cols() +
                    locGpVals2D_DPU_Minus_Interpolation.cols());
                combined_matrix << locGpVals2D, locGpVals2D_DPU_Minus_Interpolation;
            }

            VectorXd g = val_q.array() * localGpWs.array();
            locQ = combined_matrix.transpose() * (g)*qJac;
            vector<int> locFuncs;
            locFuncs.reserve(locFEMFuncs.size() + locEnrFuncs.size());
            locFuncs.insert(locFuncs.end(), locFEMFuncs.begin(), locFEMFuncs.end());
            locFuncs.insert(locFuncs.end(), locEnrFuncs.begin(), locEnrFuncs.end());

            for (size_t i = 0; i < locFuncs.size(); ++i) {
                int idx = locFuncs[i] - 1;
                assert(idx >= 0 && idx < F.size() && "Index out of bounds");
                F(idx) += locQ(i);
            }

        }
        // 使用三元组列表批量构建稀疏矩阵，避免O(n^2)的逐个插入
        K.setFromTriplets(tripletList.begin(), tripletList.end());
        tripletList.clear(); // 清空三元组列表释放内存
        
        clock_t start1 = clock();
        
        // 1. 压缩矩阵（只需一次）
        K.makeCompressed();

        

        // 安全检查：确保矩阵和向量维度匹配
        if (K.rows() != K.cols() || K.rows() != F.size()) {
            std::cerr << "Error: Dimension mismatch!" << std::endl;
            std::cerr << "K rows: " << K.rows() << ", K cols: " << K.cols() 
                      << ", F size: " << F.size() << std::endl;
            return 1;  // 或者抛出异常
        }

        // 2. 第一次对角缩放（优化：原地修改，避免稀疏矩阵乘法）
        VectorXd K_diag_vec(totalFunctionN + 1);
        #pragma omp parallel for if(totalFunctionN > 5000)
        for (int i = 0; i < totalFunctionN; ++i) {
            double diag_val = K.coeff(i, i);
            if (diag_val <= 0.0) {
                std::cerr << "Warning: Non-positive diagonal element at " << i 
                          << ": " << diag_val << std::endl;
                K_diag_vec(i) = 1.0;
            } else {
                K_diag_vec(i) = 1.0 / sqrt(diag_val);
            }
        }
        K_diag_vec(totalFunctionN) = 1.0;

        symmetricDiagonalScale(K, K_diag_vec);
        F = K_diag_vec.asDiagonal() * F;  // 缩放右端项

        // 3. 删除未使用的稠密子矩阵提取（节省大量时间）
        // 如果确实需要分析富集函数部分，建议使用稀疏格式
        /*
        int ech_size = totalFunctionN - globalFEMFunctionN;
        if (ech_size > 0 && ech_size < 5000) {  // 只在小规模时提取
            // 提取稀疏子矩阵而不是稠密矩阵
            // ...
        }
        */

        // 4. 第二次对角缩放（优化：同样原地修改）
        VectorXd K_diag2_vec(totalFunctionN + 1);
        #pragma omp parallel for if(totalFunctionN > 5000)
        for (int i = 0; i < totalFunctionN; ++i) {
            double diag_val = K.coeff(i, i);
            if (diag_val <= 0.0) {
                std::cerr << "Warning: Non-positive diagonal element at " << i 
                          << " in second scaling: " << diag_val << std::endl;
                K_diag2_vec(i) = 1.0;
            } else {
                K_diag2_vec(i) = 1.0 / sqrt(diag_val);
            }
        }
        K_diag2_vec(totalFunctionN) = 1.0;

        symmetricDiagonalScale(K, K_diag2_vec);
        F = K_diag2_vec.asDiagonal() * F;

        // 5. 优化求解策略（删除未使用的SparseLU）
        VectorXd u_cof;

        // 根据问题规模自动选择求解器
        if (totalFunctionN < 1000000) {
            // 中小规模：使用直接法（精确解）
            SimplicialLDLT<SparseMatrix<double>> ldlt_solver;
            ldlt_solver.compute(K);
            
            
            if (ldlt_solver.info() == Success) {
                u_cof = ldlt_solver.solve(F);
                double cond_scaled = computeConditionNumber1Norm(K, ldlt_solver, true);
                SCN(mesh_layer) = cond_scaled;
            } else {
                // 如果LDLT失败（矩阵不正定），使用LU分解
                SparseLU<SparseMatrix<double>> lu_solver;
                lu_solver.compute(K);
                u_cof = lu_solver.solve(F);
                double cond_scaled = computeConditionNumber1Norm(K, lu_solver, true);
                SCN(mesh_layer) = cond_scaled;

            }
        } else {
            // 大规模：使用迭代法（省内存）
            // 调试信息：输出矩阵和向量维度
            std::cout << "Matrix K dimensions: " << K.rows() << " x " << K.cols() << std::endl;
            std::cout << "Vector F size: " << F.size() << std::endl;
            std::cout << "Matrix non-zeros: " << K.nonZeros() << std::endl;
            
            try {
                // 使用更稳定的对角预处理器（IncompleteCholesky可能因矩阵不正定而失败）
                ConjugateGradient<SparseMatrix<double>, Lower | Upper, 
                                 DiagonalPreconditioner<double>> cg;
                cg.setTolerance(1e-8);
                cg.setMaxIterations(5000);
                
                // 检查compute是否成功
                cg.compute(K);
                
                if (cg.info() != Success) {
                    std::cout << "Warning: CG compute failed with info: " << cg.info() << std::endl;
                    std::cout << "Falling back to SparseLU solver" << std::endl;
                    
                    // 使用SparseLU作为fallback
                    SparseLU<SparseMatrix<double>> lu_solver;
                    lu_solver.compute(K);
                    u_cof = lu_solver.solve(F);
                } else {
                    u_cof = cg.solve(F);
                    
                    if (cg.info() != Success) {
                        std::cout << "Warning: CG solve failed with info: " << cg.info() << std::endl;
                    }
                    
                    std::cout << "CG iterations: " << cg.iterations() << ", error: " 
                              << cg.error() << std::endl;
                    
                    // 计算CG求解后的1范数条件数（利用已分解的K矩阵）
                    {
                        // CG的内部compute()已经分解了K矩阵，但不能直接获取
                        // 额外做一次LDLT分解用于条件数估计
                        SimplicialLDLT<SparseMatrix<double>> cg_cond_solver;
                        cg_cond_solver.compute(K);
                        
                        if (cg_cond_solver.info() == Success) {
                            std::cout << "\n  [CG分支] 1范数条件数估计:" << std::endl;
                            double cond_cg = computeConditionNumber1Norm(K, cg_cond_solver, true);
                            if (cond_cg > 0) {
                                std::cout << "  CG求解 κ₁(K) = " << std::scientific << cond_cg << std::endl;
                            }
                        }
                    }
                }
            } catch (const std::exception& e) {
                std::cout << "Exception in CG solver: " << e.what() << std::endl;
                std::cout << "Using SparseLU as emergency fallback" << std::endl;
                
                // 紧急fallback到SparseLU
                SparseLU<SparseMatrix<double>> lu_solver;
                lu_solver.compute(K);
                u_cof = lu_solver.solve(F);
            }
        }

        // 6. 计算缩放后K矩阵的1范数条件数（在逆缩放之前）
        

        // 7. 逆缩放（注意顺序）
        u_cof = K_diag2_vec.asDiagonal() * u_cof;
        u_cof = K_diag_vec.asDiagonal() * u_cof;

        clock_t end1 = clock();
        double ldl_time_used = static_cast<double>(end1 - start1) / CLOCKS_PER_SEC;
		ldltime(mesh_layer) = ldl_time_used;

        std::cout << "ldl time used: " << ldl_time_used << " s \n";
        
        // 注意：缩放后K矩阵的条件数已在前面计算（第3779-3822行）
        // 该条件数反映了两次对角缩放对矩阵病态程度的改善效果

        int intElemIter = 0;
        double sum_u = 0;
        double sum_uh = 0;

        for (int i = 0;i < N;i++)
        {
            for (int j = 0;j < N;j++)
            {
                if (inter[i][j] != 1)
                {
                    
                    int elemID = ElemMap.getElemId(i, j);
                    MatrixXd globalGps = ElemMap.localToGlobal(elemID, localGps2D);
                    for (int i = 0; i < globalGps.rows(); ++i)
                    {
                        double x = globalGps(i, 0);
                        double y = globalGps(i, 1);
                        sum_u += calc_exact_u(x, y) * localGpW2D(i) * h * h;
                    }
                    // 乘以 h²
                }
                else
                {
                    MatrixXd gps0 = intElemGps[intElemIter][0];
                    MatrixXd gps1 = intElemGps[intElemIter][1];
                    for (int i = 0; i < gps0.rows(); ++i) {
                        double x = gps0(i, 0);
                        double y = gps0(i, 1);
                        double weight_product = gps0(i, 2) * gps0(i, 3);  // 第3列和第4列逐元素相乘
                        sum_u += calc_exact_u(x, y) * weight_product;
                    }

                    // 计算第二部分：gps1的贡献
                    for (int i = 0; i < gps1.rows(); ++i) {
                        double x = gps1(i, 0);
                        double y = gps1(i, 1);
                        double weight_product = gps1(i, 2) * gps1(i, 3);  // 第3列和第4列逐元素相乘
                        sum_u += calc_exact_u(x, y) * weight_product;
                    }

                    intElemIter = intElemIter + 1;
                }

            }
        }

        intElemIter = 0;
        int echElemIter = 0;
        int projElemIter = 1;

        for (int i = 0;i < N;i++)
        {
            for (int j = 0;j < N;j++)
            {
                
                int elemID = ElemMap.getElemId(i, j);
                vector<int> locFEMFuncs = ElemMap.getFEMBasisIds(elemID);
                for (int& num : locFEMFuncs) {
                    num -= 1;
                }
                MatrixXd gps;
                if (inter[i][j] != 1)
                {
                    VectorXd temp = localGpValues2D * u_cof(locFEMFuncs);
                    sum_uh = sum_uh + temp.cwiseProduct(localGpW2D).sum() * h * h;
                }
                else
                {
                    gps.resize(intElemGps[intElemIter][0].rows() + intElemGps[intElemIter][1].rows(),
                        intElemGps[intElemIter][0].cols());

                    gps << intElemGps[intElemIter][0],
                        intElemGps[intElemIter][1];

                    VectorXd temp1 = gps.col(2).cwiseProduct(gps.col(3));
                    double temp = (intElemGpsVal[intElemIter][0] * u_cof(locFEMFuncs)).transpose() * temp1;

                    sum_uh += temp;
                    intElemIter = intElemIter + 1;
                }
                if (EnrichMode == 1 && elemID == echFaceNodeFreq[echElemIter][0])
                {
                    vector<int> locEnrFuncs = echLocalDPUFunctions[echElemIter];
                    for (int& num : locEnrFuncs) {
                        num -= 1;
                    }
                    if (inter[i][j] != 1)
                    {
                        VectorXd temp = echElemGpsValues[echElemIter][0] * u_cof(locEnrFuncs);
                        sum_uh = sum_uh + temp.cwiseProduct(localGpW2D).sum() * h * h;
                    }
                    else
                    {

                        VectorXd temp1 = gps.col(2).cwiseProduct(gps.col(3));
                        double temp = (echElemGpsValues[echElemIter][0] * u_cof(locEnrFuncs)).transpose() * temp1;

                        sum_uh += temp;


                    }
                    if (echElemIter != (echFaceNodeFreq.size() - 1))
                    {
                        echElemIter = echElemIter + 1;
                    }
                }
            }
        }

        intElemIter = 0;
        echElemIter = 0;
        projElemIter = 1;

        for (int i = 0;i < N;i++)
        {
            for (int j = 0; j < N;j++)
            {
                
                int elemID = ElemMap.getElemId(i, j);
                vector<int> locFEMFuncs = ElemMap.getFEMBasisIds(elemID);
                for (int& num : locFEMFuncs) {
                    num -= 1;
                }
                MatrixXd gps;
                if (inter[i][j] != 1)
                {
                    MatrixXd globalGps = ElemMap.localToGlobal(elemID, localGps2D);
                    VectorXd u_val, ux_val, uy_val;
                    int n = globalGps.rows();
                    u_val.resize(n);
                    ux_val.resize(n);
                    uy_val.resize(n);

                    // 合并循环（更高效）
                    for (int i = 0; i < n; ++i)
                    {
                        double x = globalGps(i, 0);
                        double y = globalGps(i, 1);

                        u_val(i) = calc_exact_u(x, y) - sum_u;  // 需要定义 sum_u
                        ux_val(i) = calc_exact_ux(x, y);
                        uy_val(i) = calc_exact_uy(x, y);
                    }

                    VectorXd uh_val = localGpValues2D * u_cof(locFEMFuncs);
                    VectorXd uhx_val = localGpValues2D_s * u_cof(locFEMFuncs) * N;
                    VectorXd uhy_val = localGpValues2D_t * u_cof(locFEMFuncs) * N;
                    if (EnrichMode == 1 && elemID == echFaceNodeFreq[echElemIter][0])
                    {
                        vector<int> locEnrFuncs = echLocalDPUFunctions[echElemIter];
                        for (int& num : locEnrFuncs) {
                            num -= 1;
                        }
                        uh_val = uh_val + echElemGpsValues[echElemIter][0] * u_cof(locEnrFuncs);
                        uhx_val = uhx_val + echElemGpsValues[echElemIter][1] * u_cof(locEnrFuncs);
                        uhy_val = uhy_val + echElemGpsValues[echElemIter][2] * u_cof(locEnrFuncs);
                        if (echElemIter != (echFaceNodeFreq.size() - 1))
                        {
                            echElemIter = echElemIter + 1;
                        }

                    }
                    double locKappa = a0;
                    if (partion[i][j] == 1)
                    {
                        locKappa = a1;
                    }
                    VectorXd temp = (u_val - uh_val).array().square().transpose();
                    L2_error(mesh_layer) = L2_error(mesh_layer) + temp.cwiseProduct(localGpW2D).sum() * h * h;
                    VectorXd temp1 = (ux_val - uhx_val).array().square().transpose() + (uy_val - uhy_val).array().square().transpose();
                    H1_error(mesh_layer) = H1_error(mesh_layer) + temp1.cwiseProduct(localGpW2D).sum() * h * h * locKappa;




                }
                else
                {
                    gps.resize(intElemGps[intElemIter][0].rows() + intElemGps[intElemIter][1].rows(),
                        intElemGps[intElemIter][0].cols());
                    gps << intElemGps[intElemIter][0],
                        intElemGps[intElemIter][1];
                    VectorXd u_val, ux_val, uy_val;
                    int n = gps.rows();
                    u_val.resize(n);
                    ux_val.resize(n);
                    uy_val.resize(n);

                    // 合并循环（更高效）
                    for (int i = 0; i < n; ++i)
                    {
                        double x = gps(i, 0);
                        double y = gps(i, 1);

                        u_val(i) = calc_exact_u(x, y) - sum_u;  // 需要定义 sum_u
                        ux_val(i) = calc_exact_ux(x, y);
                        uy_val(i) = calc_exact_uy(x, y);
                    }

                    VectorXd uh_val = intElemGpsVal[intElemIter][0] * u_cof(locFEMFuncs);
                    VectorXd uhx_val = intElemGpsVal[intElemIter][1] * u_cof(locFEMFuncs) * N;
                    VectorXd uhy_val = intElemGpsVal[intElemIter][2] * u_cof(locFEMFuncs) * N;

                    if (EnrichMode == 1 && elemID == echFaceNodeFreq[echElemIter][0])
                    {
                        vector<int> locEnrFuncs = echLocalDPUFunctions[echElemIter];
                        for (int& num : locEnrFuncs) {
                            num -= 1;
                        }
                        uh_val = uh_val + echElemGpsValues[echElemIter][0] * u_cof(locEnrFuncs);
                        uhx_val = uhx_val + echElemGpsValues[echElemIter][1] * u_cof(locEnrFuncs);
                        uhy_val = uhy_val + echElemGpsValues[echElemIter][2] * u_cof(locEnrFuncs);
                        if (echElemIter != (echFaceNodeFreq.size() - 1))
                        {
                            echElemIter = echElemIter + 1;
                        }

                    }
                    int rows1 = intElemGps[intElemIter][0].rows();
                    // 创建 a0 * ones(rows1, 1)
                    VectorXd part0 = VectorXd::Constant(rows1, a0);

                    // 获取 intElemGps[intElemIter][1] 的行数
                    int rows2 = intElemGps[intElemIter][1].rows();
                    // 创建 a1 * ones(rows2, 1)
                    VectorXd part1 = VectorXd::Constant(rows2, a1);

                    // 垂直拼接 part0 和 part1
                    VectorXd locKappa(part0.size() + part1.size());
                    locKappa << part0, part1;
                    VectorXd temp = (u_val - uh_val).array().square().transpose();
                    VectorXd gps_product = gps.col(2).array() * gps.col(3).array();

                    L2_error(mesh_layer) = L2_error(mesh_layer) + temp.cwiseProduct(gps_product).sum();
                    VectorXd temp1 = (ux_val - uhx_val).array().square().transpose() + (uy_val - uhy_val).array().square().transpose();
                    VectorXd temp2 = temp1.array() * gps_product.array();
                    H1_error(mesh_layer) = H1_error(mesh_layer) + (temp2).cwiseProduct(locKappa).sum();
                    intElemIter = intElemIter + 1;

                }
            }
        }

        L2_error(mesh_layer) = sqrt(L2_error(mesh_layer));
        H1_error(mesh_layer) = sqrt(H1_error(mesh_layer));



        // Your code here  
        // ...  

        clock_t end = clock();
        double cpu_time_used = static_cast<double>(end - start) / CLOCKS_PER_SEC;
		totaltime(mesh_layer) = cpu_time_used;
        //  /CLOCKS_PER_SEC将结果转为以秒为单位

        std::cout << "CPU time used: " << cpu_time_used << " s\n";


        

    }
    Eigen::VectorXd order_L2 = ((L2_error.head(L2_error.size() - 1).array() /
        L2_error.tail(L2_error.size() - 1).array()).log()) / std::log(2.0);

    Eigen::VectorXd order_H1 = ((H1_error.head(H1_error.size() - 1).array() /
        H1_error.tail(H1_error.size() - 1).array()).log()) / std::log(2.0);

    // 检查SCN是否已正确计算（避免全0导致的除零错误）
    bool scn_valid = (SCN.array() > 1e-12).any();  // 检查是否存在有效值
    
    Eigen::VectorXd order_SCN;
    if (scn_valid) {
        order_SCN = ((SCN.tail(SCN.size() - 1).array() /
            SCN.head(SCN.size() - 1).array()).log()) / std::log(2.0);
    }
    std::cout << "\n========== 运行时间汇总 ==========" << std::endl;
    std::cout << std::scientific << std::setprecision(6);
	for (int i = 0; i < mesh.size(); ++i) {
		std::cout << "  Mesh " << mesh[i] << ": LDLT time = " << ldltime(i)
			<< " s, Total time =  " << totaltime(i) << " s" << std::endl;
	}
    // 输出三个收敛阶
    std::cout << "\n========== 收敛阶汇总 ==========" << std::endl;
    std::cout << std::scientific << std::setprecision(6);
    std::cout << "  L2  convergence order: " << order_L2.transpose() << std::endl;
    std::cout << "  H1  convergence order: " << order_H1.transpose() << std::endl;
    if (scn_valid) {
        std::cout << "  SCN convergence order: " << order_SCN.transpose() << std::endl;
    } else {
        std::cout << "  SCN convergence order: (未计算)" << std::endl;
    }
    std::cout << "=================================" << std::endl;

    std::cout << "\n========== 误差 ==========" << std::endl;
    std::cout << std::scientific << std::setprecision(6);
    std::cout << "  L2  error: " << L2_error.transpose() << std::endl;
    std::cout << "  H1  error: " << H1_error.transpose() << std::endl;

    return 0;
}
