#include "GaussianQuadrature.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>

using namespace Eigen;

void GaussianQuadrature::ComputeNodesAndWeights(int number_of_points,
    double lower_limit,
    double upper_limit,
    VectorXd& integration_nodes,
    VectorXd& integration_weights) {
    if (number_of_points <= 0) {
        throw std::invalid_argument("Number of integration points must be a positive integer");
    }

    // 处理已知的低阶积分点配置
    if (number_of_points == 1) {
        integration_nodes = VectorXd::Zero(1);
        integration_weights = VectorXd::Ones(1);
        integration_nodes(0) = 0.0;
        integration_weights(0) = 2.0;
    }
    else if (number_of_points == 2) {
        integration_nodes = VectorXd::Zero(2);
        integration_weights = VectorXd::Ones(2);
        integration_nodes << -0.5773502691896257, 0.5773502691896257;
        integration_weights << 1.0, 1.0;
    }
    else if (number_of_points == 3) {
        integration_nodes = VectorXd::Zero(3);
        integration_weights = VectorXd::Zero(3);
        integration_nodes << -0.7745966692414834, 0.0, 0.7745966692414834;
        integration_weights << 0.5555555555555556, 0.8888888888888888, 0.5555555555555556;
    }
    else if (number_of_points == 4) {
        integration_nodes = VectorXd::Zero(4);
        integration_weights = VectorXd::Zero(4);
        integration_nodes << -0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526;
        integration_weights << 0.3478548451374539, 0.6521451548625461, 0.6521451548625461, 0.3478548451374539;
    }
    else if (number_of_points == 5) {
        integration_nodes = VectorXd::Zero(5);
        integration_weights = VectorXd::Zero(5);
        integration_nodes << -0.9061798459386640, -0.5384693101056831, 0.0, 0.5384693101056831, 0.9061798459386640;
        integration_weights << 0.2369268850561891, 0.4786286704993665, 0.5688888888888889, 0.4786286704993665, 0.2369268850561891;
    }
    else if (number_of_points == 6) {
        integration_nodes = VectorXd::Zero(6);
        integration_weights = VectorXd::Zero(6);
        integration_nodes << -0.9324695142031521, -0.6612093864662645, -0.2386191860831969,
            0.2386191860831969, 0.6612093864662645, 0.9324695142031521;
        integration_weights << 0.1713244923791704, 0.3607615730481386, 0.4679139345726910,
            0.4679139345726910, 0.3607615730481386, 0.1713244923791704;
    }
    else {
        // 处理高阶积分点计算
        ComputeHighOrderNodesAndWeights(number_of_points, integration_nodes, integration_weights);
    }

    // 执行区间变换
    TransformToTargetInterval(lower_limit, upper_limit, integration_nodes, integration_weights);
}

void GaussianQuadrature::ComputeHighOrderNodesAndWeights(int number_of_points,
    VectorXd& integration_nodes,
    VectorXd& integration_weights) {
    // 构造雅可比矩阵的非对角线元素
    VectorXd off_diagonal_elements(number_of_points - 1);
    for (int position = 0; position < number_of_points - 1; ++position) {
        const double term_index = position + 1;  // 从1开始计数
        off_diagonal_elements(position) = term_index / sqrt(4.0 * term_index * term_index - 1.0);
    }

    // 构建三对角对称矩阵
    MatrixXd jacobi_matrix = MatrixXd::Zero(number_of_points, number_of_points);
    for (int row = 0; row < number_of_points - 1; ++row) {
        jacobi_matrix(row, row + 1) = off_diagonal_elements(row);  // 上对角线
        jacobi_matrix(row + 1, row) = off_diagonal_elements(row);  // 下对角线
    }

    // 执行特征值分解
    SelfAdjointEigenSolver<MatrixXd> eigen_solver(jacobi_matrix);
    if (eigen_solver.info() != Success) {
        throw std::runtime_error("特征值计算未收敛，请检查输入参数");
    }

    // 获取计算结果
    VectorXd eigenvalues = eigen_solver.eigenvalues();
    MatrixXd eigenvectors = eigen_solver.eigenvectors();

    // 对特征值进行排序
    std::vector<size_t> sorted_indices(number_of_points);
    std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
    std::sort(sorted_indices.begin(), sorted_indices.end(),
        [&eigenvalues](size_t index1, size_t index2) {
            return eigenvalues(index1) < eigenvalues(index2);
        });

    // 存储排序后的结果
    integration_nodes.resize(number_of_points);
    integration_weights.resize(number_of_points);
    for (int order = 0; order < number_of_points; ++order) {
        const size_t original_index = sorted_indices[order];
        integration_nodes(order) = eigenvalues(original_index);
        integration_weights(order) = 2.0 * eigenvectors(0, original_index) * eigenvectors(0, original_index);
    }
}

void GaussianQuadrature::TransformToTargetInterval(double interval_start,
    double interval_end,
    VectorXd& integration_nodes,
    VectorXd& integration_weights) {
    const double scaling_factor = (interval_end - interval_start) / 2.0;
    const double offset = (interval_start + interval_end) / 2.0;

    // 调整节点位置
    for (int i = 0; i < integration_nodes.size(); ++i) {
        integration_nodes(i) = scaling_factor * integration_nodes(i) + offset;
    }

    // 调整权重系数
    integration_weights *= scaling_factor;
}