#include <Eigen/Dense>
#include <cmath>
#define M_PI       3.14159265358979323846

using namespace Eigen;

MatrixXd GenerateMutis1(const MatrixXd& x, const MatrixXd& y, int order, MatrixXd& mutis_x, MatrixXd& mutis_y) {
    int N = (1 + order) * order / 2;
    mutis_x = MatrixXd::Zero(x.rows(), N);
    mutis_y = MatrixXd::Zero(x.rows(), N);
    MatrixXd mutis = MatrixXd::Ones(x.rows(), N);

    int iter = 0;
    for (int k = 1; k <= order - 1; ++k) {
        // Update mutis matrix
        mutis.block(0, iter + 1, x.rows(), k) =
            mutis.block(0, iter - k + 1, x.rows(), k)
            .cwiseProduct(
                x.replicate(1, k)        // 把 x 横向复制 k 列
            );
        mutis.col(iter + k + 1) = mutis.col(iter).cwiseProduct(y);
        iter += k + 1;
    }
    iter = 0;
    int lastiter = -1;
    for (int k = 1; k <= order - 1; ++k) {

        // Update derivatives
        VectorXd coeff(k);
        for (int i = 0; i < k; ++i) coeff(i) = k - i;
        MatrixXd coeff_matrix_x = coeff.transpose().replicate(x.rows(), 1);
        mutis_x.block(0, iter + 1, x.rows(), k) =
            mutis.block(0, lastiter + 1, x.rows(), k).cwiseProduct(coeff_matrix_x);

        VectorXd coeff_y(k);
        for (int i = 0; i < k; ++i) coeff_y(i) = i + 1;
        MatrixXd coeff_matrix_y = coeff_y.transpose().replicate(x.rows(), 1);
        mutis_y.block(0, iter + 2, x.rows(), k) =
            mutis.block(0, lastiter + 1, x.rows(), k).cwiseProduct(coeff_matrix_y);
        lastiter = iter;
        iter += k + 1;
    }
    return mutis;
}

void GenerateOffsetMutis(const MatrixXd& points, int order, double x0, double y0, double xh, double yh,
    MatrixXd& Mutis, MatrixXd& Mutis_x, MatrixXd& Mutis_y) {
    VectorXd x = (points.col(0).array() - x0) / xh;
    VectorXd y = (points.col(1).array() - y0) / yh;

    Mutis = GenerateMutis1(x, y, order, Mutis_x, Mutis_y);
    Mutis_x /= xh;
    Mutis_y /= yh;
}


MatrixXd Distance_Function(const VectorXd& x, const VectorXd& y, double a, double b, double c) {
    // 使用.array()将VectorXd转换为数组表达式，然后进行逐元素运算
    VectorXd numerator = a * x.array() + b * y.array() + c;
    numerator = (numerator.array() < 0).select(0, numerator);
    double denominator = std::sqrt(a * a + b * b);
    return numerator.array().abs() / denominator;
}

MatrixXd Distance_Function_Derivatives(const VectorXd& x, const VectorXd& y, double a, double b, double c, int xOry) {
    // 使用.array()进行逐元素运算
    VectorXd dn = a * x.array() + b * y.array() + c;
    double denominator = std::sqrt(a * a + b * b);

    VectorXd numer;
    if (xOry == 0) numer = VectorXd::Constant(x.size(), a);
    else numer = VectorXd::Constant(x.size(), b);

    numer = (dn.array() >= 0).select(numer, 0);
    return numer.array() / denominator;
}

void GenerateOffsetDMutis(const MatrixXd& points, const MatrixXd& interpolate_points, int order,
    double x0, double y0, double xh, double yh,
    MatrixXd& DMutis, MatrixXd& DMutis_x, MatrixXd& DMutis_y, MatrixXd& DMutis_Cof) {
    double a = std::tan(M_PI / 6.0);
    double b = -1.0;
    double c = 1.0 - (1.0 + 1.0 / M_PI) * std::tan(M_PI / 6.0);

    MatrixXd d = Distance_Function(points.col(0), points.col(1), a, b, c);
    MatrixXd d_x = Distance_Function_Derivatives(points.col(0), points.col(1), a, b, c, 0);
    MatrixXd d_y = Distance_Function_Derivatives(points.col(0), points.col(1), a, b, c, 1);

    MatrixXd mutis, mutis_x_tmp, mutis_y_tmp;
    GenerateOffsetMutis(points, order, x0, y0, xh, yh, mutis, mutis_x_tmp, mutis_y_tmp);

    // DMutis = mutis .* d; 其中d是列向量
    DMutis = mutis.cwiseProduct(d.replicate(1, mutis.cols()));
    DMutis_x = mutis_x_tmp.cwiseProduct(d.replicate(1, mutis_x_tmp.cols())) +
        mutis.cwiseProduct(d_x.replicate(1, mutis.cols()));
    DMutis_y = mutis_y_tmp.cwiseProduct(d.replicate(1, mutis_y_tmp.cols())) +
        mutis.cwiseProduct(d_y.replicate(1, mutis.cols()));

    MatrixXd d_inter = Distance_Function(interpolate_points.col(0), interpolate_points.col(1), a, b, c);
    GenerateOffsetMutis(interpolate_points, order, x0, y0, xh, yh, mutis, mutis_x_tmp, mutis_y_tmp);
    DMutis_Cof = mutis.cwiseProduct(d_inter.replicate(1, mutis.cols()));
}
