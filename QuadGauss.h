#include <Eigen/Dense>
#include <cmath>
#include <algorithm>
#include <tuple>
#include <numeric>
using namespace Eigen;

std::tuple<VectorXd, VectorXd> QuadGauss(int n, double a = -1.0, double b = 1.0) {
    VectorXd x, w;

    // 处理默认参数
    if (n < 1) {
        throw std::invalid_argument("n must be positive");
    }

    // 预定义的高斯积分点和权重
    switch (n) {
    case 1:
        x = VectorXd::Zero(1);
        w = VectorXd::Constant(1, 2.0);
        break;
    case 2:
        x.resize(2);
        x << -0.5773502691896257, 0.5773502691896257;
        w = VectorXd::Constant(2, 1.0);
        break;
    case 3:
        x.resize(3);
        x << -0.7745966692414834, 0.0, 0.7745966692414834;
        w.resize(3);
        w << 0.5555555555555556, 0.8888888888888888, 0.5555555555555556;
        break;
    case 4:
        x.resize(4);
        x << -0.8611363115940526, -0.3399810435848563,
            0.3399810435848563, 0.8611363115940526;
        w.resize(4);
        w << 0.3478548451374539, 0.6521451548625461,
            0.6521451548625461, 0.3478548451374539;
        break;
    case 5:
        x.resize(5);
        x << -0.9061798459386640, -0.5384693101056831, 0.0,
            0.5384693101056831, 0.9061798459386640;
        w.resize(5);
        w << 0.2369268850561891, 0.4786286704993665,
            0.5688888888888889, 0.4786286704993665,
            0.2369268850561891;
        break;
    case 6:
        x.resize(6);
        x << -0.9324695142031521, -0.6612093864662645,
            -0.2386191860831969, 0.2386191860831969,
            0.6612093864662645, 0.9324695142031521;
        w.resize(6);
        w << 0.1713244923791704, 0.3607615730481386,
            0.4679139345726910, 0.4679139345726910,
            0.3607615730481386, 0.1713244923791704;
        break;
    default: {
        // 使用 Golub-Welsch 算法计算高阶高斯点
        MatrixXd A = MatrixXd::Zero(n, n);
        VectorXd u = VectorXd::LinSpaced(n - 1, 1, n - 1);
        u = u.array() / (4 * u.array().square() - 1).sqrt();

        // 填充三对角矩阵
        for (int i = 0; i < n - 1; ++i) {
            A(i + 1, i) = u(i);
            A(i, i + 1) = u(i);
        }

        // 计算特征值和特征向量
        SelfAdjointEigenSolver<MatrixXd> eigensolver(A);
        if (eigensolver.info() != Success) {
            throw std::runtime_error("Eigen decomposition failed");
        }

        // 获取特征值和特征向量
        x = eigensolver.eigenvalues();
        MatrixXd v = eigensolver.eigenvectors();

        // 排序特征值
        std::vector<int> indices(n);
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(),
            [&x](int i1, int i2) { return x(i1) < x(i2); });

        // 重新排序并计算权重
        VectorXd sorted_x(n);
        w.resize(n);
        for (int i = 0; i < n; ++i) {
            sorted_x(i) = x(indices[i]);
            w(i) = 2.0 * v(0, indices[i]) * v(0, indices[i]);
        }
        x = sorted_x;
        break;
    }
    }

    // 线性变换到区间 [a, b]
    x = (b - a) / 2.0 * x.array() + (a + b) / 2.0;
    w = (b - a) / 2.0 * w.array();

    return { x, w };
}

std::tuple<MatrixXd, VectorXd, MatrixXd, VectorXd>
calc_intbdry_gps(double (*intfacex)(double), double (*intfacey)(double),
    int i, int j, int N, int gpn, int bdry_type) {

    double h = 1.0 / N;
    double x0 = (j - 1) * h;
    double y0 = (i - 1) * h;

    MatrixXd bdryGps0, bdryGps1;
    VectorXd bdryGpW0, bdryGpW1;

    if (bdry_type == 0) {
        double y = intfacex(0);

        // 获取第一段高斯点和权重
        auto result0 = QuadGauss(gpn, y0, y);
        VectorXd x0_pts = std::get<0>(result0);
        VectorXd weights0 = std::get<1>(result0);

        bdryGps0 = MatrixXd::Zero(gpn, 2);
        bdryGps0.col(1) = x0_pts;
        bdryGpW0 = weights0;

        // 获取第二段高斯点和权重
        auto result1 = QuadGauss(gpn, y, y0 + h);
        VectorXd x1_pts = std::get<0>(result1);
        VectorXd weights1 = std::get<1>(result1);

        bdryGps1 = MatrixXd::Zero(gpn, 2);
        bdryGps1.col(1) = x1_pts;
        bdryGpW1 = weights1;
    }
    else if (bdry_type == 2) {
        double y = intfacex(1);

        // 获取第一段高斯点和权重
        auto result0 = QuadGauss(gpn, y0, y);
        VectorXd x0_pts = std::get<0>(result0);
        VectorXd weights0 = std::get<1>(result0);

        bdryGps0 = MatrixXd::Ones(gpn, 2);
        bdryGps0.col(1) = x0_pts;
        bdryGpW0 = weights0;

        // 获取第二段高斯点和权重
        auto result1 = QuadGauss(gpn, y, y0 + h);
        VectorXd x1_pts = std::get<0>(result1);
        VectorXd weights1 = std::get<1>(result1);

        bdryGps1 = MatrixXd::Ones(gpn, 2);
        bdryGps1.col(1) = x1_pts;
        bdryGpW1 = weights1;
    }
    else {
        throw std::invalid_argument("Unsupported boundary type");
    }

    return { bdryGps0, bdryGpW0, bdryGps1, bdryGpW1 };
}
